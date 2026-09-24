"""CPU-only student input preparation. Produces evidence, never a training job."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import tokenizers

from gitctx.proof_lm_train import _iter_jsonl, _load_json, _sha256, _validate_identifier
from gitctx.proof_order_ablation import leakage_keys, load_protocol
from gitctx.provenance import APPROVED_SOURCE_LICENSES
from gitctx.student_input import PROMPT_VERSION, review_provenance, student_messages
from gitctx.student_sequences import ANSWER_RESERVE, CONTEXT, inspect_budget, materialize
from gitctx.student_tokenizer import SPECIAL, VERSION, StudentTokenizer

CODE_FILES = ("student_input.py", "student_tokenizer.py", "student_sequences.py",
              "student_readiness.py", "train_artifacts.py", "provenance.py")
SYNTHETIC = (
    "", " \t\r\n  \n", "def f():\n\treturn 'İstanbul 🚀 日本語 é é'\r\n",
    "packages/@scope/a-b.test.ts C:\\src\\a.py /foo/bar.py",
    "<bos><eos><pad><unk><system><user><assistant><sep><nl>",
    "".join(chr(i) for i in range(256)),
)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_inputs(data, parent, findings_path):
    folder, protocol = load_protocol(data, parent)
    job = _load_json(data / protocol["source_job"]["path"])
    source_path = data / job["inputs"]["training_artifact"]["path"]
    rows = [r for r in _iter_jsonl(source_path) if r["data_split"] == "DEV"]
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate DEV IDs")
    train, validation = protocol["arms"]["ordered"], protocol["validation_ids"]
    if (len(set(train)) != len(train) or len(set(validation)) != len(validation)
            or set(train) & set(validation) or not set(train + validation) <= by_id.keys()):
        raise ValueError("invalid frozen split IDs")
    train_repos = {by_id[i]["source_repo_url"] for i in train}
    val_repos = {by_id[i]["source_repo_url"] for i in validation}
    keys = {k for i in train for k in leakage_keys(by_id[i])}
    if train_repos & val_repos or any(keys & leakage_keys(by_id[i]) for i in validation):
        raise ValueError("training/validation leakage")
    plans = {r["record_id"]: r for r in _iter_jsonl(data / job["inputs"]["sequence_plan"]["path"])}
    findings = list(_iter_jsonl(findings_path))
    for f in findings:
        r = by_id[f["record_id"]]
        if f["source_diff_sha256"] != r["diff_sha256"]:
            raise ValueError("finding points to a different source diff")
        expected = "train" if r["id"] in set(train) else "validation"
        if f["partition"] != expected or r["id"] not in set(train + validation):
            raise ValueError("finding partition mismatch")
    return rows, protocol, plans, findings, source_path


def partition(record_id, train_ids, validation_ids, plans):
    if record_id in train_ids:
        return "train"
    if record_id in validation_ids:
        return "validation"
    if plans[record_id]["decision"] == "exclude_oversize":
        return "legacy_excluded"
    return "reserved_dev"


def corpus(rows, train_ids):
    by_id = {r["id"]: r for r in rows}
    for record_id in train_ids:
        record = by_id[record_id]
        yield from (m["content"] for m in student_messages(record))
        yield record["target_message"]


def inspect(rows, protocol, plans, findings, tokenizer):
    train, validation = set(protocol["arms"]["ordered"]), set(protocol["validation_ids"])
    counters, budgets, provenance, queue = {}, [], [], []
    failures = []
    for value in SYNTHETIC:
        if tokenizer.decode(tokenizer.encode(value)) != value:
            failures.append("synthetic round-trip")
    for index, record in enumerate(rows):
        if index % 1000 == 0:
            print(f"Audit {index}/{len(rows)}", flush=True)
        rid = record["id"]
        part = partition(rid, train, validation, plans)
        counts = counters.setdefault(part, Counter())
        counts["records"] += 1
        values = [m["content"] for m in student_messages(record)]
        values += [record["diff"], record["target_message"]]
        roundtrip = all(tokenizer.decode(tokenizer.encode(v)) == v for v in values)
        if not roundtrip:
            failures.append(f"round-trip: {rid}")
        counts["roundtrip_records"] += int(roundtrip)
        budget = inspect_budget(record, tokenizer)
        budget["partition"] = part
        budgets.append(budget)
        for key in ("prompt_overflow", "target_overflow", "prompt_tokens",
                    "target_tokens_including_eos", "changed_lines", "removed_changed_lines"):
            counts[key] += budget[key]
        counts["fits_full"] += budget["decision"] == "fits_full"
        counts["max_prompt_tokens"] = max(counts["max_prompt_tokens"], budget["prompt_tokens"])
        counts["max_target_tokens_including_eos"] = max(
            counts["max_target_tokens_including_eos"], budget["target_tokens_including_eos"])
        if budget["diff_sha256"] != record["diff_sha256"]:
            failures.append(f"diff integrity: {rid}")
        if (record["source_license"] not in APPROVED_SOURCE_LICENSES
                or record["teacher_license"] != "Apache-2.0"
                or record["teacher_model_id"] != "ollama/qwen2.5-coder:7b"
                or not record.get("teacher_revision") or not record.get("prompt_version")):
            failures.append(f"license/provenance outside reviewed recipe: {rid}")
        p = review_provenance(record)
        p.update(record_id=rid, partition=part, original_target_sha256=stable_hash(record["target_message"]),
                 original_record_sha256=stable_hash(record), source_diff_sha256=record["diff_sha256"],
                 generated_label_id=record["generated_label_id"],
                 generated_label_review_id=record["generated_label_review_id"],
                 teacher_model_id=record["teacher_model_id"], teacher_revision=record["teacher_revision"],
                 teacher_license=record["teacher_license"], source_license=record["source_license"],
                 teacher_prompt_version=record["prompt_version"])
        provenance.append(p)
        counts["independently_reviewed"] += p["independent_factuality_review"]
        # The actual materializer must reproduce inference's prompt exactly.
        if budget["decision"] == "fits_full":
            seq = materialize(record, tokenizer)
            if len(seq["input_ids"]) > CONTEXT or sum(seq["loss_mask"]) != len(
                    tokenizer.encode(record["target_message"])) + 1:
                failures.append(f"sequence contract: {rid}")
    for f in findings:
        if f["status"] == "no_definite_error_seen_in_screen":
            continue
        queue.append({**f, "status": "pending_independent_review", "screen_status": f["status"],
                      "correction": None, "training_use": "prohibited_as_target",
                      "resolution": "licensed_open_teacher_and_verification_or_explicit_human_correction"})
    active_overflow = sum(counters[p]["prompt_overflow"] + counters[p]["target_overflow"]
                          for p in ("train", "validation"))
    blockers = list(failures)
    if active_overflow:
        blockers.append("active_split_overflow_requires_evidence_and_answer_policy")
    if counters.get("legacy_excluded", {}).get("records", 0):
        blockers.append("legacy_excluded_ids_require_explicit_coverage_decision")
    if queue:
        blockers.append("flagged_labels_require_independent_resolution")
    if counters["validation"]["independently_reviewed"] < len(validation):
        blockers.append("validation_references_require_independent_review")
    report = {
        "preparation_complete": not failures, "training_ready": not blockers,
        "training_launched": False, "blockers": blockers, "technical_failures": failures,
        "counts": {k: dict(v) for k, v in counters.items()},
        "records_audited": len(rows), "roundtrip_records": sum(c["roundtrip_records"] for c in counters.values()),
        "unknown_content_tokens": 0, "synthetic_roundtrip_cases": len(SYNTHETIC),
        "vocab_size": tokenizer.vocab_size, "review_queue_records": len(queue),
        "review_queue_status_counts": dict(Counter(f["screen_status"] for f in queue)),
        "prompt_version": PROMPT_VERSION, "tokenizer_version": VERSION,
        "context_tokens": CONTEXT, "answer_reserve_including_eos": ANSWER_RESERVE,
        "targets_modified": 0, "records_silently_dropped": 0,
        "report_or_held_out_used": False,
        "claim": "CPU input preparation only; no model quality or factuality claim",
    }
    return report, budgets, provenance, queue


def prepare(data: Path, parent: str, run: str, findings_path: Path):
    _validate_identifier(run, "run")
    folder = data / "artifacts/train-runs" / run
    if folder.exists():
        raise ValueError("output already exists; choose a new version")
    rows, protocol, plans, findings, source_path = load_inputs(data, parent, findings_path)
    print(f"Fitting byte BPE on {len(protocol['arms']['ordered'])} frozen training IDs", flush=True)
    tokenizer = StudentTokenizer.fit(corpus(rows, protocol["arms"]["ordered"]))
    if tokenizer.vocab_size != 32000:
        raise ValueError("32K vocabulary required; do not change model size silently")
    print(f"Auditing {len(rows)} DEV records, including reserved and previously excluded IDs", flush=True)
    report, budgets, provenance, queue = inspect(rows, protocol, plans, findings, tokenizer)
    folder.mkdir(parents=True)
    tokenizer.save(folder / "tokenizer.json")
    dump(folder / "readiness.json", report)
    for name, values in (("coverage", budgets), ("provenance", provenance), ("review-queue", queue)):
        with (folder / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for value in values:
                f.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
    inputs = [source_path, data / "artifacts/train-runs" / parent / "protocol.json", findings_path]
    manifest = {
        "run_id": run, "parent": parent, "findings_path": str(findings_path.relative_to(data)),
        "inputs": {str(p.relative_to(data)): _sha256(p) for p in inputs},
        "code": {n: _sha256(Path(__file__).with_name(n)) for n in CODE_FILES},
        "uv_lock_sha256": _sha256(Path(__file__).parents[2] / "uv.lock"),
        "tokenizers_version": tokenizers.__version__, "special_tokens": SPECIAL,
        "tokenizer_fit_ids": protocol["arms"]["ordered"],
        "model_contract": protocol["model_contract"], "resume_existing_checkpoint": False,
        "outputs": {p.name: _sha256(p) for p in sorted(folder.iterdir())},
    }
    dump(folder / "manifest.json", manifest)
    print(json.dumps(report, indent=2), flush=True)
    return report


def verify(data: Path, run: str):
    _validate_identifier(run, "run")
    folder = data / "artifacts/train-runs" / run
    manifest = _load_json(folder / "manifest.json")
    for name, digest in manifest["inputs"].items():
        if _sha256(data / name) != digest:
            raise ValueError(f"input hash mismatch: {name}")
    for name, digest in manifest["outputs"].items():
        if _sha256(folder / name) != digest:
            raise ValueError(f"output hash mismatch: {name}")
    for name, digest in manifest["code"].items():
        if _sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"implementation hash mismatch: {name}")
    if (tokenizers.__version__ != manifest["tokenizers_version"]
            or _sha256(Path(__file__).parents[2] / "uv.lock") != manifest["uv_lock_sha256"]):
        raise ValueError("runtime lock mismatch")
    rows, protocol, plans, findings, _ = load_inputs(data, manifest["parent"], data / manifest["findings_path"])
    if (manifest["tokenizer_fit_ids"] != protocol["arms"]["ordered"]
            or manifest["model_contract"] != protocol["model_contract"]):
        raise ValueError("tokenizer fit IDs or model contract differ from frozen protocol")
    actual = inspect(rows, protocol, plans, findings, StudentTokenizer.load(folder / "tokenizer.json"))
    expected = (_load_json(folder / "readiness.json"), *(
        list(_iter_jsonl(folder / f"{name}.jsonl")) for name in ("coverage", "provenance", "review-queue")))
    if actual != expected:
        raise ValueError("recomputed preparation differs from saved outputs")
    print(json.dumps({"verified": True, "records": len(rows), "training_ready": actual[0]["training_ready"]}))
    return actual[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--parent")
    parser.add_argument("--findings", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    if args.command == "prepare":
        if not args.parent or not args.findings:
            parser.error("prepare requires --parent and --findings (relative to data-dir)")
        report = prepare(data, args.parent, args.run, data / args.findings)
    else:
        report = verify(data, args.run)
    if not report["preparation_complete"] or (args.require_ready and not report["training_ready"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
