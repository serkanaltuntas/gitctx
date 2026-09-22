"""DEV-only memorization diagnostic using the unchanged proof model and loss.

This deliberately tests seen examples, not generalization or release quality.
The protocol is frozen before training, and all output lives in a separate run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from gitctx.conventional import parse_commit_message
from gitctx.proof_lm_eval import decode_tokens, generate, prompt_tokens
from gitctx.proof_lm_train import (
    _atomic_json, _batch_for_torch, _build_model, _configure_device_runtime,
    _iter_jsonl, _job_blockers, _load_json, _runtime_info, _seed_torch, _sha256,
    _sequence_matches_metadata, _stable_sha256, _validate_identifier,
)
from gitctx.proof_sequences import materialize_training_sequence
from gitctx.proof_tokenizer import tokenize_text

IMPLEMENTATIONS = ("proof_memorization.py", "proof_lm_train.py", "proof_lm_eval.py",
                   "proof_sequences.py", "proof_tokenizer.py", "conventional.py")


def implementation_hashes():
    return {name: _sha256(Path(__file__).with_name(name)) for name in IMPLEMENTATIONS}


def select_records(records, metadata, vocabulary, quotas, *, min_length=256,
                   max_length=1024, max_target_tokens=96):
    """Choose distinct repositories deterministically, without any model scores."""
    candidates = []
    for record in records:
        if record.get("data_split") != "DEV":
            continue
        meta = metadata.get(record["id"])
        if not meta or meta["decision"] != "use_full":
            continue
        if not min_length <= meta["input_length"] <= max_length:
            continue
        target = record.get("target_message") or next(
            m["content"] for m in record["messages"] if m["role"] == "assistant")
        tokens = tokenize_text(target)
        if not 8 <= len(tokens) <= max_target_tokens or any(t not in vocabulary for t in tokens):
            continue
        parsed = parse_commit_message(target)
        candidates.append((record, parsed.type))
    candidates.sort(key=lambda item: hashlib.sha256(item[0]["id"].encode()).hexdigest())
    selected, used_repos = [], set()
    for kind, count in quotas.items():
        picked = 0
        for record, record_type in candidates:
            if record_type == kind and record["source_repo_url"] not in used_repos:
                selected.append(record)
                used_repos.add(record["source_repo_url"])
                picked += 1
                if picked == count:
                    break
        if picked != count:
            raise ValueError(f"not enough distinct DEV repositories for {kind}: {picked}/{count}")
    return selected


def prepare(data_dir: Path, job_path: Path, run_id: str):
    _validate_identifier(run_id, "run_id")
    folder = data_dir / "artifacts/train-runs" / run_id
    if folder.exists():
        raise ValueError("diagnostic run already exists")
    job_path = data_dir / job_path
    job = _load_json(job_path)
    blockers = _job_blockers(data_dir, job_path=job_path, job=job)
    if blockers:
        raise ValueError(blockers)
    tokenizer = _load_json(data_dir / job["inputs"]["tokenizer"]["path"])
    vocabulary = {v["token"]: v["id"] for v in tokenizer["vocab"]}
    metadata = {m["record_id"]: m for m in _iter_jsonl(
        data_dir / job["inputs"]["sequence_metadata"]["path"]) if m["data_split"] == "DEV"}
    quotas = {"feat": 4, "fix": 4, "docs": 3, "chore": 3, "refactor": 2}
    selected = select_records(_iter_jsonl(data_dir / job["inputs"]["training_artifact"]["path"]),
                              metadata, vocabulary, quotas)
    protocol = {
        "run_id": run_id, "kind": "DEV-only seen-example memorization diagnostic",
        "source_job": {"path": job_path.relative_to(data_dir).as_posix(), "sha256": _sha256(job_path)},
        "model_contract": job["model_contract"], "input_hashes": job["inputs"],
        "implementation_hashes": implementation_hashes(),
        "selection": {"quotas": quotas, "distinct_repositories": True,
                      "order": "SHA256(record id), quota order", "input_length_range": [256, 1024],
                      "target_tokens_range": [8, 96], "target_unknown_tokens": 0,
                      "sequence_decision": "use_full", "selection_uses_model_outputs": False},
        "records": [{"record_id": r["id"], "repository": r["source_repo_url"],
                     "type": parse_commit_message(r["target_message"]).type,
                     "scope": parse_commit_message(r["target_message"]).scope,
                     "input_tokens": metadata[r["id"]]["input_length"],
                     "target_sha256": hashlib.sha256(r["target_message"].encode()).hexdigest()}
                    for r in selected],
        "training": {"seed": 17, "initialization": "fresh normal std=0.02", "precision": "float32",
                     "optimizer": "AdamW", "learning_rate": 0.0003, "batch_size": 1,
                     "order": "shuffle each epoch with Random(seed + epoch)",
                     "max_epochs": 100, "evaluate_every": 10,
                     "attention_chunk_size": 256, "activation_checkpointing": True},
        "evaluation": {"split": "same selected DEV examples", "decode": "greedy",
                       "max_new_tokens": 128, "prompt": "same system/user-only production helper",
                       "teacher_forced_tokens": "assistant plus terminal tokens only",
                       "pass": "all selected target token sequences and parsed type/scope match; CE <= 0.1",
                       "stop": "first passing evaluation or 100 epochs, whichever is earlier"},
        "limitations": ["Intentionally easy, short, untruncated seen examples with known target tokens.",
                        "The full context limit is preserved; this is not a long-input quality test.",
                        "Passing proves memorization only, not factuality or generalization.",
                        "REPORT and HELD_OUT are not trained on, generated on, or used for selection."]}
    folder.mkdir(parents=True)
    _atomic_json(folder / "protocol.json", protocol)
    return protocol


def load_examples(data_dir: Path, protocol):
    job_path = data_dir / protocol["source_job"]["path"]
    if _sha256(job_path) != protocol["source_job"]["sha256"]:
        raise ValueError("source job changed")
    job = _load_json(job_path)
    blockers = _job_blockers(data_dir, job_path=job_path, job=job)
    if blockers:
        raise ValueError(blockers)
    if protocol["model_contract"] != job["model_contract"]:
        raise ValueError("model contract changed")
    selected = {r["record_id"] for r in protocol["records"]}
    if not selected or len(selected) != len(protocol["records"]):
        raise ValueError("empty or duplicate selection")
    records = {r["id"]: r for r in _iter_jsonl(data_dir / job["inputs"]["training_artifact"]["path"])
               if r["id"] in selected}
    if records.keys() != selected or any(r["data_split"] != "DEV" for r in records.values()):
        raise ValueError("selection must consist entirely of existing DEV records")
    tokenizer = _load_json(data_dir / job["inputs"]["tokenizer"]["path"])
    vocabulary = {v["token"]: v["id"] for v in tokenizer["vocab"]}
    plans = {r["record_id"]: r for r in _iter_jsonl(data_dir / job["inputs"]["sequence_plan"]["path"])
             if r["record_id"] in selected}
    metadata = {r["record_id"]: r for r in _iter_jsonl(data_dir / job["inputs"]["sequence_metadata"]["path"])
                if r["record_id"] in selected}
    examples = []
    for entry in protocol["records"]:
        record = records[entry["record_id"]]
        sequence = materialize_training_sequence(record, plans[record["id"]], tokenizer,
                         context_tokens=protocol["model_contract"]["context_tokens"])
        if not _sequence_matches_metadata(sequence, metadata[record["id"]]):
            raise ValueError("sequence metadata mismatch")
        target = record.get("target_message") or next(
            m["content"] for m in record["messages"] if m["role"] == "assistant")
        tokens = tokenize_text(target)
        if any(t not in vocabulary for t in tokens):
            raise ValueError("unknown target tokens make exact memorization ambiguous")
        prompt = prompt_tokens(record, context_tokens=protocol["model_contract"]["context_tokens"],
                               max_new_tokens=protocol["evaluation"]["max_new_tokens"])
        prompt_ids = [vocabulary.get(t, vocabulary["<unk>"]) for t in prompt]
        first = sequence["loss_mask"].index(1)
        if sequence["input_ids"][:first] != prompt_ids:
            raise ValueError("training and generation prompt mismatch")
        target_ids = [vocabulary[t] for t in tokens]
        if sequence["input_ids"][first:first + len(target_ids)] != target_ids:
            raise ValueError("supervised target alignment mismatch")
        examples.append({"record": record, "sequence": sequence, "prompt_ids": prompt_ids,
                         "target_ids": target_ids, "target": target})
    if len({tuple(e["prompt_ids"]) for e in examples}) != len(examples):
        raise ValueError("duplicate generation prompts")
    return examples, vocabulary


def evaluate_examples(torch, model, examples, vocabulary, *, device, max_new_tokens):
    model.eval()
    reverse = {index: token for token, index in vocabulary.items()}
    loss_sum, token_count, correct = 0.0, 0, 0
    predictions = []
    for example in examples:
        batch = _batch_for_torch(torch, [example["sequence"]], device=device)
        with torch.inference_mode():
            logits = model(batch["input_ids"], batch["attention_mask"])
            active = batch["labels"] != -100
            logits = logits[active]
            labels = batch["labels"][active]
            loss_sum += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum").cpu())
            correct += int((logits.argmax(-1) == labels).sum().cpu())
            token_count += labels.numel()
        del logits, labels, batch
        output, reason = generate(torch, model, example["prompt_ids"], device=device,
            max_new_tokens=max_new_tokens, stop_ids={vocabulary["<sep>"], vocabulary["<eos>"]})
        message = decode_tokens([reverse[i] for i in output])
        expected = parse_commit_message(example["target"])
        try:
            parsed = parse_commit_message(message)
            type_match, scope_match = parsed.type == expected.type, parsed.scope == expected.scope
        except ValueError:
            type_match = scope_match = False
        predictions.append({"record_id": example["record"]["id"], "data_split": "DEV",
            "message": message, "target_message": example["target"], "output_token_ids": output,
            "target_token_ids": example["target_ids"], "stop_reason": reason,
            "prompt_sha256": _stable_sha256(example["prompt_ids"]),
            "exact_tokens": output == example["target_ids"] and reason == "stop_token",
            "exact_text": message == example["target"], "type_match": type_match, "scope_match": scope_match})
    counts = {key: sum(p[key] for p in predictions)
              for key in ("exact_tokens", "exact_text", "type_match", "scope_match")}
    summary = {"records": len(examples), "mean_loss": loss_sum / token_count,
               "supervised_tokens": token_count, "teacher_forced_correct_tokens": correct,
               "teacher_forced_accuracy": correct / token_count, **counts}
    summary["passed"] = (all(counts[k] == len(examples) for k in ("exact_tokens", "type_match", "scope_match"))
                         and summary["mean_loss"] <= 0.1)
    return summary, predictions


def run(data_dir: Path, run_id: str, *, device="cuda", resume=False):
    import torch
    _validate_identifier(run_id, "run_id")
    folder = data_dir / "artifacts/train-runs" / run_id
    protocol_path = folder / "protocol.json"
    protocol = _load_json(protocol_path)
    if protocol["run_id"] != run_id or protocol["implementation_hashes"] != implementation_hashes():
        raise ValueError("protocol or implementation mismatch")
    checkpoint_path = folder / "latest.pt"
    if checkpoint_path.exists() and not resume:
        raise ValueError("run has a checkpoint; resume explicitly or choose another run")
    if resume and not checkpoint_path.exists():
        raise ValueError("resume checkpoint missing")
    examples, vocabulary = load_examples(data_dir, protocol)
    train = protocol["training"]
    _configure_device_runtime(torch, device)
    _seed_torch(torch, train["seed"])
    model = _build_model(torch, protocol["model_contract"],
                        attention_chunk_size=train["attention_chunk_size"],
                        activation_checkpointing=train["activation_checkpointing"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["learning_rate"], foreach=False)
    identity = {"protocol_sha256": _sha256(protocol_path), "device": device, "torch": str(torch.__version__)}
    history, first_gradient_norm, start_epoch, prior_seconds = [], None, 0, 0.0
    if resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("checkpoint identity mismatch")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        start_epoch, history = saved["epoch"], saved["history"]
        first_gradient_norm = saved["first_gradient_norm"]
        prior_seconds = saved["elapsed_seconds"]
        torch.set_rng_state(saved["rng_cpu"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(saved["rng_cuda"])
        del saved
    runtime = _runtime_info(torch, device, model)
    expected_count = protocol["model_contract"].get("estimated_parameters")
    if expected_count is not None and runtime["parameter_count"] != expected_count:
        raise ValueError("parameter count mismatch")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    def measure(epoch):
        summary, predictions = evaluate_examples(torch, model, examples, vocabulary, device=device,
                                     max_new_tokens=protocol["evaluation"]["max_new_tokens"])
        summary.update(epoch=epoch, optimizer_steps=epoch * len(examples))
        history.append(summary)
        _atomic_json(folder / f"eval-{epoch:04d}.json", {"summary": summary, "predictions": predictions,
                                                      "identity": identity})
        print(json.dumps({"event": "evaluation", **summary}), flush=True)
        return summary

    def save(epoch, summary):
        elapsed = prior_seconds + time.monotonic() - started
        temp = checkpoint_path.with_suffix(".tmp.pt")
        torch.save({"identity": identity, "epoch": epoch, "history": history,
                    "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                    "first_gradient_norm": first_gradient_norm, "elapsed_seconds": elapsed}, temp)
        temp.replace(checkpoint_path)
        memory = {} if device != "cuda" else {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        report = {"run_id": run_id, "status": "passed" if summary["passed"] else
                  ("not_memorized" if epoch == train["max_epochs"] else "running"),
                  "identity": identity, "runtime": {**runtime, **memory}, "history": history,
                  "epochs": epoch, "optimizer_steps": epoch * len(examples),
                  "unique_dev_records": len(examples), "report_records_used": 0, "held_out_records_used": 0,
                  "first_gradient_norm": first_gradient_norm, "elapsed_seconds": elapsed,
                  "checkpoint_sha256": _sha256(checkpoint_path), "limitations": protocol["limitations"]}
        _atomic_json(folder / "report.json", report)
        return report

    if not resume:
        initial = measure(0)
        save(0, initial)
    elif history[-1]["passed"] or start_epoch >= train["max_epochs"]:
        return _load_json(folder / "report.json")
    for epoch in range(start_epoch + 1, train["max_epochs"] + 1):
        model.train()
        order = list(range(len(examples)))
        random.Random(train["seed"] + epoch).shuffle(order)
        weighted_loss, tokens = 0.0, 0
        for index in order:
            batch = _batch_for_torch(torch, [examples[index]["sequence"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            if first_gradient_norm is None:
                squared = sum(float(p.grad.detach().square().sum().cpu()) for p in model.parameters()
                              if p.grad is not None)
                first_gradient_norm = math.sqrt(squared)
                if not math.isfinite(first_gradient_norm) or first_gradient_norm <= 0:
                    raise ValueError("non-finite or zero initial gradient")
            optimizer.step()
            weighted_loss += float(loss.detach().cpu()) * batch["loss_tokens"]
            tokens += batch["loss_tokens"]
        print(json.dumps({"event": "epoch", "epoch": epoch, "steps": epoch * len(examples),
                          "online_mean_loss": weighted_loss / tokens,
                          "elapsed_seconds": prior_seconds + time.monotonic() - started}), flush=True)
        if epoch % train["evaluate_every"] == 0 or epoch == train["max_epochs"]:
            summary = measure(epoch)
            report = save(epoch, summary)
            if summary["passed"]:
                return report
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-job", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source_job is None:
            parser.error("prepare requires --source-job")
        result = prepare(args.data_dir, args.source_job, args.run_id)
    else:
        result = run(args.data_dir, args.run_id, device=args.device, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
