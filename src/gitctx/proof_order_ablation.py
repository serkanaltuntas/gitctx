"""Repository-separated DEV ablation: original order versus one fixed shuffle."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random

from gitctx.conventional import parse_commit_message
from gitctx.proof_lm_eval import DECODER_POLICY, decode_tokens, generate, prompt_tokens
from gitctx.proof_lm_train import (
    _atomic_json, _batch_for_torch, _build_model, _configure_device_runtime,
    _iter_jsonl, _job_blockers, _load_json, _sha256, _stable_sha256,
    _validate_identifier, run_proof_lm_training, validate_proof_lm_training,
)
from gitctx.proof_sequences import materialize_training_sequence, _sequence_metadata
from gitctx.proof_tokenizer import SPECIAL_TOKENS, record_tokens, tokenize_text

CODE_FILES = ("proof_order_ablation.py", "proof_lm_eval.py", "proof_lm_train.py",
              "proof_sequences.py", "proof_tokenizer.py", "conventional.py")


def code_hashes():
    return {name: _sha256(Path(__file__).with_name(name)) for name in CODE_FILES}


def leakage_keys(record):
    """Repository isolation plus exact commit, diff and whitespace-normalized diff checks."""
    return {("commit", record["source_commit"]), ("diff", record["diff_sha256"]),
            ("diff_tokens", _stable_sha256(tokenize_text(record["diff"]))) }


def split_dev(records, *, validation_repos=8, validation_records=256):
    if validation_repos < 1 or validation_records < 1:
        raise ValueError("validation sizes must be positive")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate record IDs")
    if any(r["data_split"] != "DEV" for r in records):
        raise ValueError("ablation input must be DEV only")
    grouped = {}
    for record in records:
        grouped.setdefault(record["source_repo_url"], []).append(record)
    ranked = sorted(grouped, key=lambda repo: hashlib.sha256(repo.encode()).hexdigest())
    if len(ranked) <= validation_repos:
        raise ValueError("not enough repositories to separate training and validation")
    reserved = set(ranked[:validation_repos])
    forbidden = set().union(*(leakage_keys(r) for repo in reserved for r in grouped[repo]))
    train, purged = [], []
    for record in records:
        if record["source_repo_url"] in reserved:
            continue
        (purged if leakage_keys(record) & forbidden else train).append(record)
    queues = {repo: sorted(grouped[repo], key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
              for repo in ranked[:validation_repos]}
    validation = []
    # Round robin avoids letting one large repository dominate this diagnostic.
    for index in range(max(map(len, queues.values()))):
        for repo in queues:
            if index < len(queues[repo]):
                validation.append(queues[repo][index])
                if len(validation) == validation_records:
                    return train, validation, sorted(reserved), purged
    raise ValueError("not enough validation records")


def fit_tokenizer(records, vocab_size):
    counts = Counter()
    for record in records:
        if record["data_split"] != "DEV":
            raise ValueError("tokenizer must fit DEV training only")
        counts.update(record_tokens(record))
    tokens = list(SPECIAL_TOKENS)
    tokens += [t for t, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
               if n >= 2 and t not in SPECIAL_TOKENS][:vocab_size-len(tokens)]
    if len(tokens) != vocab_size:
        raise ValueError("insufficient training vocabulary; preserve the model size explicitly")
    return {"tokenizer_kind": "dependency_free_regex_diff_frequency_tokenizer",
            "fit_policy": "selected DEV training records only; no validation targets or inputs",
            "fit_record_ids_sha256": _stable_sha256([r["id"] for r in records]),
            "vocab_size": vocab_size, "min_frequency": 2,
            "special_tokens": {t: tokens.index(t) for t in SPECIAL_TOKENS},
            "vocab": [{"id": i, "token": t, "count": counts[t]} for i, t in enumerate(tokens)]}


def prepare(data_dir, source_job, experiment_id, *, validation_repos=8, validation_records=256):
    _validate_identifier(experiment_id, "experiment_id")
    folder = data_dir / "artifacts/train-runs" / experiment_id
    if folder.exists() or any((folder.parent/f"{experiment_id}.{arm}.trainer-job.json").exists()
                              for arm in ("ordered", "shuffled")):
        raise ValueError("experiment already exists")
    source_path = data_dir / source_job
    job = _load_json(source_path)
    if blockers := _job_blockers(data_dir, job_path=source_path, job=job):
        raise ValueError(blockers)
    plans = {p["record_id"]: p for p in _iter_jsonl(data_dir/job["inputs"]["sequence_plan"]["path"])}
    records = [r for r in _iter_jsonl(data_dir/job["inputs"]["training_artifact"]["path"])
               if r["data_split"] == "DEV" and plans[r["id"]]["decision"] != "exclude_oversize"]
    train, validation, reserved, purged = split_dev(records, validation_repos=validation_repos,
                                                   validation_records=validation_records)
    if not train:
        raise ValueError("no training rows remain after leakage checks")
    tokenizer = fit_tokenizer(train, job["model_contract"]["tokenizer_vocab_size"])
    folder.mkdir(parents=True)
    _atomic_json(folder/"tokenizer.json", tokenizer)
    # Metadata includes all eligible DEV rows so the unchanged trainer can validate
    # source rows before selecting the explicit training IDs. Fitting uses train only.
    metadata = []
    for record in records:
        sequence = materialize_training_sequence(record, plans[record["id"]], tokenizer,
                          context_tokens=job["model_contract"]["context_tokens"])
        metadata.append(_sequence_metadata(sequence, plan_record=plans[record["id"]]))
    with (folder/"sequence-metadata.jsonl").open("w") as handle:
        for row in metadata:
            handle.write(json.dumps(row, sort_keys=True)+"\n")
    entry = lambda path: {"path": path.relative_to(data_dir).as_posix(), "sha256": _sha256(path)}
    _atomic_json(folder/"handoff.json", {
        "source_job":entry(source_path),
        "inputs":{"training_artifact":job["inputs"]["training_artifact"], "tokenizer":entry(folder/"tokenizer.json")},
        "training_contract":{"train_split":"DEV", "eval_split":"DEV_VALIDATION",
                             "train_record_ids_sha256":_stable_sha256([r["id"] for r in train])}})
    ids = [r["id"] for r in train]
    shuffled = ids.copy()
    random.Random(17).shuffle(shuffled)
    vocabulary = {v["token"] for v in tokenizer["vocab"]}
    validation_tokens = [t for r in validation for t in tokenize_text(r["target_message"])]
    protocol = {
        "experiment_id": experiment_id, "status": "prepared_not_trained", "source_job": entry(source_path),
        "implementation_hashes": code_hashes(), "model_contract": job["model_contract"],
        "training": {"seed":17, "epochs":1, "batch_size":1, "learning_rate":0.0003,
                     "precision":"float32", "checkpoint_every":100, "initialization":"fresh, identical seed in each arm"},
        "arms": {"ordered": ids, "shuffled": shuffled},
        "validation_ids": [r["id"] for r in validation], "reserved_repositories": reserved,
        "purged_cross_repository_duplicate_ids": [r["id"] for r in purged],
        "selection": f"First {validation_repos} repositories by SHA256(url); {validation_records} ID-hash-ranked rows round-robin; all other eligible DEV for training",
        "tokenizer_fit": "DEV training only, same regex and source model vocabulary size; shared by both arms",
        "validation_unknown_target_tokens": sum(t not in vocabulary for t in validation_tokens),
        "validation_target_tokens": len(validation_tokens),
        "evaluation": {"max_new_tokens":256, "decoder_policy":DECODER_POLICY, "split":"DEV_VALIDATION",
                       "primary":"supervised token-weighted CE", "secondary":["type_match", "scope_match", "exact_text", "dominant_type_scope_fraction"],
                       "decision":"Favor shuffled if CE is at least 5% lower and dominant type/scope fraction at least 0.10 lower; otherwise report tradeoffs/inconclusive.",
                       "timing":"final checkpoint of each one-pass arm; no adaptive stopping or hyperparameter selection"},
        "limitations":["Single seed and 256 repository-balanced DEV validation rows; no public quality claim.",
                       "Repository separation and exact/tokenized diff purging do not prove absence of all semantic near-duplicates.",
                       "Fresh fold tokenizer changes vocabulary relative to the earlier full run; only these paired arms isolate ordering.",
                       "One pass tests ordering at this exposure budget, not convergence.",
                       "REPORT and HELD_OUT remain untouched; original split labels and artifacts are not rewritten."],
        "files":{"tokenizer":entry(folder/"tokenizer.json"), "sequence_metadata":entry(folder/"sequence-metadata.jsonl"),
                 "handoff":entry(folder/"handoff.json")},
    }
    for arm in protocol["arms"]:
        run_id = f"{experiment_id}.{arm}"
        derived = copy.deepcopy(job)
        derived.update(run_id=run_id, source_job=entry(source_path), seed=17,
                       claim_policy="Internal DEV ordering ablation; not a full proof release run.")
        derived["inputs"]["tokenizer"] = protocol["files"]["tokenizer"]
        derived["inputs"]["handoff"] = protocol["files"]["handoff"]
        derived["inputs"]["sequence_metadata"] = protocol["files"]["sequence_metadata"]
        derived["inputs"].pop("sequence_materialization_report", None)
        derived["data_contract"] = {"train_split":"DEV", "eval_split":"DEV_VALIDATION", "train_dev_records":len(train),
                                    "train_report_records":0, "report_eval_records":0, "validation_records":len(validation)}
        derived["eval_contract"] = {"eval_split":"DEV_VALIDATION", "locked_report_required":False}
        derived["execution_plan"] = {**protocol["training"], "order":arm, "record_ids":protocol["arms"][arm]}
        derived["checkpoint_contract"] = {"checkpoint_dir": f"artifacts/train-runs/{run_id}/checkpoints"}
        derived["code"] = {"implementation_hashes":protocol["implementation_hashes"]}
        path = data_dir/"artifacts/train-runs"/f"{run_id}.trainer-job.json"
        _atomic_json(path, derived)
        protocol["files"][arm+"_job"] = entry(path)
    _atomic_json(folder/"protocol.json", protocol)
    _atomic_json(folder/"protocol-lock.json", {"protocol_sha256":_sha256(folder/"protocol.json")})
    return {"training_records":len(train), "validation_records":len(validation), "reserved_repositories":reserved,
            "purged_records":len(purged), "protocol":str(folder/"protocol.json")}


def load_protocol(data_dir, experiment_id):
    _validate_identifier(experiment_id, "experiment_id")
    folder = data_dir/"artifacts/train-runs"/experiment_id
    if _load_json(folder/"protocol-lock.json")["protocol_sha256"] != _sha256(folder/"protocol.json"):
        raise ValueError("frozen protocol changed")
    p = _load_json(folder/"protocol.json")
    if p["implementation_hashes"] != code_hashes():
        raise ValueError("implementation changed after protocol freeze")
    for item in [p["source_job"], *p["files"].values()]:
        if _sha256(data_dir/item["path"]) != item["sha256"]:
            raise ValueError("frozen input changed")
    source_path = data_dir/p["source_job"]["path"]
    if blockers := _job_blockers(data_dir, job_path=source_path, job=_load_json(source_path)):
        raise ValueError(blockers)
    return folder, p


def run_arm(data_dir, experiment_id, arm, resume=False, *, device="cuda"):
    _, p = load_protocol(data_dir, experiment_id)
    train = p["training"]
    return run_proof_lm_training(data_dir, run_id=f"{experiment_id}.{arm}", device=device,
        batch_size=train["batch_size"], learning_rate=train["learning_rate"],
        record_ids=p["arms"][arm], checkpoint_every=train["checkpoint_every"], resume=resume, write=True)


def evaluate_arm(data_dir, experiment_id, arm, *, device="cuda"):
    import torch
    folder, p = load_protocol(data_dir, experiment_id)
    output = folder/f"{arm}.validation.json"
    if output.exists():
        raise ValueError("validation result already exists")
    run_id = f"{experiment_id}.{arm}"
    validate_proof_lm_training(data_dir, run_id=run_id)
    manifest = _load_json(data_dir/"artifacts/train-runs"/run_id/"checkpoints/final.json")
    state_path = data_dir/manifest["state_path"]
    if _sha256(state_path) != manifest["state_sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if state["config"]["record_ids"] != p["arms"][arm]:
        raise ValueError("training IDs do not match protocol")
    if state["config"]["model_contract"] != p["model_contract"] or state["config"]["seed"] != p["training"]["seed"]:
        raise ValueError("model contract or seed changed")
    if state["trainer_state"]["completed_records"] != len(p["arms"][arm]):
        raise ValueError("training arm incomplete")
    _configure_device_runtime(torch, device)
    model = _build_model(torch, p["model_contract"], attention_chunk_size=256).to(device).eval()
    model.load_state_dict(state["model_state"])
    del state
    job = _load_json(data_dir/p["files"][arm+"_job"]["path"])
    tokenizer = _load_json(data_dir/p["files"]["tokenizer"]["path"])
    vocabulary = {v["token"]:v["id"] for v in tokenizer["vocab"]}
    reverse = {v:k for k,v in vocabulary.items()}
    selected = set(p["validation_ids"])
    records = {r["id"]:r for r in _iter_jsonl(data_dir/job["inputs"]["training_artifact"]["path"]) if r["id"] in selected}
    plans = {r["record_id"]:r for r in _iter_jsonl(data_dir/job["inputs"]["sequence_plan"]["path"]) if r["record_id"] in selected}
    predictions, counts, pairs = [], Counter(), Counter()
    loss_sum, tokens = 0.0, 0
    for identifier in p["validation_ids"]:
        record = records[identifier]
        if record["data_split"] != "DEV" or record["source_repo_url"] not in p["reserved_repositories"]:
            raise ValueError("validation boundary violated")
        sequence = materialize_training_sequence(record, plans[identifier], tokenizer, context_tokens=p["model_contract"]["context_tokens"])
        batch = _batch_for_torch(torch, [sequence], device=device)
        with torch.inference_mode():
            loss = float(model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"]).cpu())
        loss_sum += loss*batch["loss_tokens"]
        tokens += batch["loss_tokens"]
        prompt = prompt_tokens(record, context_tokens=p["model_contract"]["context_tokens"], max_new_tokens=256)
        ids = [vocabulary.get(t,vocabulary["<unk>"]) for t in prompt]
        generated, stop = generate(torch, model, ids, device=device, max_new_tokens=256,
                                   stop_ids={vocabulary["<sep>"],vocabulary["<eos>"]})
        message = decode_tokens([reverse[i] for i in generated])
        gold = parse_commit_message(record["target_message"])
        match = {"format_validity":False, "type_match":False, "scope_match":False,
                 "exact_text":message==record["target_message"]}
        try:
            parsed = parse_commit_message(message)
            match.update(format_validity=True,type_match=parsed.type==gold.type,scope_match=parsed.scope==gold.scope)
            pairs[(parsed.type,parsed.scope)] += 1
        except ValueError:
            pairs[("INVALID",None)] += 1
        counts.update({k:int(v) for k,v in match.items()})
        predictions.append({"record_id":identifier,"data_split":"DEV","message":message,"target_message":record["target_message"],
                            "output_token_ids":generated,"stop_reason":stop,"loss":loss,"loss_tokens":batch["loss_tokens"],**match})
        print(json.dumps({"arm":arm,"validated":len(predictions),"total":len(selected)}),flush=True)
    result = {"experiment_id":experiment_id,"arm":arm,"protocol_sha256":_sha256(folder/"protocol.json"),
              "checkpoint_sha256":manifest["state_sha256"],"records":len(predictions),"supervised_tokens":tokens,
              "mean_loss":loss_sum/tokens,"counts":dict(counts),"dominant_type_scope_fraction":max(pairs.values())/len(predictions),
              "dominant_type_scope":pairs.most_common(1)[0][0],"predictions":predictions}
    _atomic_json(output,result)
    return {k:v for k,v in result.items() if k!="predictions"}


def compare(data_dir, experiment_id):
    folder, p = load_protocol(data_dir, experiment_id)
    ordered, shuffled = (_load_json(folder/f"{arm}.validation.json") for arm in ("ordered","shuffled"))
    for result in (ordered, shuffled):
        if result["protocol_sha256"] != _sha256(folder/"protocol.json"):
            raise ValueError("validation protocol mismatch")
        if [r["record_id"] for r in result["predictions"]] != p["validation_ids"]:
            raise ValueError("validation coverage mismatch")
    if ordered["supervised_tokens"] != shuffled["supervised_tokens"]:
        raise ValueError("validation token budget mismatch")
    favors = (shuffled["mean_loss"] <= 0.95*ordered["mean_loss"] and
              ordered["dominant_type_scope_fraction"]-shuffled["dominant_type_scope_fraction"] >= 0.10)
    summary = {"experiment_id":experiment_id, "protocol_sha256":_sha256(folder/"protocol.json"),
               "decision":"favors_shuffled" if favors else "inconclusive_or_tradeoffs",
               "arms":{r["arm"]:{k:v for k,v in r.items() if k!="predictions"} for r in (ordered,shuffled)},
               "limitations":p["limitations"]}
    _atomic_json(folder/"comparison.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","train","evaluate","compare"))
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--experiment-id",required=True)
    parser.add_argument("--source-job",type=Path)
    parser.add_argument("--arm",choices=("ordered","shuffled"))
    parser.add_argument("--resume",action="store_true")
    a=parser.parse_args()
    if a.action=="prepare":
        if a.source_job is None: parser.error("prepare requires --source-job")
        result=prepare(a.data_dir,a.source_job,a.experiment_id)
    elif a.action=="compare":
        result=compare(a.data_dir,a.experiment_id)
    else:
        if a.arm is None: parser.error("train/evaluate requires --arm")
        result=run_arm(a.data_dir,a.experiment_id,a.arm,a.resume) if a.action=="train" else evaluate_arm(a.data_dir,a.experiment_id,a.arm)
    print(json.dumps(result,indent=2))
    if result.get("status") == "blocked":
        raise SystemExit(2)


if __name__=="__main__":
    main()
