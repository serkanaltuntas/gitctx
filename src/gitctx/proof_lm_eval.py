"""Locked REPORT generation and deterministic scoring for a completed proof LM.

Generation receives only system/user messages. The reference target is used only
for scoring; neither its contents nor its length determine the generation input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from gitctx.conventional import CommitContext, parse_commit_message, score_commit_message
from gitctx.proof_lm_train import (
    _atomic_json, _build_model, _configure_device_runtime, _data_path, _iter_jsonl, _job_blockers, _load_json,
    _load_torch, _resume_config_sha256, _sha256, _stable_sha256,
    proof_lm_final_checkpoint_path, validate_proof_lm_training,
)
from gitctx.proof_tokenizer import tokenize_text
from gitctx.proof_train_job import proof_trainer_job_path


def prompt_tokens(record: dict[str, Any], *, context_tokens: int, max_new_tokens: int) -> list[str]:
    """Use the full inference context with a fixed output reserve; never read gold text."""
    messages = [message for message in record["messages"] if message["role"] != "assistant"]
    if len([m for m in messages if m["role"] == "user"]) != 1:
        raise ValueError("evaluation expects exactly one user message")
    blocks = [(m["role"], tokenize_text(m["content"])) for m in messages]
    fixed = 2 + sum(2 + len(tokens) for role, tokens in blocks if role != "user") + 2
    budget = context_tokens - max_new_tokens - fixed
    if budget < 0:
        raise ValueError("system messages exceed inference context budget")
    result = ["<bos>"]
    for role, tokens in blocks:
        if role == "user" and len(tokens) > budget:
            prefix = (budget + 1) // 2
            suffix = budget - prefix
            tokens = tokens[:prefix] + (tokens[-suffix:] if suffix else [])
        result.extend([f"<{role}>", *tokens, "<sep>"])
    result.append("<assistant>")
    assert len(result) <= context_tokens - max_new_tokens
    return result


def decode_tokens(tokens: list[str]) -> str:
    """Deterministic whitespace reconstruction for the frozen lossy regex tokenizer."""
    text = " ".join("\n" if token == "<nl>" else token for token in tokens)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{]) +", r"\1", text)
    # A tokenized Conventional Commit scope is adjacent to its type.
    text = re.sub(r"^([a-z][a-z0-9-]*) \(", r"\1(", text)
    return text.strip()


def generate(torch: Any, model: Any, ids: list[int], *, device: str,
             max_new_tokens: int, stop_ids: set[int]) -> tuple[list[int], str]:
    output = []
    cache = None
    with torch.inference_mode():
        for step in range(max_new_tokens):
            current = ids if cache is None else [output[-1]]
            tensor = torch.tensor([current], device=device, dtype=torch.long)
            mask = torch.ones((1, len(ids) + step), device=device, dtype=torch.long)
            logits, cache = model(tensor, mask, past_key_values=cache, use_cache=True,
                                  last_token_only=True)
            token = int(logits[0, -1].argmax())
            if token in stop_ids:
                return output, "stop_token"
            output.append(token)
    return output, "token_limit"


def evaluate(data_dir: Path, run_id: str, *, device: str = "cuda",
             max_new_tokens: int = 256) -> dict[str, Any]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    validate_proof_lm_training(data_dir, run_id=run_id)
    checkpoint = _load_json(data_dir / proof_lm_final_checkpoint_path(run_id))
    job_path = data_dir / proof_trainer_job_path(run_id)
    job = _load_json(job_path)
    blockers = _job_blockers(data_dir, job_path=job_path, job=job)
    if blockers:
        raise ValueError(blockers)
    torch = _load_torch()
    if torch is None:
        raise ValueError("torch is not installed")
    _configure_device_runtime(torch, device)
    state = torch.load(_data_path(data_dir, Path(checkpoint["state_path"])), map_location="cpu", weights_only=True)
    config = state["config"]
    if _resume_config_sha256(config) != checkpoint["config_sha256"]:
        raise ValueError("checkpoint config mismatch")
    if config.get("record_ids") or config.get("max_records") is not None:
        raise ValueError("locked evaluation requires the complete DEV run, not a diagnostic subset")
    if state["trainer_state"]["completed_records"] != job["data_contract"]["train_dev_records"]:
        raise ValueError("full DEV coverage is not proven")
    for name, value in config["input_hashes"].items():
        entry = job["inputs"][name]
        if value != entry.get("sha256", entry.get("actual_sha256")):
            raise ValueError("input lineage changed since training")
    for name, sha in config["implementation_hashes"].items():
        if _sha256(Path(__file__).with_name(name)) != sha:
            raise ValueError("training implementation changed")
    model = _build_model(torch, config["model_contract"], attention_chunk_size=256).to(device).eval()
    model.load_state_dict(state["model_state"])
    del state
    tokenizer = _load_json(data_dir / job["inputs"]["tokenizer"]["path"])
    vocabulary = {item["token"]: item["id"] for item in tokenizer["vocab"]}
    reverse = {item["id"]: item["token"] for item in tokenizer["vocab"]}
    records = [r for r in _iter_jsonl(data_dir / job["inputs"]["training_artifact"]["path"])
               if r["data_split"] == "REPORT"]
    if len(records) != job["data_contract"]["report_eval_records"]:
        raise ValueError("locked REPORT coverage mismatch")
    identity = {"run_id": run_id, "checkpoint_sha256": checkpoint["state_sha256"],
                "tokenizer_sha256": _sha256(data_dir / job["inputs"]["tokenizer"]["path"]),
                "evaluation_code_sha256": _sha256(Path(__file__)),
                "max_new_tokens": max_new_tokens, "decode": "greedy",
                "prompt_policy": "system/user only; fixed generation reserve; prefix/suffix user crop",
                "device": device, "torch": str(torch.__version__)}
    identity_sha = _stable_sha256(identity)
    pred_path = data_dir / job["eval_contract"]["report_prediction_path"]
    report_path = data_dir / job["eval_contract"]["report_eval_path"]
    meta_path = pred_path.with_suffix(".meta.json")
    if meta_path.exists() and _load_json(meta_path) != identity:
        raise ValueError("existing evaluation has different generation settings")
    _atomic_json(meta_path, identity)
    predictions = list(_iter_jsonl(pred_path)) if pred_path.exists() else []
    if len(predictions) > len(records):
        raise ValueError("too many prior predictions")
    for index, prediction in enumerate(predictions):
        if prediction["record_id"] != records[index]["id"] or prediction["identity_sha256"] != identity_sha:
            raise ValueError("existing predictions do not match the locked run")
    started = time.monotonic()
    with pred_path.open("a", encoding="utf-8") as handle:
        for record in records[len(predictions):]:
            tokens = prompt_tokens(record, context_tokens=config["model_contract"]["context_tokens"],
                                   max_new_tokens=max_new_tokens)
            ids = [vocabulary.get(token, vocabulary["<unk>"]) for token in tokens]
            output, reason = generate(torch, model, ids, device=device, max_new_tokens=max_new_tokens,
                                      stop_ids={vocabulary["<sep>"], vocabulary["<eos>"]})
            message = decode_tokens([reverse[token] for token in output])
            prediction = {"record_id": record["id"], "data_split": "REPORT", "message": message,
                          "output_token_ids": output, "stop_reason": reason,
                          "prompt_tokens": len(ids), "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                          "identity_sha256": identity_sha}
            handle.write(json.dumps(prediction, sort_keys=True) + "\n")
            handle.flush()
            predictions.append(prediction)
            if len(predictions) % 10 == 0:
                print(json.dumps({"evaluated": len(predictions), "total": len(records),
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    counts = {name: Counter() for name in ("format_validity", "type_match", "scope_quality", "specificity",
                                          "brevity", "exact_message_match", "exact_token_match")}
    for record, prediction in zip(records, predictions):
        target = record.get("target_message") or next(m["content"] for m in record["messages"] if m["role"] == "assistant")
        try:
            expected_type = parse_commit_message(target).type
        except ValueError:
            expected_type = None
        score = score_commit_message(prediction["message"], CommitContext(
            changed_paths=tuple(record["changed_paths"]), expected_type=expected_type))
        values = {"format_validity": score.format_validity, "type_match": score.type_accuracy,
                  "scope_quality": score.scope_quality, "specificity": score.specificity,
                  "brevity": score.brevity, "exact_message_match": prediction["message"] == target,
                  "exact_token_match": [reverse[i] for i in prediction["output_token_ids"]] == tokenize_text(target)}
        for name, value in values.items():
            counts[name]["unknown" if value is None else str(value).lower()] += 1
    report = {"status": "evaluated", "identity": identity, "records": len(records),
              "counts": {name: dict(value) for name, value in counts.items()},
              "rates_over_all_records": {name: value["true"] / len(records) for name, value in counts.items()},
              "prediction_sha256": _sha256(pred_path),
              "limitations": ["Whitespace is reconstructed by a fixed heuristic; the frozen regex tokenizer is lossy.",
                              "Scope and specificity are deterministic proxies, not human judgments of factual correctness.",
                              "REPORT is not used for model selection or training."]}
    _atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data_dir, args.run_id, device=args.device,
                              max_new_tokens=args.max_new_tokens), indent=2))


if __name__ == "__main__":
    main()
