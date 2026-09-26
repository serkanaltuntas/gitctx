"""Prepare lossless evidence windows; never silently turn them into training labels."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

from gitctx.evidence_windows import build_windows
from gitctx.proof_lm_train import _iter_jsonl, _load_json, _sha256, _validate_identifier
from gitctx.student_readiness import dump
from gitctx.student_tokenizer import StudentTokenizer


def prepare(data, parent, run):
    _validate_identifier(parent, 'parent'); _validate_identifier(run, 'run')
    prior = data / 'artifacts/train-runs' / parent
    manifest = _load_json(prior / 'manifest.json')
    for name, digest in manifest['inputs'].items():
        if _sha256(data / name) != digest: raise ValueError('parent input changed: ' + name)
    for name, digest in manifest['outputs'].items():
        if _sha256(prior / name) != digest: raise ValueError('parent output changed: ' + name)
    for name, digest in manifest['code'].items():
        if _sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError('parent implementation changed: ' + name)
    sources = [data / n for n in manifest['inputs'] if n.startswith('artifacts/train/')]
    if len(sources) != 1: raise ValueError('ambiguous source')
    coverage = {r['record_id']: r for r in _iter_jsonl(prior / 'coverage.jsonl')}
    tokenizer = StudentTokenizer.load(prior / 'tokenizer.json')
    folder = data / 'artifacts/train-runs' / run
    folder.mkdir()  # Immutable run directory: never overwrite earlier evidence.
    counts = {}; seen = set()
    with (folder / 'windows.jsonl.partial').open('w') as out:
        for record in _iter_jsonl(sources[0]):
            if record['data_split'] != 'DEV': continue
            rid = record['id']
            if rid in seen or rid not in coverage: raise ValueError('invalid DEV ID')
            seen.add(rid)
            part = coverage[rid]['partition']
            windows = build_windows(record, tokenizer)
            c = counts.setdefault(part, Counter())
            c['records'] += 1; c['windows'] += len(windows)
            c['split_records'] += len(windows) > 1
            c['source_characters'] += len(record['diff'])
            c['source_bytes'] += len(record['diff'].encode('utf-8'))
            c['max_prompt_tokens'] = max(c['max_prompt_tokens'], max(w['prompt_tokens'] for w in windows))
            for w in windows:
                w['partition'] = part
                out.write(json.dumps(w, ensure_ascii=False, sort_keys=True) + '\n')
            if len(seen) % 100 == 0: print(f'Windows {len(seen)}/{len(coverage)}', flush=True)
    if seen != set(coverage): raise ValueError('incomplete DEV coverage')
    (folder / 'windows.jsonl.partial').rename(folder / 'windows.jsonl')
    dump(folder / 'coverage-summary.json', {'counts':{p:dict(c) for p,c in counts.items()},
        'records':len(seen), 'exact_source_reconstruction':True, 'silently_dropped_records':0,
        'source_characters_dropped':0, 'training_ready':False, 'targets_assigned':0,
        'context_tokens':8192, 'answer_reserve_including_eos':256,
        'report_or_held_out_used':False, 'training_launched':False})
    dump(folder / 'manifest.json', {'run_id':run, 'parent':parent,
        'parent_manifest_sha256':_sha256(prior/'manifest.json'),
        'source_path':str(sources[0].relative_to(data)), 'source_sha256':_sha256(sources[0]),
        'tokenizer_path':str((prior/'tokenizer.json').relative_to(data)),
        'tokenizer_sha256':_sha256(prior/'tokenizer.json'),
        'code':{n:_sha256(Path(__file__).with_name(n)) for n in
                ('evidence_windows.py','window_readiness.py')},
        'outputs':{n:_sha256(folder/n) for n in ('windows.jsonl','coverage-summary.json')}})
    print(json.dumps(_load_json(folder/'coverage-summary.json'),indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--parent',required=True);p.add_argument('--run',required=True)
    a=p.parse_args();prepare(a.data_dir,a.parent,a.run)
