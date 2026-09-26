"""Resumable local open-model evidence review of a frozen explicit DEV ID list.

This command records candidates, never approves labels or starts training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer
from gitctx.proof_lm_train import _iter_jsonl, _load_json, _sha256
from gitctx.reference_review import request_json, review_one, split_for_review, VERSION
from gitctx.student_readiness import dump


def run(source, selection, tokenizer_path, folder, model, license_path, model_revision):
    ids = _load_json(selection)
    if not isinstance(ids,list) or len(ids) != len(set(ids)) or not ids:
        raise ValueError('selection must be nonempty unique IDs')
    if 'Apache License' not in license_path.read_text():
        raise ValueError('reviewed Apache license file required')
    tags = request_json('http://127.0.0.1:11434/api/tags')['models']
    digest = next(m['digest'] for m in tags if m['name'] == model)
    protocol = {'source_sha256':_sha256(source),'selection_sha256':_sha256(selection),
        'selected_ids':ids, 'tokenizer_sha256':_sha256(tokenizer_path),
        'license_sha256':_sha256(license_path),'model_revision':model_revision,
        'model':model,'model_digest':digest,'context_tokens':16384,'output_tokens':768,
        'version':VERSION, 'temperature':0, 'seed':17,
        'code':{n:_sha256(Path(__file__).with_name(n)) for n in
                ('reference_review.py','review_batch.py','evidence_windows.py')},
        'training_promotion':False,'independent_human_review':False}
    folder.mkdir(exist_ok=True)
    path = folder/'review-protocol.json'
    if path.exists():
        if _load_json(path) != protocol: raise ValueError('resume protocol changed')
    else: dump(path,protocol)
    output=folder/'reviews.jsonl'
    done={}
    if output.exists():
        for row in _iter_jsonl(output):
            key=(row['record_id'],*row['source_range'])
            if key in done: raise ValueError('duplicate review range')
            if row['model_digest']!=digest: raise ValueError('review model changed')
            done[key]=row
    wanted=set(ids)
    records={r['id']:r for r in _iter_jsonl(source) if r['id'] in wanted}
    if set(records)!=wanted or any(r['data_split']!='DEV' for r in records.values()):
        raise ValueError('selection must contain only existing DEV records')
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    with output.open('a') as out:
        for n,rid in enumerate(ids,1):
            r=records[rid]
            current=next(m['digest'] for m in request_json('http://127.0.0.1:11434/api/tags')['models'] if m['name']==model)
            if current!=digest:raise ValueError('runtime model changed')
            chunks=split_for_review(r,tokenizer)
            for start,end in chunks:
                key=(rid,start,end)
                if key in done:continue
                result=review_one(r,start,end,tokenizer=tokenizer,model=model,model_digest=digest)
                out.write(json.dumps(result,ensure_ascii=False,sort_keys=True)+'\n');out.flush()
                done[key]=result
            print(f'Review {n}/{len(ids)}: {rid} ({len(chunks)} parts)',flush=True)
    dump(folder/'review-completion.json',{'records':len(ids),'review_parts':len(done),
        'valid_parts':sum(r['status']=='valid_review' for r in done.values()),
        'unresolved_parts':sum(r['status']!='valid_review' for r in done.values()),
        'reviews_sha256':_sha256(output),'training_ready':False,'labels_promoted':0})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','selection','tokenizer','output','license'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--model-revision',required=True)
    a=p.parse_args();run(a.source,a.selection,a.tokenizer,a.output,a.model,a.license,a.model_revision)
