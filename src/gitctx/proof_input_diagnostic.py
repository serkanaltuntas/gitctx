"""Frozen DEV-only input ablation and lossy-tokenizer diagnostics; never trains."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from gitctx.conventional import parse_commit_message
from gitctx.proof_lm_eval import decode_tokens, generate, prompt_tokens
from gitctx.proof_lm_train import (_atomic_json, _batch_for_torch, _build_model,
    _configure_device_runtime, _iter_jsonl, _load_json, _sha256, _validate_identifier)
from gitctx.proof_order_ablation import load_protocol
from gitctx.proof_sequences import materialize_training_sequence
from gitctx.proof_tokenizer import tokenize_text

VARIANTS = ('real', 'empty_diff', 'swapped_diff', 'swapped_user')

def user_message(record):
    users = [m for m in record['messages'] if m['role'] == 'user']
    if len(users) != 1:
        raise ValueError('exactly one user message required')
    return users[0]['content']

def replace_diff(record, diff):
    result = deepcopy(record)
    old = record['diff']
    content = user_message(record)
    marker = '\nDiff:\n'
    start = content.index(marker) + len(marker)
    if not old or content[start:start+len(old)] != old:
        raise ValueError('diff is not exactly embedded after its marker')
    for message in result['messages']:
        if message['role'] == 'user':
            message['content'] = content[:start] + diff + content[start+len(old):]
    result['diff'] = diff
    return result

def variant_record(record, donor, variant):
    if variant == 'real':
        return deepcopy(record)
    if variant == 'empty_diff':
        return replace_diff(record, '')
    if variant == 'swapped_diff':
        return replace_diff(record, donor['diff'])
    if variant == 'swapped_user':
        result = deepcopy(record)
        for message in result['messages']:
            if message['role'] == 'user':
                message['content'] = user_message(donor)
        return result
    raise ValueError(variant)

def select_diverse(records, count):
    if count < 1 or count > len(records):
        raise ValueError('invalid sample count')
    groups = defaultdict(list)
    for r in records:
        length = len(tokenize_text(user_message(r)))
        bucket = 'short' if length < 1024 else 'medium' if length < 4096 else 'long'
        key = (r['source_repo_url'], parse_commit_message(r['target_message']).type, bucket)
        groups[key].append(r)
    for rows in groups.values():
        rows.sort(key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
    keys = sorted(groups, key=lambda k: hashlib.sha256(repr(k).encode()).hexdigest())
    selected = []
    while len(selected) < count:
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop(0))
                if len(selected) == count:
                    break
    return selected

def choose_donors(records):
    lengths = {r['id']: len(tokenize_text(r['diff'])) for r in records}
    return {r['id']: min((d for d in records if d['source_repo_url'] != r['source_repo_url']),
            key=lambda d: (abs(lengths[d['id']]-lengths[r['id']]), d['id']))['id'] for r in records}

def sequence_for_prompt(prompt, target, vocabulary):
    # Identical inference prompt across generation and teacher-forced loss.
    suffix = tokenize_text(target) + ['<sep>', '<eos>']
    tokens = prompt + suffix
    if len(tokens) > 8192:
        raise ValueError('reference exceeds fixed output reserve')
    return {'input_ids': [vocabulary.get(t, vocabulary['<unk>']) for t in tokens],
            'loss_mask': [0]*len(prompt) + [1]*len(suffix)}

def retained_changes(record, kept_user_tokens):
    content = user_message(record)
    raw = len(tokenize_text(content))
    budget = len(kept_user_tokens)
    prefix = (budget+1)//2 if budget < raw else raw
    suffix = budget-prefix if budget < raw else 0
    position = len(tokenize_text(content[:content.index('\nDiff:\n')+len('\nDiff:\n')]))
    counts = Counter()
    for line in record['diff'].splitlines(keepends=True):
        size = len(tokenize_text(line))
        if line.startswith(('+','-')) and not line.startswith(('+++','---')):
            retained = sum(i < prefix or (suffix and i >= raw-suffix) for i in range(position, position+size))
            counts['changed_lines'] += 1
            counts['fully_retained'] += int(retained == size)
            counts['fully_removed'] += int(retained == 0)
            counts['partly_retained'] += int(0 < retained < size)
        position += size
    return dict(counts)

def inspect_row(record, plan, tokenizer):
    vocabulary = {v['token']:v['id'] for v in tokenizer['vocab']}
    seq = materialize_training_sequence(record, plan, tokenizer, context_tokens=8192)
    ai = seq['tokens'].index('<assistant>')
    expected = tokenize_text(record['target_message']) + ['<sep>','<eos>']
    assert seq['tokens'][ai+1:] == expected
    assert seq['loss_mask'] == [0]*(ai+1) + [1]*len(expected)
    prompt = prompt_tokens(record, context_tokens=8192, max_new_tokens=256)
    user_start = prompt.index('<user>')+1
    user_end = prompt.index('<sep>', user_start)
    target = tokenize_text(record['target_message'])
    inputs = [t for m in record['messages'] if m['role'] != 'assistant' for t in tokenize_text(m['content'])]
    decoded = decode_tokens(target)
    decoded_ids = decode_tokens([t if t in vocabulary else '<unk>' for t in target])
    try:
        json.loads(record['target_message'])
        json_target = True
    except json.JSONDecodeError:
        json_target = False
    return {'record_id':record['id'], 'repo':record['source_repo_url'],
        'type':parse_commit_message(record['target_message']).type,
        'raw_input_tokens':len(inputs), 'unknown_input_tokens':sum(t not in vocabulary for t in inputs),
        'target_tokens':len(target), 'unknown_target_tokens':sum(t not in vocabulary for t in target),
        'target_text_roundtrip':decoded == record['target_message'],
        'target_id_roundtrip':decoded_ids == record['target_message'],
        'json_instruction_plain_target':any('json' in m['content'].lower() for m in record['messages'] if m['role'] != 'assistant') and not json_target,
        'train_sequence_tokens':len(seq['tokens']), 'inference_prompt_tokens':len(prompt),
        'train_inference_prompt_equal':seq['tokens'][:ai+1] == prompt,
        'train_cropped':seq['crop']['policy'] != 'full',
        'inference_cropped':len(tokenize_text(user_message(record))) > user_end-user_start,
        'inference_changed_lines':retained_changes(record, prompt[user_start:user_end]),
        'mask_target_alignment':True,
        'label_source':record.get('label_source'), 'review_notes':record.get('review_notes')}

def aggregate(rows):
    sums = ('raw_input_tokens','unknown_input_tokens','target_tokens','unknown_target_tokens',
            'target_text_roundtrip','target_id_roundtrip','json_instruction_plain_target',
            'train_inference_prompt_equal','train_cropped','inference_cropped','mask_target_alignment')
    return {'records':len(rows), **{k:sum(r[k] for r in rows) for k in sums},
            'changed_lines':dict(sum((Counter(r['inference_changed_lines']) for r in rows),Counter())),
            'types':dict(Counter(r['type'] for r in rows)), 'repositories':len({r['repo'] for r in rows})}

def inputs(data, parent):
    folder, p = load_protocol(data, parent)
    job = _load_json(data/p['files']['shuffled_job']['path'])
    wanted = set(p['arms']['shuffled']) | set(p['validation_ids'])
    records = {r['id']:r for r in _iter_jsonl(data/job['inputs']['training_artifact']['path']) if r['id'] in wanted}
    assert len(records) == len(wanted) and all(r['data_split']=='DEV' for r in records.values())
    plans = {r['record_id']:r for r in _iter_jsonl(data/job['inputs']['sequence_plan']['path']) if r['record_id'] in wanted}
    return folder, p, records, plans, _load_json(data/p['files']['tokenizer']['path'])

def prepare(data, parent, run):
    _validate_identifier(run, 'run')
    output = data/'artifacts/train-runs'/run
    if output.exists():
        raise ValueError('diagnostic already exists')
    folder,p,records,plans,tokenizer = inputs(data,parent)
    selected = select_diverse([records[i] for i in p['validation_ids']],128)
    review_train = select_diverse([records[i] for i in p['arms']['shuffled']],32)
    manifest_path = data/'artifacts/train-runs'/f'{parent}.shuffled/checkpoints/final.json'
    manifest = _load_json(manifest_path)
    protocol = {'id':run,'parent':parent,'parent_protocol_sha256':_sha256(folder/'protocol.json'),
        'implementation_sha256':_sha256(Path(__file__)), 'checkpoint_manifest':str(manifest_path.relative_to(data)),
        'checkpoint_manifest_sha256':_sha256(manifest_path), 'checkpoint_sha256':manifest['state_sha256'],
        'selected_ids':[r['id'] for r in selected], 'review_training_ids':[r['id'] for r in review_train],
        'selection':'Hash-ranked round-robin repository/type/raw-user-length strata; 128 reserved DEV validation + 32 training review rows',
        'donors':choose_donors(selected),'donor_policy':'Different repository; nearest raw diff token length; ID tie break; donors may repeat',
        'variants':list(VARIANTS),'max_new_tokens':256,'context_tokens':8192,
        'metrics':['paired output identity','type/scope match against original reference','target-token weighted conditional CE on identical inference prompts',
                   'same-reference mean paired CE differences','raw and ID tokenizer roundtrip','unknown token counts','cropped changed-line token retention'],
        'interpretation':'A changed output alone does not prove useful grounding. Lower real-input CE and better reference metrics are diagnostic proxies, not semantic correctness. Empty/swapped diffs retain original metadata; swapped_user is a separate whole-context control. Perturbations are out of distribution.',
        'scope':'DEV only; no optimization, tokenizer fitting, reference rewriting, REPORT or HELD_OUT evaluation',
        'budget':'128 examples x 4 variants; one fixed completed checkpoint; no adaptive sampling or training',
        'review_policy':'Assistant evidence review only, not new training labels or human acceptance; original provenance remains unchanged.'}
    output.mkdir(parents=True)
    _atomic_json(output/'protocol.json',protocol)
    _atomic_json(output/'protocol-lock.json',{'sha256':_sha256(output/'protocol.json')})
    summaries={}
    with (output/'coverage.jsonl').open('w') as handle:
        for name,ids in [('train',p['arms']['shuffled']),('validation',p['validation_ids'])]:
            rows=[]
            for index,i in enumerate(ids):
                row=inspect_row(records[i],plans[i],tokenizer);row['partition']=name;rows.append(row)
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
                if index%1000==0: print(json.dumps({'coverage':name,'processed':index}),flush=True)
            summaries[name]=aggregate(rows)
    _atomic_json(output/'coverage-summary.json',summaries)
    # Review packets reference the existing data instead of duplicating full diffs.
    with (output/'review-index.jsonl').open('w') as handle:
        for r in selected+review_train:
            handle.write(json.dumps({'record_id':r['id'],'partition':'validation' if r['id'] in p['validation_ids'] else 'train',
                'repo':r['source_repo_url'],'target':r['target_message'],'changed_paths':r['changed_paths'],
                'source_diff_sha256':r['diff_sha256'],'review_status':'pending'},ensure_ascii=False)+'\n')
    return {'folder':str(output),'coverage':summaries}

def score_message(message, target):
    gold = parse_commit_message(target)
    try:
        parsed = parse_commit_message(message)
        return {'format_valid':True,'type_match':parsed.type==gold.type,'scope_match':parsed.scope==gold.scope,
                'pair':[parsed.type,parsed.scope], 'exact_text':message==target}
    except ValueError:
        return {'format_valid':False,'type_match':False,'scope_match':False,'pair':['INVALID',None],'exact_text':False}

def summarize_predictions(rows):
    result={}
    grouped=defaultdict(dict)
    for r in rows: grouped[r['record_id']][r['variant']]=r
    for variant in VARIANTS:
        subset=[r for r in rows if r['variant']==variant]
        total=sum(r['loss_tokens'] for r in subset)
        result[variant]={'records':len(subset),'mean_ce':sum(r['ce']*r['loss_tokens'] for r in subset)/total,
            'loss_tokens':total, **{k:sum(r[k] for r in subset) for k in ('format_valid','type_match','scope_match','exact_text')},
            'pairs':dict(Counter(json.dumps(r['pair']) for r in subset))}
        if variant!='real':
            pairs=[(g['real'],g[variant]) for g in grouped.values()]
            diffs=[b['ce']-a['ce'] for a,b in pairs]
            result[variant].update(identical_to_real=sum(a['output_token_ids']==b['output_token_ids'] for a,b in pairs),
                mean_paired_ce_increase=sum(diffs)/len(diffs),real_lower_ce=sum(d>0 for d in diffs))
    return result

def run(data,parent,run_id):
    import torch
    folder,p,records,plans,tokenizer=inputs(data,parent)
    output=data/'artifacts/train-runs'/run_id
    protocol=_load_json(output/'protocol.json')
    assert _load_json(output/'protocol-lock.json')['sha256']==_sha256(output/'protocol.json')
    assert protocol['parent']==parent and protocol['parent_protocol_sha256']==_sha256(folder/'protocol.json')
    assert protocol['implementation_sha256']==_sha256(Path(__file__))
    if (output/'predictions.jsonl').exists(): raise ValueError('outputs already exist')
    manifest_path=data/protocol['checkpoint_manifest']
    assert protocol['checkpoint_manifest_sha256']==_sha256(manifest_path)
    manifest=_load_json(manifest_path)
    state_path=data/manifest['state_path']
    assert _sha256(state_path)==protocol['checkpoint_sha256']
    state=torch.load(state_path,map_location='cpu',weights_only=True)
    assert state['config']['record_ids']==p['arms']['shuffled']
    _configure_device_runtime(torch,'cuda')
    torch.set_num_threads(4)
    model=_build_model(torch,p['model_contract'],attention_chunk_size=256).cuda().eval()
    model.load_state_dict(state['model_state']);del state
    vocabulary={v['token']:v['id'] for v in tokenizer['vocab']};reverse={v:k for k,v in vocabulary.items()}
    old={r['record_id']:r for r in _load_json(folder/'shuffled.validation.json')['predictions']}
    rows=[];start=time.monotonic()
    with (output/'predictions.jsonl').open('w') as handle:
        for index,identifier in enumerate(protocol['selected_ids']):
            record=records[identifier];donor=records[protocol['donors'][identifier]]
            assert identifier in p['validation_ids'] and donor['source_repo_url']!=record['source_repo_url']
            for variant in VARIANTS:
                modified=variant_record(record,donor,variant)
                prompt=prompt_tokens(modified,context_tokens=8192,max_new_tokens=256)
                sequence=sequence_for_prompt(prompt,record['target_message'],vocabulary)
                batch=_batch_for_torch(torch,[sequence],device='cuda')
                with torch.inference_mode():
                    ce=float(model(batch['input_ids'],batch['attention_mask'],labels=batch['labels']).cpu())
                ids=sequence['input_ids'][:len(prompt)]
                generated,stop=generate(torch,model,ids,device='cuda',max_new_tokens=256,stop_ids={vocabulary['<sep>'],vocabulary['<eos>']})
                message=decode_tokens([reverse[i] for i in generated])
                row={'record_id':identifier,'variant':variant,'donor_id':donor['id'] if variant.startswith('swapped') else None,
                    'prompt_tokens':len(prompt),'ce':ce,'loss_tokens':batch['loss_tokens'],'message':message,
                    'output_token_ids':generated,'stop_reason':stop,**score_message(message,record['target_message'])}
                if variant=='real':
                    row['reproduces_parent']=generated==old[identifier]['output_token_ids']
                    assert row['reproduces_parent'],'original prediction not reproduced'
                rows.append(row);handle.write(json.dumps(row,ensure_ascii=False)+'\n');handle.flush()
            print(json.dumps({'completed':index+1,'total':128,'seconds':round(time.monotonic()-start,1)}),flush=True)
    result={'protocol_sha256':_sha256(output/'protocol.json'),'checkpoint_sha256':protocol['checkpoint_sha256'],
        'seconds':time.monotonic()-start,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'metrics':summarize_predictions(rows),
        'real_predictions_reproduced':sum(r.get('reproduces_parent',False) for r in rows)}
    _atomic_json(output/'result.json',result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('data_dir',type=Path)
    parser.add_argument('parent')
    parser.add_argument('run_id')
    args=parser.parse_args()
    result=prepare(args.data_dir,args.parent,args.run_id) if args.command=='prepare' else run(args.data_dir,args.parent,args.run_id)
    print(json.dumps(result,ensure_ascii=False,indent=2))
