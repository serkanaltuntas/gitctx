"""Supervised execution of a pinned, explicitly authorized reviewed DEV run.

No network access, budget selection, label generation or protected-split inference.
An external supervisor should additionally enforce the same wall-clock deadline.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import time

from gitctx.conventional import parse_commit_message
from gitctx.proof_lm_train import _build_model, _configure_device_runtime, _load_torch
from gitctx.reference_overlay import artifact_hash
from gitctx.reviewed_inputs import load_dataset
from gitctx.reviewed_dataset import score_example, predict_record
from gitctx.reviewed_training import run_epochs
from gitctx.student_tokenizer import StudentTokenizer


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, immutable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True) + '\n'
    if immutable and path.exists():
        if path.read_text() != content:
            raise ValueError('immutable result differs: ' + str(path))
        return
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


class BudgetStop(RuntimeError):
    pass


class Guard:
    def __init__(self, directory, deadline, minimum_free_bytes):
        self.directory, self.deadline, self.minimum_free_bytes = directory, deadline, minimum_free_bytes

    def __call__(self):
        if time.time() >= self.deadline:
            raise BudgetStop('wall_time_limit')
        if shutil.disk_usage(self.directory).free < self.minimum_free_bytes:
            raise BudgetStop('disk_free_floor')


class GuardedDataset:
    def __init__(self, dataset, guard):
        self.dataset, self.guard, self.fingerprint = dataset, guard, dataset.fingerprint

    def ids(self, partition):
        return self.dataset.ids(partition)

    def example(self, rid, *, partition):
        self.guard()
        return self.dataset.example(rid, partition=partition)


def prediction_metrics(predictions, references):
    headers, counts = Counter(), Counter()
    for p in predictions:
        rid = p['record_id']; message = p['message']
        counts['records'] += 1
        counts['decode_errors'] += p['decode_error'] is not None
        counts['token_limit'] += p['stop_reason'] == 'token_limit'
        headers[message.splitlines()[0] if message and message.splitlines() else '<empty>'] += 1
        try:
            parsed = parse_commit_message(message or '')
        except ValueError:
            continue
        reference = parse_commit_message(references[rid])
        counts['format_valid'] += 1
        counts['type_match'] += parsed.type == reference.type
        counts['scope_match'] += parsed.scope == reference.scope
        counts['exact_match'] += message == references[rid]
    total = counts['records']
    return {**counts, 'dominant_header': headers.most_common(1)[0][0],
            'dominant_header_count': headers.most_common(1)[0][1],
            'dominant_header_fraction': headers.most_common(1)[0][1] / total}


def review_accuracy(review, *, predictions_sha256, panel_sha256, panel, records):
    if (review.get('predictions_sha256') != predictions_sha256
            or review.get('panel_sha256') != panel_sha256
            or review.get('reviewer_kind') not in ('assistant', 'human')
            or review.get('independent_human_review') is not (review.get('reviewer_kind') == 'human')):
        raise ValueError('review lineage mismatch')
    entries = review['entries']
    required = {p['record_id'] for p in panel['entries']}
    if len(entries) != len(required) or {r['record_id'] for r in entries} != required:
        raise ValueError('review must cover every frozen panel record exactly once')
    for entry in entries:
        if type(entry.get('correct_principal_change')) is not bool or type(entry.get('unsupported_claim')) is not bool:
            raise ValueError('explicit factuality verdicts required')
        if not entry.get('note') or not entry.get('evidence'):
            raise ValueError('source evidence required')
        lines = records[entry['record_id']]['diff'].splitlines(keepends=True)
        for evidence in entry['evidence']:
            n = evidence['line']
            if type(n) is not int or not 1 <= n <= len(lines) or evidence['quote'] != lines[n-1]:
                raise ValueError('review evidence differs from source')
    return sum(r['correct_principal_change'] for r in entries) / len(entries)


def execute(root, protocol_path, expected_digest, *, resume=False):
    root = Path(root).resolve(); protocol_path = Path(protocol_path).resolve()
    if digest(protocol_path) != expected_digest:
        raise ValueError('selected protocol changed')
    q = read(protocol_path)
    if q.get('training_authorized') is not True or q['epochs'] not in (1, 2, 4):
        raise ValueError('explicitly authorized bounded protocol required')
    if q['allowed_partitions'] != ['train', 'validation'] or not {'REPORT', 'HELD_OUT'} <= set(q['forbidden_partitions']):
        raise ValueError('only the frozen active DEV experiment is supported')
    repo = Path(__file__).resolve().parents[2]
    for name, sha in q['module_hashes'].items():
        if digest(repo/name) != sha:
            raise ValueError('execution source changed: ' + name)

    def local(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('path outside data root')
        return path

    for item in q['pinned_files'].values():
        if digest(local(item['path'])) != item['sha256']:
            raise ValueError('pinned experiment input changed')
    out, cache = local(q['output_dir']), local(q['cache_dir'])
    out.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock_handle = (cache/'worker.lock').open('a')
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'result.json').exists():
        raise ValueError('run already has a terminal result; refusing to restart')
    launch = out / 'launch.json'
    if launch.exists():
        if not resume or read(launch)['protocol_sha256'] != expected_digest:
            raise ValueError('existing run requires an identical explicit resume')
        started = read(launch)['started_unix']
    else:
        if resume:
            raise ValueError('cannot resume a run that never started')
        started = time.time()
        write(launch, {'started_unix': started, 'started_utc': datetime.now(timezone.utc).isoformat(),
                      'deadline_unix': started + q['wall_seconds'], 'protocol_sha256': expected_digest}, immutable=True)
    guard = Guard(cache, started + q['wall_seconds'], q['disk_free_floor_bytes'])
    status = {'protocol_sha256': expected_digest, 'steps': 0, 'epochs_completed': 0}

    def progress(phase, **values):
        status.update(phase=phase, **values, updated_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.time()-started)
        write(out/'progress.json', status)
        print(json.dumps(status, sort_keys=True), flush=True)

    def terminate(signum, frame):
        raise BudgetStop('supervisor_signal_' + str(signum))

    signal.signal(signal.SIGTERM, terminate); signal.signal(signal.SIGINT, terminate)
    try:
        guard(); progress('loading_verified_inputs')
        manifest = read(local(q['input_manifest']))
        dataset = load_dataset(root, manifest)
        if dataset.fingerprint != q['dataset_fingerprint']:
            raise ValueError('dataset identity changed')
        if [len(dataset.ids(p)) for p in ('train','validation')] != q['expected_partition_sizes']:
            raise ValueError('active partition size changed')
        ds = GuardedDataset(dataset, guard)
        torch = _load_torch()
        if torch is None or not torch.cuda.is_available():
            raise ValueError('CUDA training runtime unavailable')
        if str(torch.__version__) != q['torch_version']:
            raise ValueError('pinned torch runtime changed')
        torch.set_num_threads(4); _configure_device_runtime(torch, 'cuda'); torch.manual_seed(q['seed'])
        model = _build_model(torch, q['model_contract'], attention_chunk_size=q['attention_chunk_size'],
                             activation_checkpointing=q['activation_checkpointing']).to('cuda')
        if sum(p.numel() for p in model.parameters()) != q['model_contract']['estimated_parameters']:
            raise ValueError('model size changed')
        optimizer = torch.optim.AdamW(model.parameters(), **q['optimizer_kwargs'])
        runtime = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(), 'parameter_count':sum(p.numel() for p in model.parameters())}
        progress('runtime_ready', runtime=runtime)
        checkpoints = cache/'checkpoints'; latest = checkpoints/'latest.json'
        baseline = out/'epoch00-baseline.json'
        if not baseline.exists():
            if latest.exists(): raise ValueError('baseline missing for existing trained checkpoint')
            nll, tokens = 0., 0
            progress('epoch_zero_validation')
            for i, rid in enumerate(ds.ids('validation')):
                m = score_example(torch, model, ds.example(rid, partition='validation'), device='cuda')
                nll += m['nll_sum']; tokens += m['loss_tokens']
                if (i+1)%32 == 0: progress('epoch_zero_validation', validation_records=i+1)
            write(baseline, {'validation_token_nll':nll/tokens,'tokens':tokens,
                            'records':len(ds.ids('validation')), 'optimizer_steps':0,
                            'protocol_sha256':expected_digest}, immutable=True)
        elif read(baseline)['protocol_sha256'] != expected_digest:
            raise ValueError('baseline identity changed')
        records = {r['id']:r for r in map(json.loads,local(manifest['files']['source']['path']).read_text().splitlines())
                   if r['data_split']=='DEV' and r['id'] in dataset.ids('validation')}
        windows = {}
        for w in map(json.loads,local(manifest['files']['windows']['path']).read_text().splitlines()):
            if w['record_id'] in records: windows.setdefault(w['record_id'],[]).append(w)
        tokenizer = StudentTokenizer.load(local(manifest['files']['tokenizer']['path']))
        panel = read(local(q['factuality_panel'])); panel_digest = digest(local(q['factuality_panel']))
        if not {e['record_id'] for e in panel['entries']} <= records.keys():
            raise ValueError('quality panel outside frozen validation partition')
        for entry in panel['entries']:
            if entry['mismatched_input_record_id'] not in records:
                raise ValueError('control outside frozen validation partition')
        references = {rid:dataset.reference(rid)['target_message'] for rid in records}

        def predictions(epoch, *, control=False):
            checkpoint_sha = read(latest)['state_sha256']
            prefix = f'epoch{epoch:02d}-' + ('control' if control else 'validation')
            results = []
            pairs = [(e['record_id'],e['mismatched_input_record_id']) for e in panel['entries']] if control else [(rid,rid) for rid in ds.ids('validation')]
            progress(prefix, evaluation_completed=0)
            for i, (rid, source_id) in enumerate(pairs):
                guard(); dest = cache/'predictions'/prefix/(hashlib.sha256(rid.encode()).hexdigest()+'.json')
                if dest.exists():
                    saved = read(dest)
                    if saved['checkpoint_sha256'] != checkpoint_sha or saved['protocol_sha256'] != expected_digest:
                        raise ValueError('saved prediction identity changed')
                    result = saved['prediction']
                else:
                    result = predict_record(torch, model, records[source_id], tokenizer, device='cuda',
                        windows=windows[source_id], max_new_tokens=256, max_cache_bytes=q['inference_cpu_cache_bytes'])
                    result.update(record_id=rid, input_record_id=source_id)
                    write(dest, {'checkpoint_sha256':checkpoint_sha,'protocol_sha256':expected_digest,'prediction':result}, immutable=True)
                if result['record_id'] != rid or result['input_record_id'] != source_id:
                    raise ValueError('prediction source identity changed')
                results.append(result)
                if (i+1)%16 == 0: progress(prefix, evaluation_completed=i+1)
            dest=out/(prefix+'.json')
            write(dest, {'checkpoint_sha256':checkpoint_sha, 'protocol_sha256':expected_digest,
                         'predictions':results,'metrics':prediction_metrics(results,references)}, immutable=True)
            return dest

        # Resume validates the entire run identity before any resumed evaluation.
        options = dict(device='cuda', epochs=q['epochs'], seed=q['seed'], checkpoint_dir=checkpoints,
                       run_contract=q, training_authorized=True, checkpoint_every=q['checkpoint_every'],
                       keep_checkpoints=q['keep_checkpoints'], max_grad_norm=q['max_grad_norm'])
        if latest.exists():
            saved=read(latest)
            # run_epochs(max_steps=...) would train immediately. Restore with the
            # already-pinned identity after validating it against our saved progress.
            from gitctx.reviewed_training import _restore
            identity_file=out/'run-identity.json'
            if not identity_file.exists() or read(identity_file)['protocol_sha256'] != expected_digest:
                raise ValueError('resume identity is missing')
            state=_restore(torch,checkpoints,model,optimizer,read(identity_file)['identity'],'cuda')
            status.update(steps=state['steps'],epochs_completed=state['epoch'])
            metrics=state['metrics']
        else: metrics=[]
        while True:
            guard()
            epoch=status['epochs_completed']
            if epoch:
                prediction_file=predictions(epoch)
                m=read(prediction_file)['metrics']; stop=None
                if m['dominant_header_fraction'] > .8: stop='dominant_header_above_80_percent'
                if epoch>1 and metrics[-1]['validation_token_nll'] > 1.1*min(v['validation_token_nll'] for v in metrics[:-1]):
                    stop='validation_nll_worsened_over_10_percent'
                if stop or epoch==q['epochs']:
                    predictions(epoch,control=True)
                    progress('stopped_quality' if stop else 'completed', stop_reason=stop, metrics=metrics)
                    write(out/'result.json', status, immutable=True); return status
                review_file=out/f'epoch{epoch:02d}-review.json'
                progress('awaiting_factuality_review', review_file=str(review_file.relative_to(root)),
                         predictions_sha256=digest(prediction_file), panel_sha256=panel_digest)
                while not review_file.exists(): guard(); time.sleep(30)
                accuracy=review_accuracy(read(review_file),predictions_sha256=digest(prediction_file),
                    panel_sha256=panel_digest,panel=panel,records=records)
                if accuracy < .4:
                    predictions(epoch,control=True)
                    progress('stopped_quality',stop_reason='panel_accuracy_below_40_percent',panel_accuracy=accuracy,metrics=metrics)
                    write(out/'result.json',status,immutable=True); return status
            target=(epoch+1)*len(ds.ids('train'))
            while status['steps'] < target:
                guard(); progress('training', target_steps=q['epochs']*len(ds.ids('train')))
                # First successful update is checkpointed immediately; thereafter
                # bounded chunks permit supervision without altering shuffle/state.
                remaining=target-status['steps']
                chunk=1 if not latest.exists() else min(q['checkpoint_every']-(status['steps']%q['checkpoint_every']),remaining)
                result=run_epochs(torch,ds,model,optimizer,resume=latest.exists(),max_steps=chunk,**options)
                write(out/'run-identity.json',{'identity':result['identity'],'protocol_sha256':expected_digest},immutable=True)
                metrics=result['metrics']
                progress('checkpoint_saved',steps=result['steps'],epochs_completed=result['epochs_completed'],metrics=metrics,
                         checkpoint=read(latest),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BudgetStop as exc:
        progress('stopped_budget',stop_reason=str(exc))
        write(out/'result.json',status,immutable=True); return status
    except Exception as exc:
        progress('failed',error_type=type(exc).__name__,error=str(exc))
        write(out/'failure.json',status)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True);parser.add_argument('--protocol',required=True)
    parser.add_argument('--protocol-sha256',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    execute(args.data_root,args.protocol,args.protocol_sha256,resume=args.resume)


if __name__=='__main__':main()
