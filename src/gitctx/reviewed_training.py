"""Bounded, resumable epochs over the reviewed joint-window dataset.

Real training needs separately authorized resources and a passed readiness audit.
This runner never selects a budget, source split or reference on the user's behalf.
"""
from pathlib import Path
import hashlib
import json
import math
import random
import uuid

from gitctx.reference_overlay import artifact_hash
from gitctx.reviewed_dataset import backward_example, score_example

VERSION = 'reviewed-joint-window-trainer-v1'


def _hash(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _save(torch, directory, model, optimizer, state, identity, device, keep_checkpoints):
    directory.mkdir(parents=True, exist_ok=True)
    prior = []
    latest = directory / 'latest.json'
    if latest.exists():
        previous = json.loads(latest.read_text())
        if previous.get('identity') != identity:
            raise ValueError('checkpoint directory belongs to a different run')
        prior = previous.get('retained_states', [{'state_file': previous['state_file'],
                                                'state_sha256': previous['state_sha256']}])
    # Immutable step files make an interrupted latest-manifest update recoverable.
    # An interrupted save can leave either a partial file or an unreferenced
    # complete state. Preserve those bytes and allow replay from latest.json
    # to save the same logical step under a fresh attempt identity.
    stem = f"step-{state['steps']:08d}-epoch-{state['epoch']:02d}-{uuid.uuid4().hex}"
    path = directory / (stem + '.pt')
    temporary = directory / (stem + '.partial')
    payload = {'identity': identity, 'state': state, 'model': model.state_dict(),
               'optimizer': optimizer.state_dict(), 'rng_cpu': torch.get_rng_state(),
               'rng_cuda': torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None}
    with temporary.open('xb') as handle:
        torch.save(payload, handle)
    if path.exists():
        temporary.unlink()
        raise ValueError('checkpoint step already exists')
    temporary.replace(path)
    retained = prior + [{'state_file': path.name, 'state_sha256': _hash(path)}]
    discarded, retained = retained[:-keep_checkpoints], retained[-keep_checkpoints:]
    manifest = {'retained_states': retained, 'version': VERSION, 'identity': identity, 'state_file': path.name,
                'state_sha256': _hash(path), 'steps': state['steps'], 'epoch': state['epoch']}
    target = directory / 'latest.json'
    temporary_manifest = directory / (stem + '.manifest.partial')
    with temporary_manifest.open('x') as handle:
        json.dump(manifest, handle, indent=2); handle.write('\n')
    temporary_manifest.replace(target)
    # Delete only files listed by this run's previously successful manifest,
    # after the new checkpoint is authoritative. Never sweep directory globs.
    for item in discarded:
        old_name = item['state_file']
        old = directory / old_name
        if (Path(old_name).name != old_name or not old.resolve().is_relative_to(directory.resolve())
                or old_name == path.name):
            raise ValueError('invalid retained checkpoint path')
        if old.exists():
            if _hash(old) != item['state_sha256']:
                raise ValueError('old checkpoint changed; refusing to delete it')
            old.unlink()


def _restore(torch, directory, model, optimizer, identity, device):
    manifest = json.loads((directory / 'latest.json').read_text())
    if manifest.get('version') != VERSION or manifest.get('identity') != identity:
        raise ValueError('checkpoint run identity changed')
    name = manifest['state_file']
    path = directory / name
    if Path(name).name != name or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError('invalid checkpoint path')
    if _hash(path) != manifest['state_sha256']:
        raise ValueError('checkpoint bytes changed')
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload['identity'] != identity:
        raise ValueError('checkpoint payload identity changed')
    state = payload['state']
    if state['steps'] != manifest['steps'] or state['epoch'] != manifest['epoch']:
        raise ValueError('checkpoint cursor differs from manifest')
    model.load_state_dict(payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    torch.set_rng_state(payload['rng_cpu'].cpu())
    if payload['rng_cuda'] is not None:
        torch.cuda.set_rng_state(payload['rng_cuda'].cpu(), device)
    return state


def run_epochs(torch, dataset, model, optimizer, *, device, epochs, seed,
               checkpoint_dir, run_contract, training_authorized=False,
               resume=False, max_steps=None, checkpoint_every=100, max_grad_norm=1.0, keep_checkpoints=2):
    """One optimizer update per complete commit; window count never reweights it.

    ``max_steps`` bounds this invocation, not the total resumable experiment.
    ``run_contract`` must bind model configuration, software and readiness inputs.
    Toy tests may authorize their fixture runs; real data requires user approval.
    """
    if not training_authorized:
        raise ValueError('training execution requires explicit authorization')
    if type(epochs) is not int or epochs not in (1, 2, 4):
        raise ValueError('select a bounded 1/2/4-epoch run')
    if type(seed) is not int or not run_contract:
        raise ValueError('seed and pinned run contract required')
    if type(checkpoint_every) is not int or checkpoint_every <= 0:
        raise ValueError('checkpoint interval must be positive')
    if type(keep_checkpoints) is not int or keep_checkpoints < 2:
        raise ValueError('retain at least the current and previous checkpoints')
    if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
        raise ValueError('invocation step limit must be positive')
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError('invalid gradient clipping limit')
    ids, validation = dataset.ids('train'), dataset.ids('validation')
    if not ids or not validation:
        raise ValueError('both frozen active partitions are required')
    directory = Path(checkpoint_dir)
    identity = artifact_hash({'version': VERSION, 'dataset': dataset.fingerprint,
        'epochs': epochs, 'seed': seed, 'run_contract': run_contract,
        'max_grad_norm': max_grad_norm, 'keep_checkpoints': keep_checkpoints,
        'checkpoint_every': checkpoint_every,
        'optimizer_class': type(optimizer).__module__ + '.' + type(optimizer).__qualname__,
        'optimizer_groups': [{k: v for k, v in group.items() if k != 'params'} for group in optimizer.param_groups],
        'parameters': [(name, list(p.shape), str(p.dtype)) for name, p in model.named_parameters()]})
    if resume:
        state = _restore(torch, directory, model, optimizer, identity, device)
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('checkpoint directory is not empty; resume or select a new run')
        state = dict(epoch=0, cursor=0, steps=0, train_nll=0.0, train_tokens=0, metrics=[])
    if (not 0 <= state['epoch'] <= epochs or not 0 <= state['cursor'] < len(ids)
            or state['steps'] != state['epoch'] * len(ids) + state['cursor']
            or len(state['metrics']) != state['epoch']):
        raise ValueError('invalid resumable epoch cursor')
    invocation_steps = 0
    model.train()
    while state['epoch'] < epochs:
        order = list(ids)
        random.Random(seed + state['epoch']).shuffle(order)
        example = dataset.example(order[state['cursor']], partition='train')
        if example['dataset_fingerprint'] != dataset.fingerprint:
            raise ValueError('example dataset identity changed')
        optimizer.zero_grad(set_to_none=True)
        try:
            result = backward_example(torch, model, example, device=device)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            if not math.isfinite(result['loss']) or not bool(torch.isfinite(norm)):
                raise ValueError('non-finite training update')
        except Exception:
            optimizer.zero_grad(set_to_none=True)
            raise
        optimizer.step()
        state['steps'] += 1; state['cursor'] += 1; invocation_steps += 1
        state['train_nll'] += result['loss'] * result['loss_tokens']
        state['train_tokens'] += result['loss_tokens']
        completed_epoch = state['cursor'] == len(ids)
        if completed_epoch:
            nll, tokens = 0.0, 0
            for rid in validation:
                metric = score_example(torch, model, dataset.example(rid, partition='validation'), device=device)
                nll += metric['nll_sum']; tokens += metric['loss_tokens']
            state['metrics'].append({'epoch': state['epoch']+1,
                'train_token_nll': state['train_nll']/state['train_tokens'],
                'validation_token_nll': nll/tokens, 'train_tokens': state['train_tokens'],
                'validation_tokens': tokens, 'train_records': len(ids), 'validation_records': len(validation)})
            state.update(epoch=state['epoch']+1, cursor=0, train_nll=0.0, train_tokens=0)
        limit = max_steps is not None and invocation_steps >= max_steps
        if completed_epoch or limit or state['steps'] % checkpoint_every == 0:
            _save(torch, directory, model, optimizer, state, identity, device, keep_checkpoints)
        if limit:
            break
    return {'version': VERSION, 'identity': identity, 'complete': state['epoch'] == epochs,
            'steps': state['steps'], 'epochs_completed': state['epoch'],
            'cursor': state['cursor'], 'metrics': state['metrics'], 'invocation_steps': invocation_steps}
