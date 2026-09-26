"""Join frozen active partitions, reviewed targets and lossless input groups.

This adapter never promotes historical automatic acceptance to human review.
Callers must pin/replay source files and supply the independently frozen review
selection and partitions. Constructing a dataset is not training authorization.
"""
from copy import deepcopy

from gitctx import window_mixture as wm
from gitctx.conventional import parse_commit_message
from gitctx.reference_overlay import artifact_hash
from gitctx.reference_review import sha
from gitctx.reviewed_references import resolve_reference
from gitctx.student_tokenizer import SPECIAL
from gitctx.student_input import student_messages


def _index(rows, key):
    result = {}
    for row in rows:
        rid = row[key]
        if rid in result:
            raise ValueError('duplicate ' + key)
        result[rid] = deepcopy(row)
    return result


class ReviewedDataset:
    """All active examples or failure; no silent filtering of unresolved labels.

    ``review_ids`` must come from the frozen selection, separately from the
    current index. ``artifacts`` maps record IDs to their exact audited review
    artifacts. Windows and optional group metadata are replayed when accessed.
    """
    def __init__(self, records, *, coverage, review_ids, review_index, artifacts,
                 tokenizer, windows=None, groups=None):
        records = _index(records, 'id')
        coverage = _index(coverage, 'record_id')
        review_index = _index(review_index, 'record_id')
        required = list(review_ids)
        if len(required) != len(set(required)):
            raise ValueError('duplicate review selection')
        active = {rid: row['partition'] for rid, row in coverage.items()
                  if row['partition'] in {'train', 'validation'}}
        if not active or not set(active) <= records.keys():
            raise ValueError('missing active source records')
        if set(required) != review_index.keys() or not set(required) <= active.keys():
            raise ValueError('review index differs from frozen selection')
        if not {rid for rid, part in active.items() if part == 'validation'} <= set(required):
            raise ValueError('every validation reference requires explicit review')
        if not set(artifacts) <= set(required):
            raise ValueError('unexpected review artifact')
        self._records = {rid: records[rid] for rid in active}
        self._partitions = active
        self._references = {}
        self._tokenizer = tokenizer
        self._windows = deepcopy(windows) if windows is not None else None
        self._groups = _index(groups, 'record_id') if groups is not None else None
        if self._windows is not None and not set(active) <= self._windows.keys():
            raise ValueError('missing active windows')
        if self._groups is not None and not set(active) <= self._groups.keys():
            raise ValueError('missing prepared groups')
        # Resolve every active target before exposing any training examples.
        for rid, partition in active.items():
            record = self._records[rid]
            if record.get('data_split') != 'DEV' or record.get('evaluation_only'):
                raise ValueError('protected source in active partition')
            if rid in review_index:
                reference = resolve_reference(record, review_index[rid], artifacts.get(rid, {}),
                                              partition=partition)
            else:
                if (record.get('teacher_license') != 'Apache-2.0'
                        or not record.get('teacher_model_id') or not record.get('teacher_revision')):
                    raise ValueError('missing original open-teacher provenance')
                target = record['target_message']
                parse_commit_message(target)
                reference = dict(record_id=rid, partition=partition, target_message=target,
                    target_sha256=sha(target), original_target_sha256=sha(target),
                    source_diff_sha256=sha(record['diff']), target_origin='original_open_teacher',
                    verification_kind='historical_automatic', independent_human_review=False,
                    teacher_model=record['teacher_model_id'], teacher_revision=record['teacher_revision'],
                    training_run_approved=False)
            answer = tokenizer.encode(reference['target_message']) + [SPECIAL['<eos>']]
            if len(answer) > wm.ANSWER_RESERVE:
                raise ValueError('target exceeds fixed answer reserve')
            if tokenizer.decode(answer[:-1]) != reference['target_message']:
                raise ValueError('target does not round-trip')
            self._references[rid] = reference
        self.fingerprint = artifact_hash({
            'method': wm.VERSION, 'partitions': self._partitions,
            'source_inputs': {rid: artifact_hash(student_messages(record)) for rid, record in self._records.items()},
            'references': self._references,
            'windows': self._windows, 'groups': self._groups,
            'tokenizer': tokenizer.backend.to_str(),
        })

    def ids(self, partition):
        if partition not in {'train', 'validation'}:
            raise ValueError('only active train/validation partitions are exposed')
        return tuple(sorted(rid for rid, part in self._partitions.items() if part == partition))

    def reference(self, record_id):
        return deepcopy(self._references[record_id])

    def example(self, record_id, *, partition):
        if self._partitions.get(record_id) != partition or partition not in {'train', 'validation'}:
            raise ValueError('record outside requested partition')
        record = self._records[record_id]
        windows = None if self._windows is None else self._windows[record_id]
        if windows is not None and any(w.get('partition') != partition for w in windows):
            raise ValueError('window partition mismatch')
        group = wm.prepare(record, self._tokenizer, windows=windows)
        if self._groups is not None:
            frozen = self._groups[record_id]
            if (frozen.get('partition') != partition or frozen.get('target_assigned') is not False
                    or frozen.get('group_sha256') != artifact_hash(group)
                    or frozen.get('window_ids') != group['window_ids']):
                raise ValueError('prepared input group changed')
        reference = self._references[record_id]
        answer = self._tokenizer.encode(reference['target_message']) + [SPECIAL['<eos>']]
        return {'group': group, 'reference': deepcopy(reference), 'answer': answer,
                'partition': partition, 'dataset_fingerprint': self.fingerprint}

    def examples(self, partition, *, order=None):
        ids = self.ids(partition)
        selected = ids if order is None else tuple(order)
        if len(selected) != len(ids) or set(selected) != set(ids):
            raise ValueError('epoch must contain every partition example exactly once')
        for rid in selected:
            yield self.example(rid, partition=partition)


def backward_example(torch, model, example, *, device):
    """The trainer owns optimizer actions; validation examples cannot backpropagate."""
    if example['partition'] != 'train' or example['reference']['partition'] != 'train':
        raise ValueError('only training examples may backpropagate')
    return wm.backward_joint(torch, model, example['group']['prompts'], example['answer'], device=device)


def score_example(torch, model, example, *, device):
    """Token-weightable likelihood of the exact joint objective; no parameter updates."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            scores = torch.stack([wm.answer_log_probs(torch, model, p, example['answer'], device=device).cpu()
                                  for p in example['group']['prompts']])
            if not bool(torch.isfinite(scores).all()):
                raise ValueError('non-finite evaluation scores')
            loss = -wm.joint_log_probs(torch, scores).sum()
            return {'nll_sum': float(loss), 'loss_tokens': len(example['answer']),
                    'window_count': len(example['group']['prompts'])}
    finally:
        model.train(was_training)


def predict_record(torch, model, record, tokenizer, *, device, windows=None, **generation_options):
    """Inference uses only source input; labels/review artifacts are never consulted."""
    group = wm.prepare(record, tokenizer, windows=windows)
    tokens, reason = wm.generate(torch, model, group['prompts'], device=device, **generation_options)
    # Byte-level decoding can end in incomplete UTF-8 at the token limit. Report
    # this explicitly rather than silently replacing bytes or deleting tokens.
    try:
        message = tokenizer.decode(tokens)
        error = None
    except (UnicodeDecodeError, ValueError) as exc:
        message, error = None, type(exc).__name__
    return {'record_id': record['id'], 'token_ids': tokens, 'message': message,
            'stop_reason': reason, 'decode_error': error, 'window_count': len(group['prompts'])}
