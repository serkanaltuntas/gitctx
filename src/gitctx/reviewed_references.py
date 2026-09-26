"""Resolve audited reference artifacts without changing the source dataset.

The caller must pin and audit the status index and artifact collection. Hash
binding prevents mixing revisions; it does not establish semantic correctness
or authorize a training run. Partial-window targets need their own alignment.
"""
from gitctx.conventional import parse_commit_message
from gitctx.reference_overlay import artifact_hash
from gitctx.reference_review import sha
from gitctx.student_sequences import materialize


def resolve_reference(record, index_row, artifact, *, partition):
    """Select one original/overlay target against a frozen partition assignment."""
    if record.get('data_split') != 'DEV' or record.get('evaluation_only'):
        raise ValueError('only real DEV references are eligible')
    if partition not in {'train', 'validation'}:
        raise ValueError('reference is outside the active partition')
    if index_row.get('reference_approved') is not True:
        raise ValueError('unresolved reference')
    for item in (index_row, artifact):
        if (item.get('record_id') != record['id']
                or item.get('source_diff_sha256') != sha(record['diff'])
                or item.get('original_target_sha256') != sha(record['target_message'])
                or item.get('partition') != partition):
            raise ValueError('reference source or frozen partition mismatch')
    if index_row.get('artifact_sha256') != artifact_hash(artifact):
        raise ValueError('review artifact changed')
    if not artifact.get('verification_sha256'):
        raise ValueError('missing review lineage')
    kind = index_row.get('verification_kind')
    if kind not in {'assistant', 'human'}:
        raise ValueError('explicit review kind required')
    if artifact.get('independent_human_review') is not (kind == 'human'):
        raise ValueError('review provenance mismatch')
    if (artifact.get('teacher_license') != 'Apache-2.0'
            or not artifact.get('teacher_model') or not artifact.get('teacher_revision')):
        raise ValueError('missing reviewed teacher provenance')

    if artifact.get('status') == 'verified_overlay':
        if (index_row.get('status') != f'{kind}_verified_open_teacher_overlay'
                or artifact.get('target_origin') != 'licensed_open_teacher'
                or artifact.get('verification_kind') != kind
                or artifact.get('verification_method') != 'full_diff_review'
                or not artifact.get('candidate_sha256') or not artifact.get('teacher_digest')
                or artifact.get('in_place_source_modified') is not False
                or artifact.get('eligible_for_training_partition') is not (partition == 'train')):
            raise ValueError('invalid reviewed overlay')
        target = artifact.get('replacement_message')
        if not isinstance(target, str) or artifact.get('replacement_sha256') != sha(target):
            raise ValueError('replacement target changed')
    elif artifact.get('status') == f'{kind}_verified_original_reference':
        if (index_row.get('status') != artifact['status']
                or artifact.get('target_origin') != 'original_open_teacher'
                or artifact.get('decision') != 'retain' or artifact.get('reviewer_kind') != kind
                or artifact.get('method') != 'full_diff_review'
                or artifact.get('all_target_lines_considered') is not True
                or artifact.get('original_reference_modified') is not False
                or artifact['teacher_model'] != record.get('teacher_model_id')
                or artifact['teacher_revision'] != record.get('teacher_revision')
                or artifact['teacher_license'] != record.get('teacher_license')):
            raise ValueError('invalid original-reference retention')
        target = record['target_message']
        claims = artifact.get('claims', [])
        if ([c.get('text') for c in claims] != [s for s in target.splitlines() if s.strip()]
                or any(c.get('decision') != 'supported' for c in claims)):
            raise ValueError('retention does not cover the complete target')
    else:
        raise ValueError('unsupported review artifact')
    parse_commit_message(target)
    return {'record_id': record['id'], 'partition': partition, 'target_message': target,
            'target_sha256': sha(target), 'original_target_sha256': sha(record['target_message']),
            'source_diff_sha256': sha(record['diff']), 'artifact_sha256': artifact_hash(artifact),
            'target_origin': artifact['target_origin'], 'verification_kind': kind,
            'teacher_model': artifact['teacher_model'], 'teacher_revision': artifact['teacher_revision'],
            'independent_human_review': kind == 'human', 'training_run_approved': False}


def materialize_reviewed_full(record, index_row, artifact, tokenizer, *, partition):
    """Use the common full-input format and answer mask; overflow still fails."""
    reference = resolve_reference(record, index_row, artifact, partition=partition)
    sequence = materialize({**record, 'target_message': reference['target_message']}, tokenizer)
    return {**sequence, 'reference': reference}
