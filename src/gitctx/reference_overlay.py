"""Build immutable reference overlays from open-teacher text and explicit review.

The caller is responsible for a trustworthy semantic attestation. This checks
its binding and provenance; it does not infer factual approval from model output.
"""
import json
from gitctx.conventional import parse_commit_message, DEFAULT_TYPES
from gitctx.reference_review import sha
from gitctx.student_sequences import physical_lines, diff_units
from gitctx.teacher_response import decode_response


def artifact_hash(value):
    return sha(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')))


def build_override(record,candidate,attestation,tokenizer,*,teacher_revision):
    if record.get('evaluation_only') or record.get('data_split')!='DEV':
        raise ValueError('only real DEV references can receive overlays')
    target=candidate.get('target')
    if (candidate.get('record_id')!=record['id'] or candidate.get('source_diff_sha256')!=sha(record['diff'])
        or candidate.get('target_origin')!='licensed_open_teacher' or candidate.get('model_license')!='Apache-2.0'
        or candidate.get('validation_errors') or not candidate.get('model_digest') or not teacher_revision
        or not isinstance(target,str) or candidate.get('target_sha256')!=sha(target)):
        raise ValueError('candidate source/provenance mismatch')
    # Reference text must come from the teacher's raw fields or plain response,
    # never a replacement string authored by the reviewer.
    fields,expected=decode_response(candidate['response']['response'],candidate.get('output_format','json'))
    if fields!=candidate.get('fields') or fields.get('type') not in DEFAULT_TYPES:
        raise ValueError('teacher fields changed')
    if target!=expected:raise ValueError('replacement differs from teacher output')
    parse_commit_message(target)
    if len(tokenizer.encode(target))+1>256:raise ValueError('answer reserve exceeded')
    if (attestation.get('decision')!='accept' or attestation.get('method')!='full_diff_review'
        or attestation.get('reviewer_kind') not in {'assistant','human'}
        or attestation.get('candidate_sha256')!=artifact_hash(candidate)
        or attestation.get('source_diff_sha256')!=sha(record['diff'])
        or not attestation.get('note') or not attestation.get('timestamp')):
        raise ValueError('missing explicit full-source semantic review')
    claims=attestation.get('claims',[])
    if [c.get('text') for c in claims]!=[l for l in target.splitlines() if l.strip()]:
        raise ValueError('review must cover every target line')
    lines=physical_lines(record['diff'])
    changed={i+1 for u in diff_units(record['diff']) for i in u['changed_line_indices']}
    for claim in claims:
        if claim.get('decision')!='supported':raise ValueError('unverified target claim')
        ids=[]
        for e in claim.get('evidence',[]):
            i,q=e.get('line'),e.get('quote')
            if type(i) is not int or not 1<=i<=len(lines) or not isinstance(q,str) or not q or q not in lines[i-1]:
                raise ValueError('invalid source evidence')
            ids.append(i)
        if not changed.intersection(ids):raise ValueError('no changed-line evidence')
    return {'record_id':record['id'],'source_diff_sha256':sha(record['diff']),
        'original_target_sha256':sha(record['target_message']),'replacement_message':target,
        'replacement_sha256':sha(target),'target_origin':'licensed_open_teacher',
        'teacher_model':candidate['model'],'teacher_digest':candidate['model_digest'],
        'teacher_revision':teacher_revision,'teacher_license':'Apache-2.0',
        'candidate_sha256':artifact_hash(candidate),'verification_sha256':artifact_hash(attestation),
        'verification_kind':attestation['reviewer_kind'],'verification_method':'full_diff_review',
        'independent_human_review':attestation['reviewer_kind']=='human',
        'status':'verified_overlay','in_place_source_modified':False}
