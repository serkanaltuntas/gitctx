"""Load a pinned reviewed-data manifest before exposing the active dataset."""
from pathlib import Path
import hashlib
import json

from gitctx.reviewed_dataset import ReviewedDataset
from gitctx.student_tokenizer import StudentTokenizer

VERSION = 'reviewed-input-manifest-v1'


def load_dataset(root, manifest):
    """Hash every declared input, bind independent splits/selection, then resolve.

    The external readiness protocol must pin this manifest itself. A matching
    hash is an integrity check, not source-license or semantic-review approval.
    """
    root = Path(root).resolve()
    if manifest.get('version') != VERSION:
        raise ValueError('unsupported reviewed input manifest')
    files = manifest['files']
    verified = {}
    for name, item in files.items():
        path = root / item['path']
        if not path.resolve().is_relative_to(root):
            raise ValueError('input path escapes data root')
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != item['sha256']:
            raise ValueError('input hash mismatch: ' + name)
        verified[name] = path

    def value(name):
        return json.loads(verified[name].read_text())

    def rows(name):
        return [json.loads(line) for line in verified[name].read_text().splitlines() if line.strip()]

    required = {'source', 'coverage', 'tokenizer', 'student_manifest', 'split_protocol',
                'review_selection', 'review_index', 'windows', 'groups'}
    if not required <= verified.keys():
        raise ValueError('missing pinned input resources')
    student = value('student_manifest')
    if (student['inputs'].get(files['source']['path']) != files['source']['sha256']
            or student['outputs']['tokenizer.json'] != files['tokenizer']['sha256']
            or student['outputs']['coverage.jsonl'] != files['coverage']['sha256']):
        raise ValueError('student preparation provenance mismatch')
    split_path = files['split_protocol']['path']
    if student['inputs'].get(split_path) != files['split_protocol']['sha256']:
        raise ValueError('independent split protocol mismatch')
    coverage = rows('coverage')
    if ({r['record_id'] for r in coverage if r['partition'] == 'train'} != set(student['tokenizer_fit_ids'])
            or {r['record_id'] for r in coverage if r['partition'] == 'validation'} != set(value('split_protocol')['validation_ids'])):
        raise ValueError('frozen active partition mismatch')
    selection = rows('review_selection')
    index = rows('review_index')
    artifacts = {}
    by_path = {item['path']: name for name, item in files.items()}
    artifact_cache = {}
    for row in index:
        if not row['reference_approved']:
            continue
        path = row['artifact_file']
        if path not in by_path:
            raise ValueError('review artifact is not pinned')
        if path not in artifact_cache:
            entries = rows(by_path[path])
            keyed = {a['record_id']: a for a in entries}
            if len(keyed) != len(entries):
                raise ValueError('duplicate artifact identity')
            artifact_cache[path] = keyed
        artifacts[row['record_id']] = artifact_cache[path][row['record_id']]
    windows = {}
    for row in rows('windows'):
        windows.setdefault(row['record_id'], []).append(row)
    return ReviewedDataset(rows('source'), coverage=coverage,
        review_ids=[r['record_id'] for r in selection], review_index=index, artifacts=artifacts,
        tokenizer=StudentTokenizer.load(verified['tokenizer']), windows=windows, groups=rows('groups'))
