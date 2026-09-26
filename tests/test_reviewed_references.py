from copy import deepcopy
import unittest

from gitctx.reference_overlay import artifact_hash
from gitctx.reference_review import sha
from gitctx.reviewed_references import resolve_reference, materialize_reviewed_full
from gitctx.student_sequences import prompt_ids
from gitctx.student_tokenizer import SPECIAL
from test_student_input import record


class ReviewedReferenceTests(unittest.TestCase):
    def setUp(self):
        self.record = {**record(), 'data_split': 'DEV', 'teacher_model_id': 'teacher',
                       'teacher_revision': 'original', 'teacher_license': 'Apache-2.0'}
        r = self.record
        self.artifact = {'record_id': r['id'], 'partition': 'validation',
            'source_diff_sha256': sha(r['diff']), 'original_target_sha256': sha(r['target_message']),
            'status': 'verified_overlay', 'replacement_message': 'fix(a): use new value',
            'replacement_sha256': sha('fix(a): use new value'),
            'target_origin': 'licensed_open_teacher', 'teacher_license': 'Apache-2.0',
            'teacher_model': 'teacher', 'teacher_revision': 'replacement', 'teacher_digest': 'digest',
            'candidate_sha256': 'candidate', 'verification_sha256': 'review',
            'verification_kind': 'assistant', 'verification_method': 'full_diff_review',
            'independent_human_review': False, 'in_place_source_modified': False,
            'eligible_for_training_partition': False}
        self.index = {k: self.artifact[k] for k in
            ('record_id', 'partition', 'source_diff_sha256', 'original_target_sha256')}
        self.index.update(reference_approved=True, status='assistant_verified_open_teacher_overlay',
                          verification_kind='assistant', artifact_sha256=artifact_hash(self.artifact))
        class ByteTokenizer:
            def encode(self, text):
                return [int(x) + 32 for x in text.encode()]
        self.tokenizer = ByteTokenizer()

    def test_reviewed_target_preserves_input_and_answer_shift_boundary(self):
        before = deepcopy(self.record)
        sequence = materialize_reviewed_full(self.record, self.index, self.artifact,
                                            self.tokenizer, partition='validation')
        prompt = prompt_ids(self.record, self.tokenizer)
        answer = self.tokenizer.encode(self.artifact['replacement_message']) + [SPECIAL['<eos>']]
        self.assertEqual(sequence['input_ids'], prompt + answer)
        self.assertEqual(sequence['loss_mask'], [0] * len(prompt) + [1] * len(answer))
        self.assertEqual(sequence['input_ids'][len(prompt) - 1], SPECIAL['<assistant>'])
        self.assertEqual(self.record, before)
        self.assertFalse(sequence['reference']['independent_human_review'])
        self.assertFalse(sequence['reference']['training_run_approved'])

    def test_stale_source_target_artifact_and_partition_are_rejected(self):
        for mutation in ({'diff': self.record['diff'] + '\n'}, {'target_message': 'fix: other'},
                         {'data_split': 'REPORT'}, {'evaluation_only': True}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                resolve_reference({**self.record, **mutation}, self.index, self.artifact, partition='validation')
        for mutation in ({'reference_approved': False}, {'artifact_sha256': 'other'}, {'partition': 'train'}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                resolve_reference(self.record, {**self.index, **mutation}, self.artifact, partition='validation')
        for partition in ('train', 'reserved', 'REPORT'):
            with self.subTest(partition=partition), self.assertRaises(ValueError):
                resolve_reference(self.record, self.index, self.artifact, partition=partition)

    def test_forged_provenance_and_substituted_target_fail_even_with_new_artifact_hash(self):
        for mutation in ({'replacement_message': 'fix: edited by reviewer'}, {'teacher_license': 'unknown'},
                         {'independent_human_review': True}, {'verification_kind': 'automation'},
                         {'eligible_for_training_partition': True}, {'candidate_sha256': ''}):
            artifact = {**self.artifact, **mutation}
            index = {**self.index, 'artifact_sha256': artifact_hash(artifact)}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                resolve_reference(self.record, index, artifact, partition='validation')

    def test_retention_requires_exact_original_and_complete_claims(self):
        artifact = {**self.artifact, 'status': 'assistant_verified_original_reference',
                    'target_origin': 'original_open_teacher', 'decision': 'retain',
                    'reviewer_kind': 'assistant', 'method': 'full_diff_review',
                    'all_target_lines_considered': True, 'original_reference_modified': False,
                    'teacher_revision': 'original',
                    'claims': [{'text': s, 'decision': 'supported'} for s in
                               self.record['target_message'].splitlines() if s.strip()]}
        index = {**self.index, 'status': artifact['status'], 'artifact_sha256': artifact_hash(artifact)}
        result = resolve_reference(self.record, index, artifact, partition='validation')
        self.assertEqual(result['target_message'], self.record['target_message'])
        artifact['claims'] = []
        index['artifact_sha256'] = artifact_hash(artifact)
        with self.assertRaises(ValueError):
            resolve_reference(self.record, index, artifact, partition='validation')

    def test_full_materializer_never_assigns_whole_commit_to_partial_input(self):
        self.record['diff'] += 'x' * 9000
        self.artifact['source_diff_sha256'] = sha(self.record['diff'])
        self.index.update(source_diff_sha256=self.artifact['source_diff_sha256'],
                          artifact_sha256=artifact_hash(self.artifact))
        with self.assertRaisesRegex(ValueError, 'prompt overflow'):
            materialize_reviewed_full(self.record, self.index, self.artifact,
                                      self.tokenizer, partition='validation')
