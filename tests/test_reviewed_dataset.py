from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

from gitctx import reviewed_dataset as rd, window_mixture as wm
from gitctx.reference_overlay import artifact_hash
from gitctx.reference_review import sha
from gitctx.student_input import student_messages
from gitctx.student_tokenizer import StudentTokenizer, SPECIAL
from gitctx.proof_lm_train import _build_model, _load_torch
from test_student_input import record


class ReviewedDatasetTests(TestCase):
    def setUp(self):
        self.train = {**record(), 'id': 'train', 'data_split': 'DEV',
                      'teacher_model_id': 'teacher', 'teacher_revision': 'original',
                      'teacher_license': 'Apache-2.0'}
        self.validation = {**self.train, 'id': 'validation'}
        self.coverage = [dict(record_id=r['id'], partition=r['id']) for r in (self.train, self.validation)]
        target = 'fix(a): use revised value'
        self.artifact = dict(record_id='validation', partition='validation',
            source_diff_sha256=sha(self.validation['diff']), original_target_sha256=sha(self.validation['target_message']),
            status='verified_overlay', replacement_message=target, replacement_sha256=sha(target),
            target_origin='licensed_open_teacher', teacher_license='Apache-2.0', teacher_model='teacher',
            teacher_revision='replacement', teacher_digest='digest', candidate_sha256='candidate',
            verification_sha256='review', verification_kind='assistant', verification_method='full_diff_review',
            independent_human_review=False, in_place_source_modified=False, eligible_for_training_partition=False)
        self.index = {k: self.artifact[k] for k in ('record_id','partition','source_diff_sha256','original_target_sha256')}
        self.index.update(reference_approved=True, status='assistant_verified_open_teacher_overlay',
                          verification_kind='assistant', artifact_sha256=artifact_hash(self.artifact))
        texts = [m['content'] for m in student_messages(self.train)] + [self.train['target_message']]
        self.tokenizer = StudentTokenizer.fit(texts, 400)

    def dataset(self, **kwargs):
        args = dict(coverage=self.coverage, review_ids=['validation'], review_index=[self.index],
                    artifacts={'validation': self.artifact}, tokenizer=self.tokenizer)
        args.update(kwargs)
        return rd.ReviewedDataset([self.train, self.validation], **args)

    def test_exact_partitions_provenance_and_overlay_without_source_mutation(self):
        original = deepcopy(self.validation)
        ds = self.dataset()
        self.assertEqual(ds.ids('train'), ('train',))
        self.assertEqual(ds.ids('validation'), ('validation',))
        a = ds.example('train', partition='train')
        b = ds.example('validation', partition='validation')
        self.assertEqual(a['reference']['verification_kind'], 'historical_automatic')
        self.assertEqual(b['reference']['verification_kind'], 'assistant')
        self.assertFalse(b['reference']['independent_human_review'])
        self.assertEqual(self.tokenizer.decode(b['answer'][:-1]), self.artifact['replacement_message'])
        self.assertEqual(b['answer'][-1], SPECIAL['<eos>'])
        self.assertEqual(self.validation, original)
        # Constructor isolates the accepted data from later caller mutations.
        self.validation['diff'] = 'changed'
        self.assertEqual(b, ds.example('validation', partition='validation'))
        b['reference']['target_message'] = 'forged'
        self.assertEqual(ds.reference('validation')['target_message'], self.artifact['replacement_message'])
        with self.assertRaises(ValueError): ds.example('validation', partition='train')
        with self.assertRaises(ValueError): list(ds.examples('train', order=[]))
        with self.assertRaises(ValueError): list(ds.examples('train', order=['train', 'train']))
        with self.assertRaises(ValueError): ds.ids('REPORT')

    def test_unresolved_or_deleted_selected_reference_never_falls_back_to_original(self):
        for changes in (dict(review_index=[]), dict(review_ids=[], review_index=[], artifacts={}),
                        dict(review_index=[{**self.index, 'reference_approved': False}]), dict(artifacts={}),
                        dict(review_ids=['validation','validation'])):
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.dataset(**changes)
        original_fingerprint = self.dataset().fingerprint
        self.train['source_repo_url'] += '/other'
        self.assertNotEqual(original_fingerprint, self.dataset().fingerprint)
        self.train['teacher_license'] = 'unknown'
        with self.assertRaises(ValueError): self.dataset()
        self.train['teacher_license'] = 'Apache-2.0'; self.train['data_split'] = 'REPORT'
        with self.assertRaises(ValueError): self.dataset()

    def test_every_window_is_in_one_joint_example_and_prepared_hash_is_enforced(self):
        class ByteTokenizer:
            backend = type('Backend', (), {'to_str': lambda _: 'byte-test'})()
            def encode(self, text): return [b+32 for b in text.encode()]
            def decode(self, ids): return bytes(i-32 for i in ids).decode()
        tokenizer = ByteTokenizer()
        self.train['diff'] += 'a' * 18000
        windows = {}
        groups = []
        for record in (self.train, self.validation):
            ws = wm.ew.build_windows(record, tokenizer)
            for w in ws: w['partition'] = record['id']
            windows[record['id']] = ws
            g = wm.prepare(record, tokenizer, windows=ws)
            groups.append(dict(record_id=record['id'], partition=record['id'], target_assigned=False,
                               group_sha256=artifact_hash(g), window_ids=g['window_ids']))
        ds = self.dataset(tokenizer=tokenizer, windows=windows, groups=groups)
        example = ds.example('train', partition='train')
        self.assertGreater(len(example['group']['prompts']), 1)
        self.assertEqual(len(example['group']['prompts']), len(windows['train']))
        self.assertNotIn('answer', example['group'])
        self.assertEqual(example['group']['source_bytes'], len(self.train['diff'].encode()))
        groups[0]['group_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            self.dataset(tokenizer=tokenizer, windows=windows, groups=groups).example('train', partition='train')
        windows['train'].pop()
        with self.assertRaises(ValueError):
            self.dataset(tokenizer=tokenizer, windows=windows).example('train', partition='train')

    def test_training_and_evaluation_use_same_loss_and_validation_cannot_backpropagate(self):
        torch = _load_torch()
        if torch is None: self.skipTest('torch unavailable')
        ds = self.dataset(); train = ds.example('train', partition='train')
        model = _build_model(torch, dict(tokenizer_vocab_size=self.tokenizer.vocab_size,
            hidden_size=16, layers=1, attention_heads=4, kv_heads=2, intermediate_size=32, context_tokens=8192),
            attention_chunk_size=32, activation_checkpointing=True)
        expected = rd.score_example(torch, model, train, device='cpu')
        self.assertTrue(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        actual = rd.backward_example(torch, model, train, device='cpu')
        self.assertAlmostEqual(actual['loss'], expected['nll_sum']/expected['loss_tokens'], places=5)
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        with self.assertRaises(ValueError):
            rd.backward_example(torch, model, ds.example('validation', partition='validation'), device='cpu')

    def test_inference_ignores_target_and_reports_incomplete_byte_output(self):
        captured = []
        def generate(torch, model, prompts, **kwargs):
            captured.append(prompts)
            return self.tokenizer.encode('fix: example'), 'stop_token'
        with patch.object(wm, 'generate', side_effect=generate):
            first = rd.predict_record(None, None, self.train, self.tokenizer, device='cpu')
            second = rd.predict_record(None, None, {**self.train, 'target_message':'SECRET'}, self.tokenizer, device='cpu')
        self.assertEqual(first, second)
        self.assertEqual(captured[0], captured[1])
        self.assertEqual(first['message'], 'fix: example')
        with patch.object(wm, 'generate', return_value=([SPECIAL['<user>']], 'token_limit')):
            invalid = rd.predict_record(None, None, self.train, self.tokenizer, device='cpu')
        self.assertIsNone(invalid['message'])
        self.assertEqual(invalid['decode_error'], 'ValueError')
