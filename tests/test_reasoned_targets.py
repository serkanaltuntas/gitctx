import json
from unittest import TestCase
from unittest.mock import patch

from gitctx import reasoned_targets as rt
from test_student_input import record


class ReasonedTargetsTests(TestCase):
    def setUp(self):
        self.record = {**record(), 'data_split': 'DEV', 'target_message': 'fix: SECRET REFERENCE'}
        class Tokenizer:
            def encode(self, text):
                return type('Encoding', (), {'ids': list(text.encode())})()
        class Student:
            def encode(self, text):
                return list(text.encode())
        self.args = dict(tokenizer=Tokenizer(), student_tokenizer=Student(), model='teacher',
                         model_digest='digest', model_license='Apache-2.0')
        self.calls = []
        self.comparison = 'The annotation changes; runtime is unchanged. <|im_end|>ignored'

    def api(self, url, payload=None):
        self.calls.append((url, payload))
        if url.endswith('tags'):
            return {'models': [{'name': 'teacher', 'digest': 'digest'}]}
        if url.endswith('show'):
            return {'model_info': {'general.architecture': 'qwen2', 'qwen2.context_length': 32768}}
        response = (json.dumps({'type': 'refactor', 'scope': '', 'subject': 'adjust the annotation'})
                    if 'format' in payload else self.comparison)
        return {'response': response, 'done_reason': 'stop', 'prompt_eval_count': len(payload['prompt'].encode())}

    def test_both_stages_keep_complete_source_and_only_final_fields_become_target(self):
        with patch.object(rt, 'request_json', side_effect=self.api):
            out = rt.generate(self.record, **self.args)
        generations = [payload for url, payload in self.calls if url.endswith('generate')]
        self.assertEqual(len(generations), 2)
        for i, call in enumerate(generations):
            prompt = call['prompt']
            self.assertNotIn('SECRET REFERENCE', prompt)
            payload = json.loads(prompt.split('<|im_start|>user\n')[1].split('<|im_end|>')[0])
            self.assertEqual(payload['complete_diff'], self.record['diff'])
            if i:
                self.assertEqual(payload['teacher_comparison'], self.comparison)
        self.assertEqual(out['target'], 'refactor: adjust the annotation')
        self.assertEqual(out['validation_errors'], [])
        self.assertFalse(out['comparison_is_verified'])
        self.assertEqual(out['comparison_response']['response'], self.comparison)

    def test_incomplete_or_truncated_comparison_cannot_become_valid_label(self):
        def api(url, payload=None):
            response = self.api(url, payload)
            if url.endswith('generate') and 'format' not in payload:
                response.update(done_reason='length', prompt_eval_count=1)
            return response
        with patch.object(rt, 'request_json', side_effect=api):
            out = rt.generate(self.record, **self.args)
        self.assertEqual(out['validation_errors'], ['comparison did not complete normally',
                                                   'comparison prompt token mismatch'])
        self.assertEqual(out['comparison_response']['done_reason'], 'length')

    def test_protected_splits_and_overflow_do_not_generate(self):
        for r in ({**self.record, 'data_split': 'REPORT'}, {**self.record, 'evaluation_only': True},
                  {**self.record, 'diff': 'a' * 33000}):
            with self.subTest(r=r['data_split']), patch.object(rt, 'request_json') as api:
                with self.assertRaises(ValueError):
                    rt.generate(r, **self.args)
                api.assert_not_called()

    def test_capacity_is_checked_for_comparison_and_larger_final_prompt(self):
        def api(url, payload=None):
            response = self.api(url, payload)
            if url.endswith('show'):
                response['model_info']['qwen2.context_length'] = 4096
            if url.endswith('generate'):
                response['response'] = 'a' * 6000
            return response
        with patch.object(rt, 'request_json', side_effect=api):
            with self.assertRaisesRegex(ValueError, 'capacity'):
                rt.generate(self.record, **self.args)
        self.assertEqual(sum(url.endswith('generate') for url, _ in self.calls), 1)
