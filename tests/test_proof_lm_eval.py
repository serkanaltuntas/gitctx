from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from gitctx.proof_lm_eval import decode_tokens, decode_tokens_legacy, evaluate, prompt_tokens
from gitctx.conventional import parse_commit_message
from gitctx.proof_lm_train import _load_torch, run_proof_lm_training
from gitctx.proof_tokenizer import tokenize_text
from gitctx.proof_train_job import proof_trainer_job_path
from test_proof_lm_train import _prepare_trainer_job, _record


class ProofLmEvalTests(unittest.TestCase):
    def test_gold_content_and_length_do_not_change_prompt(self):
        record = _record('test', 'REPORT', 'fix(api): short')
        first = prompt_tokens(record, context_tokens=64, max_new_tokens=16)
        record['messages'][-1]['content'] = 'SECRET_REFERENCE ' * 1000
        second = prompt_tokens(record, context_tokens=64, max_new_tokens=16)
        self.assertEqual(first, second)
        self.assertNotIn('SECRET_REFERENCE', second)
        self.assertLessEqual(len(second), 48)
        record['messages'][1]['content'] = 'begin ' + 'middle ' * 1000 + 'end'
        cropped = prompt_tokens(record, context_tokens=64, max_new_tokens=16)
        self.assertLessEqual(len(cropped), 48)
        self.assertIn('begin', cropped)
        self.assertIn('end', cropped)

    def test_fixed_whitespace_reconstruction(self):
        message = 'fix(api): handle missing path\n\nExplain parser behavior.'
        self.assertEqual(decode_tokens(tokenize_text(message)), message)

    def test_scope_paths_packages_and_breaking_marker_roundtrip(self):
        for scope in ('test_web_urldispatcher.py', 'tokio/src/sync/watch.rs',
                      'tests/test_unicode_data.py', 'raises.py', '@pkg/parser-core',
                      'src\\api.py', 'crate::module', 'user interface', 'api v2'):
            for breaking in ('', '!'):
                message = f'fix({scope}){breaking}: handle missing path'
                with self.subTest(scope=scope, breaking=breaking):
                    decoded = decode_tokens(tokenize_text(message))
                    self.assertEqual(decoded, message)
                    self.assertEqual(parse_commit_message(decoded).scope, scope)

    def test_scope_policy_does_not_rewrite_subject_body_or_invalid_header(self):
        for message in ('fix(src/api.py): mention a / b and a . b\n\nBody (a / b) `x.y`',
                        'fix(user interface): preserve words\n\nBREAKING CHANGE: a / b',
                        'fix: no scope; a / b', 'not a commit (a / b): text',
                        'fix(bad\nscope): subject'):
            tokens = tokenize_text(message)
            old, new = decode_tokens_legacy(tokens), decode_tokens(tokens)
            if message.startswith(('fix(src/api.py)', 'fix(user interface)')):
                self.assertEqual(new.split(': ', 1)[1], old.split(': ', 1)[1])
            else:
                self.assertEqual(new, old)

    @unittest.skipIf(_load_torch() is None, 'torch is not installed')
    def test_tiny_complete_training_evaluation_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = 'eval-test'
            _prepare_trainer_job(root, run_id=run_id)
            path = root / proof_trainer_job_path(run_id)
            job = json.loads(path.read_text())
            job['data_contract'].update(train_dev_records=2, report_eval_records=1)
            job['eval_contract'] = {
                'report_prediction_path': f'artifacts/train-runs/{run_id}.predictions.jsonl',
                'report_eval_path': f'artifacts/train-runs/{run_id}.eval.json',
            }
            path.write_text(json.dumps(job))
            run_proof_lm_training(root, run_id=run_id, write=True)
            report = evaluate(root, run_id, device='cpu', max_new_tokens=8)
            self.assertEqual(report['records'], 1)
            self.assertEqual(report['status'], 'evaluated')
            again = evaluate(root, run_id, device='cpu', max_new_tokens=8)
            self.assertEqual(again['prediction_sha256'], report['prediction_sha256'])
            for count in report['counts'].values():
                self.assertEqual(sum(count.values()), 1)
            with self.assertRaisesRegex(ValueError, 'different generation settings'):
                evaluate(root, run_id, device='cpu', max_new_tokens=9)


if __name__ == '__main__':
    unittest.main()
