from copy import deepcopy
import math
import unittest

from gitctx import evidence_windows as ew
from gitctx import window_mixture as wm
from gitctx.proof_lm_train import _build_model, _batch_for_torch, _load_torch
from test_student_input import record


torch = _load_torch()


class WindowPreparationTests(unittest.TestCase):
    def test_lossless_intervals_and_label_independence_with_long_lines(self):
        class Tokenizer:
            def encode(self, text):
                return [x + 32 for x in text.encode()]
        tokenizer = Tokenizer()
        r = record(diff='diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-old\n+' + 'a' * 18000 + '\n')
        windows = ew.build_windows(r, tokenizer)
        original = deepcopy(windows)
        group = wm.prepare(r, tokenizer, windows=windows)
        self.assertGreater(len(group['prompts']), 1)
        self.assertTrue(all(len(p) <= 7936 for p in group['prompts']))
        self.assertEqual(group['source_bytes'], len(r['diff'].encode()))
        self.assertEqual(windows, original)
        self.assertTrue(ew.verify_coverage(r, windows))
        changed_label = {**r, 'target_message': 'SECRET LABEL', 'historical_subject': 'SECRET TITLE'}
        self.assertEqual(group, wm.prepare(changed_label, tokenizer, windows=windows))
        bad = deepcopy(windows); bad[0]['target'] = r['target_message']
        with self.assertRaises(ValueError): wm.prepare(r, tokenizer, windows=bad)
        with self.assertRaises(ValueError): wm.prepare(r, tokenizer, windows=windows[:-1])
        bad = deepcopy(windows); bad[0]['prompt_sha256'] = 'changed'
        with self.assertRaises(ValueError): wm.prepare(r, tokenizer, windows=bad)


@unittest.skipIf(torch is None, 'torch is not installed')
class MixtureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.contract = dict(tokenizer_vocab_size=64, hidden_size=16, layers=2,
                             attention_heads=4, kv_heads=2, intermediate_size=32, context_tokens=64)
        self.model = _build_model(torch, self.contract, attention_chunk_size=3, activation_checkpointing=True)
        self.prompts = [[1, 5, 6], [1, 8, 9, 10, 11]]
        self.answer = [14, 15, 2]

    def test_selective_head_matches_full_logits_and_parameter_gradients(self):
        other = _build_model(torch, self.contract)
        other.load_state_dict(self.model.state_dict())
        ids = torch.tensor([[1, 9, 8, 7, 6, 5, 4, 3]])
        positions = [2, 4, 7]
        full = other(ids, torch.ones_like(ids))[:, positions]
        selected = self.model(ids, torch.ones_like(ids), logit_positions=positions)
        torch.testing.assert_close(full, selected)
        full.square().sum().backward(); selected.square().sum().backward()
        for (name, a), (_, b) in zip(other.named_parameters(), self.model.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5, msg=name)
        for bad in ([], [1.5], [-1], [8], [[1, 2]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.model(ids, torch.ones_like(ids), logit_positions=bad)

    def test_streaming_gradient_matches_single_joint_autograd_graph(self):
        other = _build_model(torch, self.contract)
        other.load_state_dict(self.model.state_dict())
        logs = torch.stack([wm.answer_log_probs(torch, other, p, self.answer, device='cpu') for p in self.prompts])
        expected = -wm.joint_log_probs(torch, logs).mean()
        expected.backward()
        result = wm.backward_joint(torch, self.model, self.prompts, self.answer, device='cpu')
        self.assertAlmostEqual(result['loss'], float(expected.detach()), places=6)
        self.assertEqual(result['loss_tokens'], len(self.answer))
        self.assertEqual(result['forward_passes'], 4)
        for (name, a), (_, b) in zip(other.named_parameters(), self.model.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5, msg=name)

    def test_one_window_matches_causal_masked_loss_and_no_future_target_leak(self):
        prompt = self.prompts[0]
        ids = prompt + self.answer
        batch = _batch_for_torch(torch, [{'input_ids': ids, 'loss_mask': [0]*len(prompt)+[1]*len(self.answer)}], device='cpu')
        expected = self.model(batch['input_ids'], batch['attention_mask'], labels=batch['labels'])
        scores = wm.answer_log_probs(torch, self.model, prompt, self.answer, device='cpu')
        torch.testing.assert_close(-scores.mean(), expected)
        altered = wm.answer_log_probs(torch, self.model, prompt, [14, 32, 33], device='cpu')
        torch.testing.assert_close(scores[0], altered[0])
        self.assertEqual(batch['loss_tokens'], len(self.answer))

    def test_all_windows_contribute_as_probabilities_not_mean_log_likelihood(self):
        logs = torch.tensor([[math.log(.9), math.log(.2)], [math.log(.1), math.log(.6)]])
        actual = wm.joint_log_probs(torch, logs)
        torch.testing.assert_close(actual.exp(), torch.tensor([.5, .4]))
        self.assertFalse(torch.allclose(actual, logs.mean(0)))
        torch.testing.assert_close(actual, wm.joint_log_probs(torch, logs.flip(0)))
        changed = logs.clone(); changed[1, 0] = math.log(.7)
        self.assertNotEqual(float(actual[0]), float(wm.joint_log_probs(torch, changed)[0]))

    def test_cached_generation_matches_full_replay_with_shared_prefix(self):
        expected = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(5):
                scores = []
                for prompt in self.prompts:
                    ids = torch.tensor([prompt + expected])
                    logits = self.model(ids, torch.ones_like(ids), last_token_only=True)
                    scores.append(logits[0, -1].log_softmax(-1))
                probs = torch.stack(scores).exp().mean(0)
                expected.append(int(probs.argmax()))
        self.model.train()
        actual, reason = wm.generate(torch, self.model, self.prompts, device='cpu', max_new_tokens=5, stop_ids=set())
        self.assertEqual(actual, expected)
        self.assertEqual(reason, 'token_limit')
        self.assertTrue(self.model.training)
        reversed_output, _ = wm.generate(torch, self.model, self.prompts[::-1], device='cpu', max_new_tokens=5, stop_ids=set())
        self.assertEqual(reversed_output, expected)
        for budget in (0, 1, 1000):
            uncached, _ = wm.generate(torch, self.model, self.prompts, device='cpu',
                                     max_new_tokens=5, stop_ids=set(), max_cache_bytes=budget)
            self.assertEqual(uncached, expected)
        stopped, reason = wm.generate(torch, self.model, self.prompts, device='cpu', max_new_tokens=5, stop_ids={expected[0]})
        self.assertEqual((stopped, reason), ([], 'stop_token'))

    def test_reserves_fail_before_model_call(self):
        from unittest.mock import Mock
        model = Mock()
        for prompts, answer in (([], [1]), ([[1]*7937], [2]), ([[1]], [2]*257), ([[True]], [2])):
            with self.subTest(prompts=len(prompts)), self.assertRaises(ValueError):
                wm.backward_joint(torch, model, prompts, answer, device='cpu')
        model.assert_not_called()
