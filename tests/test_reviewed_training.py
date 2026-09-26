from copy import deepcopy
from pathlib import Path
import tempfile
from unittest import TestCase

from gitctx import reviewed_training as rt
from gitctx.proof_lm_train import _build_model, _load_torch


class ToyDataset:
    fingerprint = 'pinned-toy-data'
    def ids(self, partition):
        return ('a', 'b') if partition == 'train' else ('v',)
    def example(self, rid, *, partition):
        assert rid in self.ids(partition)
        return {'group': {'prompts': [[1,8,9]] if rid == 'a' else [[1,7],[1,9,10]]},
                'answer': [12,13,2], 'reference': {'partition': partition},
                'partition': partition, 'dataset_fingerprint': self.fingerprint}


class ReviewedTrainingTests(TestCase):
    def setUp(self):
        self.torch = _load_torch()
        if self.torch is None: self.skipTest('torch unavailable')
        self.contract = dict(tokenizer_vocab_size=32, hidden_size=16, layers=1,
            attention_heads=4, kv_heads=2, intermediate_size=32, context_tokens=64)
        self.ds = ToyDataset()
        self.options = dict(device='cpu', epochs=2, seed=17, run_contract=self.contract,
                            training_authorized=True, checkpoint_every=2)

    def pair(self):
        self.torch.manual_seed(37)
        model = _build_model(self.torch, self.contract, attention_chunk_size=3, activation_checkpointing=True)
        return model, self.torch.optim.AdamW(model.parameters(), lr=0.0003)

    def test_resume_mid_epoch_matches_uninterrupted_parameters_moments_and_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            a, oa = self.pair(); b, ob = self.pair()
            full = rt.run_epochs(self.torch, self.ds, a, oa, checkpoint_dir=Path(root)/'full', **self.options)
            first = rt.run_epochs(self.torch, self.ds, b, ob, checkpoint_dir=Path(root)/'resume', max_steps=1, **self.options)
            self.assertFalse(first['complete']); self.assertEqual(first['cursor'], 1)
            c, oc = self.pair()
            done = rt.run_epochs(self.torch, self.ds, c, oc, checkpoint_dir=Path(root)/'resume', resume=True, **self.options)
            self.assertTrue(done['complete']); self.assertEqual(done['steps'], 4)
            self.assertEqual(full['metrics'], done['metrics'])
            for x, y in zip(a.parameters(), c.parameters()):
                self.torch.testing.assert_close(x, y, rtol=0, atol=0)
            for key, value in oa.state_dict()['state'].items():
                for field, tensor in value.items():
                    self.torch.testing.assert_close(tensor, oc.state_dict()['state'][key][field], rtol=0, atol=0)
            again = rt.run_epochs(self.torch, self.ds, c, oc, checkpoint_dir=Path(root)/'resume', resume=True, **self.options)
            self.assertEqual(again['invocation_steps'], 0)

    def test_changed_contract_or_dataset_and_tampered_checkpoint_fail_before_updates(self):
        with tempfile.TemporaryDirectory() as root:
            model, optimizer = self.pair()
            rt.run_epochs(self.torch, self.ds, model, optimizer, checkpoint_dir=root, max_steps=1, **self.options)
            before = deepcopy(model.state_dict())
            with self.assertRaisesRegex(ValueError, 'identity'):
                rt.run_epochs(self.torch, self.ds, model, optimizer, checkpoint_dir=root, resume=True,
                              **{**self.options,'seed':18})
            altered = ToyDataset(); altered.fingerprint = 'different'
            with self.assertRaisesRegex(ValueError, 'identity'):
                rt.run_epochs(self.torch, altered, model, optimizer, checkpoint_dir=root, resume=True, **self.options)
            checkpoint = next(Path(root).glob('*.pt'))
            with checkpoint.open('ab') as handle: handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                rt.run_epochs(self.torch, self.ds, model, optimizer, checkpoint_dir=root, resume=True, **self.options)
            for key, value in before.items():
                self.torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)

    def test_no_authorization_or_invalid_budget_never_updates_or_writes(self):
        with tempfile.TemporaryDirectory() as root:
            model, optimizer = self.pair(); before = deepcopy(model.state_dict())
            for changes in ({'training_authorized':False},{'epochs':8},{'max_steps':0}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    rt.run_epochs(self.torch, self.ds, model, optimizer, checkpoint_dir=root, **{**self.options,**changes})
            self.assertEqual(list(Path(root).iterdir()), [])
            for key, value in before.items():
                self.torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
