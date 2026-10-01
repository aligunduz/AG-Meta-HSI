"""Behavioral and output-contract checks for first-order Proto-MAML."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
from scipy.io import savemat

from .benchmark import run_benchmark
from .data import make_pixel_split, save_pixel_split
from .foprotomaml import (AdaptedClassifier, PrototypeHead, adapt_task,
                          backbone_parameters, make_task_pool,
                          meta_task_step, resolve_config, run_baseline,
                          prototype_initialization, set_support_stats, ssarn_embedding)
from .methods import get_method
from .qmtn import resolve_meta_config, sample_task
from .ssarn import SSARN
from .wandb_log import load_results


def synthetic_scene(root: Path) -> tuple[Path, np.ndarray]:
    labels = np.tile(np.arange(1, 10, dtype=np.uint8)[:, None], (1, 9))
    cube = np.random.default_rng(37).normal(size=(9, 9, 103)).astype(np.float32)
    savemat(root / "PaviaU.mat", {"paviaU": cube})
    savemat(root / "PaviaU_gt.mat", {"paviaU_gt": labels})
    split = root / "split.tsv"
    save_pixel_split(split, make_pixel_split(labels, seed=93, k=5))
    return split, labels


class FOProtoMAMLTests(unittest.TestCase):
    def test_embedding_matches_ssarn_forward_exactly(self):
        torch.manual_seed(8)
        net = SSARN(103, 9).eval()
        x = torch.randn(3, 1, 103, 9, 9)
        with torch.inference_mode():
            self.assertTrue(torch.equal(net.classifier(ssarn_embedding(net, x)), net(x)))

    def test_prototype_head_is_nearest_euclidean_centroid(self):
        prototypes = torch.tensor([[0.0, 2.0], [3.0, 0.0], [-1.0, -1.0]])
        embeddings = torch.tensor([[0.1, 1.8], [2.8, 0.2], [-0.8, -0.9]])
        head = PrototypeHead(prototypes)
        expected = torch.cdist(embeddings, prototypes).argmin(dim=1)
        self.assertTrue(torch.equal(head(embeddings).argmax(dim=1), expected))
        self.assertTrue(head.weight.is_leaf and head.bias.is_leaf)
        self.assertTrue(torch.equal(head.weight, 2 * prototypes))

    def test_task_pool_matches_qmtn_sequence(self):
        records = np.array([(row, class_id, class_id)
                            for class_id in range(1, 10)
                            for row in range(1, 6)], dtype=np.int64)
        meta = resolve_config("UP", 5, {})
        actual = make_task_pool(records, classes=9, meta=meta,
                                rng=np.random.default_rng(93))
        qmtn = resolve_meta_config("UP", 5, {})
        rng = np.random.default_rng(93)
        expected = [sample_task(records, classes=9, ways=qmtn["ways"],
                                support_shots=qmtn["support_shots"], rng=rng)
                    for _ in range(qmtn["tasks_per_epoch"])]
        for (support_a, query_a), (support_b, query_b) in zip(actual, expected):
            np.testing.assert_array_equal(support_a, support_b)
            np.testing.assert_array_equal(query_a, query_b)

    def test_meta_gradient_includes_nonzero_prototype_initialization_gradient(self):
        torch.manual_seed(31)
        initial = SSARN(103, 9)
        reference = copy.deepcopy(initial)
        actual = copy.deepcopy(initial)
        sx = torch.randn(8, 1, 103, 9, 9)
        sy = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        qx = torch.randn(4, 1, 103, 9, 9)
        qy = torch.tensor([0, 1, 0, 1])

        set_support_stats(reference, sx)
        weight0, bias0 = prototype_initialization(reference, sx, sy, ways=2)
        self.assertIsNotNone(weight0.grad_fn)
        self.assertIsNotNone(bias0.grad_fn)
        adapted, head, _, _ = adapt_task(
            reference, sx, sy, ways=2, steps=2, lr=0.01,
            initialization=(weight0, bias0))
        self.assertTrue(head.weight.is_leaf and head.bias.is_leaf)
        adapted.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        F.cross_entropy(head(ssarn_embedding(adapted, qx)), qy).backward()
        proto_grads = torch.autograd.grad(
            [weight0, bias0], backbone_parameters(reference),
            grad_outputs=[head.weight.grad, head.bias.grad], allow_unused=True)
        detached_grads = [p.grad.detach().clone() for p in backbone_parameters(adapted)]
        expected = [query_grad + (proto_grad if proto_grad is not None else 0)
                    for query_grad, proto_grad in zip(detached_grads, proto_grads)]
        self.assertGreater(sum(g.abs().sum().item() for g in proto_grads if g is not None), 0)
        self.assertTrue(any(not torch.allclose(total, detached)
                            for total, detached in zip(expected, detached_grads)))

        optimizer = torch.optim.SGD(backbone_parameters(actual), lr=0.002)
        meta_task_step(actual, optimizer, sx, sy, qx, qy,
                       ways=2, inner_steps=2, inner_lr=0.01)
        for parameter, gradient, original in zip(
                backbone_parameters(actual), expected, backbone_parameters(initial)):
            torch.testing.assert_close(parameter.grad, gradient)
            torch.testing.assert_close(parameter, original - 0.002 * gradient)
            self.assertIsNone(parameter.grad.grad_fn)
        # Meta buffers remain its own support estimate, never the adapted estimate.
        for actual_buffer, reference_buffer in zip(actual.buffers(), reference.buffers()):
            torch.testing.assert_close(actual_buffer, reference_buffer)

    def test_support_stats_eval_matches_same_batch_train_output(self):
        torch.manual_seed(41)
        net = SSARN(103, 9)
        sx = torch.randn(20, 1, 103, 9, 9)
        set_support_stats(net, sx)
        self.assertFalse(net.training)
        for module in net.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                self.assertFalse(module.training)
                self.assertEqual(module.momentum, 0.1)
                self.assertEqual(module.num_batches_tracked.item(), 1)
                self.assertFalse(module.running_mean.requires_grad)
                self.assertFalse(module.running_var.requires_grad)
        self.assertTrue(all(p.grad is None for p in net.parameters()))
        with torch.no_grad():
            eval_output = ssarn_embedding(net, sx)
            net.train()
            train_output = ssarn_embedding(net, sx)
        # Running variance is unbiased; train-mode normalization uses biased variance.
        torch.testing.assert_close(eval_output, train_output, rtol=0.02, atol=0.002)

    def test_adaptation_refreshes_support_stats_and_query_cannot_change_them(self):
        torch.manual_seed(43)
        net = SSARN(103, 9)
        sx = torch.randn(8, 1, 103, 9, 9)
        sy = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        with patch("pytorch.foprotomaml.set_support_stats", wraps=set_support_stats) as stats:
            adapted, head, _, _ = adapt_task(net, sx, sy, ways=2, steps=3, lr=0.01)
        self.assertEqual(stats.call_count, 4)  # Before each step and after the last.
        for call in stats.call_args_list:
            self.assertIs(call.args[0], adapted)
            self.assertIs(call.args[1], sx)
        self.assertFalse(adapted.training)
        buffers = [buffer.clone() for buffer in adapted.buffers()]
        classifier = AdaptedClassifier(adapted, head).eval()
        query = 10 + torch.randn(4, 1, 103, 9, 9)
        with torch.no_grad():
            full = classifier(query)
            individual = torch.cat([classifier(x.unsqueeze(0)) for x in query])
        torch.testing.assert_close(full, individual, rtol=1e-4, atol=1e-4)
        for before, after in zip(buffers, adapted.buffers()):
            self.assertTrue(torch.equal(before, after))

    def test_one_task_updates_meta_but_keeps_adapted_copy_separate(self):
        torch.manual_seed(17)
        model = SSARN(103, 9)
        before = [p.detach().clone() for p in backbone_parameters(model)]
        classifier_before = [p.detach().clone() for p in model.classifier.parameters()]
        optimizer = torch.optim.Adam(backbone_parameters(model), lr=0.002)
        sx = torch.randn(5, 1, 103, 9, 9)
        qx = torch.randn(2, 1, 103, 9, 9)
        support_loss, query_loss, adapted = meta_task_step(
            model, optimizer, sx, torch.tensor([0, 0, 1, 1, 1]),
            qx, torch.tensor([0, 1]), ways=2, inner_steps=1, inner_lr=0.01)
        self.assertTrue(np.isfinite([support_loss, query_loss]).all())
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, backbone_parameters(model))))
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(backbone_parameters(model), backbone_parameters(adapted))))
        self.assertTrue(all(torch.equal(a, b) for a, b in
                            zip(classifier_before, model.classifier.parameters())))

    def test_invalid_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "epoch_transfer"):
            resolve_config("UP", 5, {"epoch_transfer": "none"})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_config("UP", 5, {"other": 3})
        with self.assertRaisesRegex(ValueError, "test_adapt_steps"):
            resolve_config("UP", 5, {"test_adapt_steps": 0})

    def test_two_epoch_run_writes_compatible_artifacts(self):
        self.assertIs(get_method("foprotomaml").runner(), run_baseline)
        self.assertEqual(get_method("foprotomaml").default_learning_rate, 0.002)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split, _ = synthetic_scene(root)
            output = root / "run"
            result = run_baseline(dataset="UP", data_dir=root, split_path=split,
                                  output_dir=output, seed=93, k=5, epochs=2,
                                  batch_size=15, test_batch_size=16, device="cpu",
                                  mode="train", method_config={"tasks_per_epoch": 2,
                                                                 "inner_steps": 1,
                                                                 "test_adapt_steps": 1})
            self.assertEqual(result["epoch"], 2)
            self.assertEqual(load_results(output)[0], result)
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["model"], "FOProtoMAML-SSARN-v2")
            self.assertEqual(config["prototype_gradient"],
                             "flows through W0,b0 (Meta-Dataset fo-Proto-MAML)")
            self.assertIn("moments are constants for gradients", config["bn_protocol"])
            self.assertIn("test moments use only split.train", config["bn_protocol"])
            self.assertEqual(config["effective_meta_config"]["support_shots"], 4)
            self.assertEqual(len(config["source_sha256"]), 5)
            checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu",
                                    weights_only=True)
            self.assertIn("meta_model_state_dict", checkpoint)
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("test_head_state_dict", checkpoint)
            self.assertIn("optimizer_state_dict", checkpoint)

    def test_benchmark_resumes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, labels = synthetic_scene(root)
            split_dir = root / "splits"
            split_dir.mkdir()
            for seed in (101, 102):
                save_pixel_split(split_dir / f"up_seed{seed}_k5.tsv",
                                 make_pixel_split(labels, seed=seed, k=5))
            kwargs = dict(baseline="FOPROTOMAML", dataset="UP", data_dir=root,
                          split_dir=split_dir, output_dir=root / "benchmark",
                          seed_start=101, runs=2, k=5, epochs=1, device="cpu",
                          method_config={"tasks_per_epoch": 1, "inner_steps": 1,
                                         "test_adapt_steps": 1})
            first = run_benchmark(**kwargs)
            self.assertEqual(first["baseline"], "FOPROTOMAML")
            self.assertEqual(run_benchmark(**kwargs), first)
            self.assertEqual(len(list((root / "benchmark" / "runs" / "seed_101").glob("attempt_*"))), 1)


if __name__ == "__main__":
    unittest.main()
