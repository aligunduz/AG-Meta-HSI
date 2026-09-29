"""Behavioral and output-contract checks for first-order Proto-MAML."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from .benchmark import run_benchmark
from .data import make_pixel_split, save_pixel_split
from .foprotomaml import (PrototypeHead, backbone_parameters, make_task_pool,
                          meta_task_step, resolve_config, run_baseline,
                          ssarn_embedding)
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
            self.assertEqual(config["model"], "FOProtoMAML-SSARN-v1")
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
