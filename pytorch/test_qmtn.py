"""Checks for QMTN's task partition, twin update, and output contract."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from .benchmark import run_benchmark
from .data import make_pixel_split, save_pixel_split
from .methods import get_method
from .qmtn import query_step, resolve_meta_config, run_baseline, sample_task
from .ssarn import SSARN
from .wandb_log import load_results


class QMTNTests(unittest.TestCase):
    def test_tasks_are_disjoint_and_draw_only_from_training_pixels(self):
        records = np.array([(row, class_id, class_id)
                            for class_id in range(1, 10)
                            for row in range(1, 6)], dtype=np.int64)
        support, query = sample_task(records, classes=9, ways=5,
                                     support_shots=3, rng=np.random.default_rng(7))
        self.assertEqual(len(support), 15)
        self.assertEqual(len(query), 10)
        self.assertEqual(len(set(map(tuple, np.concatenate((support, query))))), 25)
        self.assertEqual(len(set(support[:, 2])), 5)
        with self.assertRaisesRegex(ValueError, "support_shots"):
            resolve_meta_config("UP", 5, {"support_shots": 5})
        self.assertEqual(resolve_meta_config("UP", 5, {})["support_shots"], 4)
        self.assertEqual(resolve_meta_config("UP", 5, {})["epoch_transfer"], "none")
        with self.assertRaisesRegex(ValueError, "epoch_transfer"):
            resolve_meta_config("UP", 5, {"epoch_transfer": "invalid"})

    def test_query_gradient_updates_twin_only(self):
        torch.manual_seed(11)
        model = torch.nn.Linear(2, 2)
        twin = torch.nn.Linear(2, 2)
        twin.load_state_dict(model.state_dict())
        before = [p.detach().clone() for p in model.parameters()]
        optimizer = torch.optim.Adam(twin.parameters(), lr=0.002)
        query_step(model, twin, optimizer,
                   torch.tensor([[1.0, 2.0], [2.0, -1.0]]),
                   torch.tensor([0, 1]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, twin.parameters())))

    def test_two_epochs_write_standard_artifacts(self):
        self.assertIs(get_method("qmtn").runner(), run_baseline)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            labels = np.tile(np.arange(1, 10, dtype=np.uint8)[:, None], (1, 9))
            cube = np.random.default_rng(5).normal(size=(9, 9, 103)).astype(np.float32)
            savemat(root / "PaviaU.mat", {"paviaU": cube})
            savemat(root / "PaviaU_gt.mat", {"paviaU_gt": labels})
            split = root / "split.tsv"
            save_pixel_split(split, make_pixel_split(labels, seed=93, k=5))
            output = root / "run"
            result = run_baseline(dataset="UP", data_dir=root, split_path=split,
                                  output_dir=output, seed=93, k=5, epochs=2,
                                  batch_size=15, test_batch_size=16, device="cpu",
                                  mode="train", method_config={"tasks_per_epoch": 2,
                                                                 "inner_steps": 1})
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["effective_meta_config"]["ways"], 5)
            self.assertEqual(config["effective_meta_config"]["support_shots"], 4)
            self.assertEqual(config["epoch_transfer"], "none")
            self.assertEqual(config["learning_rate"], 0.002)
            self.assertEqual(result["epoch"], 2)
            self.assertEqual(load_results(output)[0], result)
            checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu",
                                    weights_only=True)
            self.assertIn("twin_state_dict", checkpoint)
            SSARN(103, 9).load_state_dict(checkpoint["model_state_dict"])
            self.assertTrue(any(not torch.equal(checkpoint["model_state_dict"][name],
                                                checkpoint["twin_state_dict"][name])
                                for name in checkpoint["model_state_dict"]))
            transferred = root / "transferred"
            run_baseline(dataset="UP", data_dir=root, split_path=split,
                         output_dir=transferred, seed=93, k=5, epochs=2,
                         batch_size=15, test_batch_size=16, device="cpu",
                         mode="train", method_config={"tasks_per_epoch": 2,
                                                        "inner_steps": 1,
                                                        "epoch_transfer": "twin_to_model"})
            transferred_checkpoint = torch.load(transferred / "checkpoint.pt",
                                                map_location="cpu", weights_only=True)
            self.assertTrue(any(not torch.equal(checkpoint["model_state_dict"][name],
                                                transferred_checkpoint["model_state_dict"][name])
                                for name in checkpoint["model_state_dict"]))

    def test_benchmark_resumes_qmtn_runs(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            labels = np.tile(np.arange(1, 10, dtype=np.uint8)[:, None], (1, 9))
            cube = np.random.default_rng(8).normal(size=(9, 9, 103)).astype(np.float32)
            savemat(root / "PaviaU.mat", {"paviaU": cube})
            savemat(root / "PaviaU_gt.mat", {"paviaU_gt": labels})
            split_dir = root / "splits"
            split_dir.mkdir()
            for seed in (101, 102):
                save_pixel_split(split_dir / f"up_seed{seed}_k5.tsv",
                                 make_pixel_split(labels, seed=seed, k=5))
            kwargs = dict(baseline="QMTN", dataset="UP", data_dir=root,
                          split_dir=split_dir, output_dir=root / "benchmark",
                          seed_start=101, runs=2, k=5, epochs=1, device="cpu",
                          method_config={"tasks_per_epoch": 1, "inner_steps": 1})
            first = run_benchmark(**kwargs)
            self.assertEqual(first["baseline"], "QMTN")
            self.assertEqual(run_benchmark(**kwargs), first)
            self.assertEqual(len(list((root / "benchmark" / "runs" / "seed_101").glob("attempt_*"))), 1)


if __name__ == "__main__":
    unittest.main()
