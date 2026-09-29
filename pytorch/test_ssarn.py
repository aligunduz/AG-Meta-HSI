"""Small end-to-end checks for the PyTorch baseline."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from .benchmark import run_benchmark
from .data import load_pixel_split, make_pixel_split, save_pixel_split
from .methods import available_methods, get_method
from .ssarn import SSARN
from .train import run_baseline
from .wandb_benchmark import load_benchmark
from .wandb_log import load_results


class SSARNTests(unittest.TestCase):
    def test_method_registry(self):
        self.assertIn("SSARN", available_methods())
        self.assertIs(get_method("ssarn").runner(), run_baseline)
        with self.assertRaisesRegex(ValueError, "not implemented"):
            get_method("UNREGISTERED")

    def test_output_shapes_for_supported_band_counts(self):
        for bands, classes in ((103, 9), (204, 16), (200, 16)):
            with self.subTest(bands=bands):
                model = SSARN(bands, classes).eval()
                with torch.inference_mode():
                    logits = model(torch.ones(2, 1, bands, 9, 9))
                self.assertEqual(tuple(logits.shape), (2, classes))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_one_epoch_writes_consistent_metrics_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            labels = np.tile(np.arange(1, 10, dtype=np.uint8)[:, None], (1, 9))
            cube = np.random.default_rng(5).normal(size=(9, 9, 103)).astype(np.float32)
            savemat(root / "PaviaU.mat", {"paviaU": cube})
            savemat(root / "PaviaU_gt.mat", {"paviaU_gt": labels})
            split_path, output_dir = root / "split.tsv", root / "run"
            save_pixel_split(split_path, make_pixel_split(labels, seed=93, k=5))
            result = run_baseline(dataset="UP", data_dir=root, split_path=split_path,
                                  output_dir=output_dir, seed=93, k=5, epochs=1,
                                  batch_size=15, test_batch_size=16,
                                  device="cpu", mode="train")
            self.assertEqual(result["epoch"], 1)
            config = json.loads((output_dir / "config.json").read_text())
            metrics = json.loads((output_dir / "metrics.json").read_text())
            self.assertEqual(config["train_count"], 45)
            self.assertEqual(config["test_count"], 36)
            self.assertEqual(result, metrics)
            self.assertEqual(len((output_dir / "training.tsv").read_text().splitlines()), 2)
            self.assertEqual(len(load_pixel_split(output_dir / "split.tsv").test), 36)
            self.assertEqual(load_results(output_dir)[0], metrics)
            checkpoint = torch.load(output_dir / "checkpoint.pt",
                                    map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            model = SSARN(103, 9)
            model.load_state_dict(checkpoint["model_state_dict"])
            self.assertEqual(len((output_dir / "confusion.tsv").read_text().splitlines()), 10)

    def test_two_seed_benchmark_has_independent_splits_and_resumes(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            labels = np.tile(np.arange(1, 10, dtype=np.uint8)[:, None], (1, 9))
            cube = np.random.default_rng(6).normal(size=(9, 9, 103)).astype(np.float32)
            savemat(root / "PaviaU.mat", {"paviaU": cube})
            savemat(root / "PaviaU_gt.mat", {"paviaU_gt": labels})
            output_dir = root / "benchmark"
            split_dir = root / "fixed_splits"
            for seed in (101, 102):
                save_pixel_split(split_dir / f"up_seed{seed}_k5.tsv",
                                 make_pixel_split(labels, seed=seed, k=5))
            kwargs = dict(dataset="UP", data_dir=root, output_dir=output_dir,
                          split_dir=split_dir,
                          seed_start=101, runs=2, k=5, epochs=1,
                          batch_size=15, test_batch_size=16, device="cpu")
            summary = run_benchmark(**kwargs)
            self.assertEqual(summary["n"], 2)
            self.assertEqual(summary["seeds"], [101, 102])
            loaded, settings, rows = load_benchmark(output_dir)
            self.assertEqual(loaded, summary)
            self.assertEqual(settings["seeds"], [101, 102])
            self.assertEqual(settings["split_source"], "fixed TSV files")
            self.assertNotEqual(rows[0]["split_sha256"], rows[1]["split_sha256"])
            split_a = load_pixel_split(output_dir / "splits" / "seed_101.tsv")
            split_b = load_pixel_split(output_dir / "splits" / "seed_102.tsv")
            self.assertFalse(np.array_equal(split_a.train, split_b.train))
            for name in ("OA", "AA", "kappa"):
                expected_mean = (float(rows[0][name]) + float(rows[1][name])) / 2
                self.assertTrue(math.isclose(
                    summary["metrics"][name]["mean"], expected_mean, abs_tol=1e-12))
            self.assertEqual(run_benchmark(**kwargs), summary)
            self.assertEqual(len(list((output_dir / "runs" / "seed_101").glob("attempt_*"))), 1)
            broken = output_dir / "runs" / "seed_102" / "attempt_001" / "confusion.tsv"
            broken.write_text("interrupted\n", encoding="utf-8")
            self.assertEqual(run_benchmark(**kwargs), summary)
            self.assertEqual(len(list((output_dir / "runs" / "seed_102").glob("attempt_*"))), 2)

    def test_benchmark_rejects_missing_fixed_split_before_creating_output(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            labels = np.tile(np.arange(1, 10, dtype=np.uint8)[:, None], (1, 9))
            savemat(root / "PaviaU.mat", {"paviaU": np.zeros((9, 9, 103), dtype=np.float32)})
            savemat(root / "PaviaU_gt.mat", {"paviaU_gt": labels})
            output_dir = root / "benchmark"
            with self.assertRaises(FileNotFoundError):
                run_benchmark(dataset="UP", data_dir=root, output_dir=output_dir,
                              split_dir=root / "empty_splits", seed_start=101,
                              runs=2, k=5, epochs=1, device="cpu")
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
