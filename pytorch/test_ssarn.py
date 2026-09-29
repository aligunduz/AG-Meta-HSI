"""Small end-to-end checks for the PyTorch baseline."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from .data import load_pixel_split, make_pixel_split, save_pixel_split
from .ssarn import SSARN
from .train import run_baseline
from .wandb_log import load_results


class SSARNTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
