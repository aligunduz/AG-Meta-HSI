"""Train and evaluate the supervised SSARN baseline with PyTorch.

Run from the repository root: python -m pytorch.train --train --dataset UP --data data
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import (dataset_spec, file_sha256, load_pixel_split, load_scene,
                   padded_cube, patch_batch, validate_pixel_split)
from .ssarn import SSARN


ROOT = Path(__file__).resolve().parents[1]


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested != "gpu":
        raise ValueError("device must be gpu or cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; select a GPU runtime or pass --device cpu")
    return torch.device("cuda")


def classification_metrics(confusion: np.ndarray) -> dict:
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("Confusion matrix must be square")
    if np.any(confusion < 0):
        raise ValueError("Confusion counts must be nonnegative")
    support = confusion.sum(axis=1)
    if np.any(support == 0):
        raise ValueError("Every class needs test support")
    correct = np.diag(confusion)
    total = int(confusion.sum())
    oa = float(correct.sum() / total)
    per_class = correct / support
    expected = float(np.dot(support / total, confusion.sum(axis=0) / total))
    kappa = float((oa - expected) / (1 - expected)) if expected != 1 else math.nan
    return {"OA": oa, "AA": float(per_class.mean()), "kappa": kappa,
            "per_class": per_class.tolist(), "support": support.tolist(),
            "correct": correct.tolist()}


def evaluate(model: SSARN, padded: np.ndarray, records: np.ndarray,
             *, batch_size: int, device: torch.device, classes: int) -> np.ndarray:
    model.eval()
    confusion = np.zeros((classes, classes), dtype=np.int64)
    with torch.inference_mode():
        for first in range(0, len(records), batch_size):
            current = records[first:first + batch_size]
            x = torch.from_numpy(patch_batch(padded, current)).to(device)
            logits = model(x)
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("Nonfinite test logits")
            predicted = logits.argmax(dim=1).cpu().numpy()
            truth = current[:, 2] - 1
            np.add.at(confusion, (truth, predicted), 1)
    return confusion


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_baseline(*, baseline: str = "SSARN", dataset: str = "UP", data_dir: str | Path = "data",
                 split_path: str | Path | None = None, output_dir: str | Path | None = None,
                 seed: int = 93, k: int = 5, epochs: int = 300, learning_rate: float = 0.001,
                 batch_size: int = 15, test_batch_size: int = 32, device: str = "gpu",
                 mode: str = "check", method_config: dict | None = None) -> dict:
    baseline, dataset = baseline.upper(), dataset.upper()
    if baseline != "SSARN":
        raise ValueError("Only SSARN baseline is implemented")
    method_config = {} if method_config is None else method_config
    if method_config:
        raise ValueError("SSARN does not accept method-specific settings")
    _, _, _, _, bands, classes = dataset_spec(dataset)
    if mode not in ("check", "smoke", "train"):
        raise ValueError("mode must be check, smoke, or train")
    if k <= 0 or epochs <= 0 or batch_size < 2 or test_batch_size <= 0:
        raise ValueError("Invalid k, epochs, or batch size")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if mode == "smoke" and batch_size != 15:
        raise ValueError("Smoke mode requires batch_size=15")
    target = resolve_device(device)
    scene = load_scene(dataset, data_dir)
    tracked_split = ROOT / "splits" / f"{dataset.lower()}_seed{seed}_k{k}.tsv"
    if split_path is not None:
        split_file, split_source = Path(split_path), "provided TSV"
    else:
        split_file, split_source = tracked_split, "project TSV"
    if not split_file.is_file():
        raise FileNotFoundError(f"Fixed split file does not exist: {split_file}")
    split = load_pixel_split(split_file)
    if split.seed != seed or split.k != k:
        raise ValueError("Split metadata does not match seed/k")
    validate_pixel_split(scene.labels, split)
    split_digest = file_sha256(split_file)

    torch.manual_seed(seed)
    if target.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model = SSARN(bands, classes).to(target)
    padded = padded_cube(scene)
    model.eval()
    with torch.inference_mode():
        xcheck = torch.from_numpy(patch_batch(padded, split.train[:1])).to(target)
        logits = model(xcheck)
        if logits.shape != (1, classes) or not torch.isfinite(logits).all().item():
            raise RuntimeError("Invalid SSARN forward check")
    print(f"Forward check: {tuple(xcheck.shape)} -> {tuple(logits.shape)}; device={target}", flush=True)
    print(f"Fixed split: {len(split.train)} train, {len(split.test)} test; SHA256={split_digest}", flush=True)
    if mode == "check":
        return {"split_sha256": split_digest, "device": str(target)}

    xtrain = torch.from_numpy(patch_batch(padded, split.train))
    ytrain = torch.from_numpy(split.train[:, 2].copy() - 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if mode == "smoke":
        indices = np.random.default_rng(seed).permutation(len(split.train))[:15]
        x, y = xtrain[indices].to(target), ytrain[indices].to(target)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_before = F.cross_entropy(model(x), y)
        loss_before.backward()
        if not torch.isfinite(loss_before).item() or any(
                not torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters() if parameter.grad is not None):
            raise RuntimeError("Smoke loss or gradients are nonfinite")
        optimizer.step()
        changed = sum(torch.count_nonzero(previous != current).item()
                      for previous, current in zip(before, model.parameters()))
        if changed == 0:
            raise RuntimeError("Smoke optimizer step changed no parameters")
        with torch.no_grad():
            loss_after = F.cross_entropy(model(x), y)
        print(f"Smoke: one Adam update; loss {loss_before.item():.6f} -> {loss_after.item():.6f}; changed={changed}")
        return {"loss_before": loss_before.item(), "loss_after": loss_after.item(),
                "changed_parameters": changed}

    destination = Path(output_dir) if output_dir is not None else ROOT / "outputs" / f"ssarn_{dataset.lower()}_seed{seed}"
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    destination.mkdir(parents=True)
    config = {
        "model": "SSARN-Fig2-assumptions-v1", "baseline": baseline,
        "method_config": method_config,
        "framework": "PyTorch", "dataset": dataset, "seed": seed, "k": k,
        "split_seed": split.seed, "split_sha256": split_digest,
        "split_source": split_source,
        "epochs": epochs, "learning_rate": learning_rate, "optimizer": "Adam",
        "batch_size": batch_size, "test_batch_size": test_batch_size,
        "train_count": len(split.train), "test_count": len(split.test),
        "patch_size": [9, 9, bands], "classes": list(range(1, classes + 1)),
        "preprocessing": "raw Float32; zero padding; no augmentation",
        "selection": "fixed final epoch; test once after checkpoint",
        "python_version": __import__("sys").version.split()[0],
        "torch_version": str(torch.__version__),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "device": "CUDA" if target.type == "cuda" else "CPU",
        "checkpoint": "checkpoint.pt",
    }
    if target.type == "cuda":
        config["gpu_name"] = torch.cuda.get_device_name(target)
    cube_file, gt_file, *_ = dataset_spec(dataset)
    config["data_sha256"] = {name: file_sha256(Path(data_dir) / name)
                             for name in (cube_file, gt_file)}
    config["source_sha256"] = {name: file_sha256(ROOT / "pytorch" / name)
                               for name in ("data.py", "ssarn.py", "train.py")}
    write_json(destination / "config.json", config)
    shutil.copyfile(split_file, destination / "split.tsv")

    rng = np.random.default_rng(seed)
    model.train()
    with (destination / "training.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["epoch", "train_cross_entropy"])
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(split.train))
            total = 0.0
            for first in range(0, len(order), batch_size):
                indices = order[first:first + batch_size]
                x, y = xtrain[indices].to(target), ytrain[indices].to(target)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), y)
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Nonfinite training loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                total += loss.item() * len(indices)
            epoch_loss = total / len(split.train)
            writer.writerow([epoch, epoch_loss])
            handle.flush()
            print(f"Epoch {epoch}/{epochs}; train CE={epoch_loss:.6f}", flush=True)

    if file_sha256(split_file) != split_digest:
        raise RuntimeError("Split changed during training")
    # The fixed final checkpoint is saved before test labels are evaluated.
    torch.save({"model_state_dict": cpu_tree(model.state_dict()),
                "optimizer_state_dict": cpu_tree(optimizer.state_dict()),
                "epoch": epochs, "config": config}, destination / "checkpoint.pt")
    confusion = evaluate(model, padded, split.test, batch_size=test_batch_size,
                         device=target, classes=classes)
    results = classification_metrics(confusion)
    metrics = {key: results[key] for key in ("OA", "AA", "kappa")}
    metrics.update({"accuracy_units": "fraction", "epoch": epochs, "seed": seed,
                    "learning_rate": learning_rate, "checkpoint": "checkpoint.pt",
                    "split_sha256": split_digest,
                    "finished_at": datetime.now(timezone.utc).isoformat()})
    write_json(destination / "metrics.json", metrics)
    with (destination / "class_accuracy.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["class", "support", "correct", "accuracy"])
        for class_id in range(1, classes + 1):
            writer.writerow([class_id, results["support"][class_id - 1],
                             results["correct"][class_id - 1],
                             results["per_class"][class_id - 1]])
    with (destination / "confusion.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["truth/prediction", *range(1, classes + 1)])
        for class_id, row in enumerate(confusion, start=1):
            writer.writerow([class_id, *row.tolist()])
    print(f"Saved final checkpoint and test metrics to {destination}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--check", action="store_true")
    parser.add_argument("--baseline", default="SSARN")
    parser.add_argument("--dataset", default="UP")
    parser.add_argument("--data", default="data")
    parser.add_argument("--split")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    run_baseline(baseline=args.baseline, dataset=args.dataset, data_dir=args.data,
                 split_path=args.split, output_dir=args.output, seed=args.seed, k=args.k,
                 epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size,
                 test_batch_size=args.test_batch_size, device=args.device,
                 mode="train" if args.train else "smoke" if args.smoke else "check")


if __name__ == "__main__":
    main()
