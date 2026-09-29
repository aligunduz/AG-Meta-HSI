"""Run SSARN on independent seeded splits and summarize OA/AA/kappa."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch

from .data import (dataset_spec, file_sha256, load_pixel_split, load_scene,
                   make_pixel_split, save_pixel_split, validate_pixel_split)
from .train import ROOT, run_baseline, write_json
from .wandb_log import load_results


METRICS = ("OA", "AA", "kappa")


def _validated_run(path: Path, *, seed: int, split_hash: str, settings: dict) -> dict:
    metrics, config, _, _, _ = load_results(path)
    for key in ("baseline", "dataset", "k", "epochs", "learning_rate",
                "batch_size", "test_batch_size", "device", "data_sha256"):
        if config[key] != settings[key]:
            raise ValueError(f"Completed run {path} has different {key}")
    if config["seed"] != seed or config["split_sha256"] != split_hash:
        raise ValueError(f"Completed run {path} has different seed or split")
    if config["source_sha256"] != {
            name: settings["source_sha256"][name]
            for name in ("data.py", "ssarn.py", "train.py")}:
        raise ValueError(f"Completed run {path} used different source code")
    for key in METRICS:
        if not math.isfinite(metrics[key]):
            raise ValueError(f"Nonfinite {key} in {path}")
    return metrics


def _write_runs(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("seed", "split_sha256", "output_dir", *METRICS),
            delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(*, baseline: str = "SSARN", dataset: str = "UP",
                  data_dir: str | Path = "data", output_dir: str | Path | None = None,
                  seed_start: int = 101, runs: int = 10, k: int = 5,
                  epochs: int = 300, learning_rate: float = 0.001,
                  batch_size: int = 15, test_batch_size: int = 32,
                  device: str = "gpu") -> dict:
    baseline, dataset = baseline.upper(), dataset.upper()
    if baseline != "SSARN":
        raise ValueError("Only SSARN baseline is implemented")
    dataset_spec(dataset)
    if runs < 2 or seed_start < 0 or k < 1:
        raise ValueError("Use at least two runs, a nonnegative seed, and k >= 1")
    if epochs < 1 or batch_size < 2 or test_batch_size < 1:
        raise ValueError("Invalid training sizes")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid learning rate")
    if device not in ("gpu", "cpu"):
        raise ValueError("Device must be gpu or cpu")
    if device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")

    seeds = list(range(seed_start, seed_start + runs))
    data_dir = Path(data_dir).resolve()
    root = (Path(output_dir) if output_dir is not None else
            ROOT / "outputs" / f"ssarn_{dataset.lower()}_k{k}_seeds{seeds[0]}-{seeds[-1]}")
    root = root.resolve()
    cube_file, gt_file, *_ = dataset_spec(dataset)
    settings = {
        "schema": 1, "baseline": baseline, "dataset": dataset,
        "seeds": seeds, "k": k, "epochs": epochs, "learning_rate": learning_rate,
        "batch_size": batch_size, "test_batch_size": test_batch_size,
        "device": "CUDA" if device == "gpu" else "CPU",
        "split_generator": "NumPy PCG64 default_rng, independent per seed",
        "data_sha256": {
            name: file_sha256(data_dir / name) for name in (cube_file, gt_file)},
        "source_sha256": {
            name: file_sha256(ROOT / "pytorch" / name)
            for name in ("data.py", "ssarn.py", "train.py", "benchmark.py")},
    }
    manifest_path = root / "experiment.json"
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved != settings:
            raise ValueError(f"Existing experiment has different settings: {root}")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Nonempty experiment directory without manifest: {root}")
        root.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, settings)

    scene = load_scene(dataset, data_dir)
    split_dir = root / "splits"
    split_dir.mkdir(exist_ok=True)
    splits = {}
    training_sets = set()
    for seed in seeds:
        split_file = split_dir / f"seed_{seed}.tsv"
        if not split_file.is_file():
            save_pixel_split(split_file, make_pixel_split(scene.labels, seed=seed, k=k))
        split = load_pixel_split(split_file)
        if split.seed != seed or split.k != k:
            raise ValueError(f"Split metadata mismatch: {split_file}")
        validate_pixel_split(scene.labels, split)
        coordinates = tuple(map(tuple, split.train.tolist()))
        if coordinates in training_sets:
            raise ValueError(f"Duplicate training split for seed {seed}")
        training_sets.add(coordinates)
        splits[seed] = (split_file, file_sha256(split_file))

    rows = []
    for ordinal, seed in enumerate(seeds, start=1):
        split_file, split_hash = splits[seed]
        seed_dir = root / "runs" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        attempts = sorted(seed_dir.glob("attempt_*"))
        required = ("config.json", "metrics.json", "training.tsv", "split.tsv",
                    "class_accuracy.tsv", "confusion.tsv", "checkpoint.pt")
        completed = []
        for path in attempts:
            if not all((path / name).is_file() for name in required):
                continue
            try:
                previous_metrics = _validated_run(
                    path, seed=seed, split_hash=split_hash, settings=settings)
            except (OSError, ValueError, KeyError, csv.Error) as error:
                print(f"Ignoring incomplete attempt {path}: {error}", flush=True)
            else:
                completed.append((path, previous_metrics))
        if completed:
            attempt, metrics = completed[0]
            print(f"[{ordinal}/{runs}] seed {seed}: reusing {attempt}", flush=True)
        else:
            attempt_number = max((int(path.name[8:]) for path in attempts
                                  if path.name[8:].isdigit()), default=0) + 1
            attempt = seed_dir / f"attempt_{attempt_number:03d}"
            print(f"[{ordinal}/{runs}] seed {seed}: training {attempt}", flush=True)
            run_baseline(
                baseline=baseline, dataset=dataset, data_dir=data_dir,
                split_path=split_file, output_dir=attempt, seed=seed, k=k,
                epochs=epochs, learning_rate=learning_rate,
                batch_size=batch_size, test_batch_size=test_batch_size,
                device=device, mode="train")
            metrics = _validated_run(attempt, seed=seed, split_hash=split_hash,
                                     settings=settings)
        rows.append({"seed": seed, "split_sha256": split_hash,
                     "output_dir": str(attempt.relative_to(root)),
                     **{key: metrics[key] for key in METRICS}})
        _write_runs(root / "runs.tsv", rows)

    aggregated = {}
    for key in METRICS:
        values = [row[key] for row in rows]
        mean, std = statistics.fmean(values), statistics.stdev(values)
        aggregated[key] = {
            "mean": mean, "std": std,
            "mean_percent": 100 * mean, "std_percent": 100 * std,
        }
    summary = {
        "baseline": baseline, "dataset": dataset, "k": k,
        "n": runs, "seeds": seeds, "std_definition": "sample (ddof=1)",
        "units": "fractions; percent fields multiplied by 100",
        "metrics": aggregated, "experiment_dir": str(root),
        "run_table": "runs.tsv",
    }
    write_json(root / "summary.json", summary)
    with (root / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["metric", "mean", "std", "mean_percent", "std_percent"])
        for key, value in aggregated.items():
            writer.writerow([key, value["mean"], value["std"],
                             value["mean_percent"], value["std_percent"]])
    for key, value in aggregated.items():
        print(f"{key}: {value['mean_percent']:.2f} +/- {value['std_percent']:.2f}",
              flush=True)
    print(f"Saved {runs}-run summary: {root / 'summary.json'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="SSARN")
    parser.add_argument("--dataset", default="UP")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output")
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    run_benchmark(
        baseline=args.baseline, dataset=args.dataset, data_dir=args.data,
        output_dir=args.output, seed_start=args.seed_start, runs=args.runs,
        k=args.k, epochs=args.epochs, learning_rate=args.lr,
        batch_size=args.batch_size, test_batch_size=args.test_batch_size,
        device=args.device)


if __name__ == "__main__":
    main()
