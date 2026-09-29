"""Log a completed PyTorch SSARN run to Weights & Biases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import wandb

from .data import file_sha256


def load_results(output_dir: str | Path) -> tuple[dict, dict, list, list, list]:
    directory = Path(output_dir)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    with (directory / "training.tsv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle, delimiter="\t"))
    with (directory / "class_accuracy.tsv").open(encoding="utf-8", newline="") as handle:
        classes = list(csv.DictReader(handle, delimiter="\t"))
    with (directory / "confusion.tsv").open(encoding="utf-8", newline="") as handle:
        confusion = list(csv.reader(handle, delimiter="\t"))
    if not history or not classes or not confusion:
        raise ValueError("Incomplete training history or test metrics")
    if int(history[-1]["epoch"]) != int(metrics["epoch"]):
        raise ValueError("History and final epoch disagree")
    if sum(int(row["support"]) for row in classes) != int(config["test_count"]):
        raise ValueError("Class supports and test count disagree")
    correct = sum(int(row["correct"]) for row in classes)
    if abs(correct / int(config["test_count"]) - metrics["OA"]) > 1e-10:
        raise ValueError("Per-class counts and OA disagree")
    digest = file_sha256(directory / "split.tsv")
    if digest != metrics["split_sha256"] or digest != config["split_sha256"]:
        raise ValueError("Split SHA256 mismatch")
    return metrics, config, history, classes, confusion


def log_output(output_dir: str | Path, *, project: str = "ag-meta-hsi",
               entity: str = "") -> str:
    directory = Path(output_dir)
    metrics, config, history, classes, confusion = load_results(directory)
    wandb.login()
    with wandb.init(
        project=project, entity=entity or None,
        name=f"{config['baseline']}-{config['dataset']}-seed{config['seed']}-{directory.name}",
        group=f"{config['baseline']}-{config['dataset']}", job_type="baseline",
        config={**config, "output_dir": str(directory)},
    ) as run:
        run.define_metric("epoch")
        run.define_metric("train/cross_entropy", step_metric="epoch")
        for row in history:
            run.log({"epoch": int(row["epoch"]),
                     "train/cross_entropy": float(row["train_cross_entropy"])})
        run.log({"test/OA": metrics["OA"], "test/AA": metrics["AA"],
                 "test/kappa": metrics["kappa"]})
        run.summary.update({"test/OA_percent": 100 * metrics["OA"],
                            "test/AA_percent": 100 * metrics["AA"],
                            "test/kappa_x100": 100 * metrics["kappa"]})
        class_rows = []
        for row in classes:
            class_id = int(row["class"])
            accuracy = float(row["accuracy"])
            class_rows.append([class_id, int(row["support"]),
                               int(row["correct"]), accuracy])
            run.summary[f"test/class_{class_id:02d}_accuracy"] = accuracy
        run.log({"test/class_accuracy_table": wandb.Table(
            columns=["class", "support", "correct", "accuracy"], data=class_rows)})
        run.log({"test/confusion_table": wandb.Table(
            columns=confusion[0],
            data=[[int(value) for value in row] for row in confusion[1:]])})
        return run.url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project", default="ag-meta-hsi")
    parser.add_argument("--entity", default="")
    args = parser.parse_args()
    print(log_output(args.output, project=args.project, entity=args.entity))


if __name__ == "__main__":
    main()
