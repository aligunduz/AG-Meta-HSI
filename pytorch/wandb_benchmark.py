"""Upload one aggregate W&B run for a completed benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from .wandb_log import load_results


def load_benchmark(directory: str | Path) -> tuple[dict, dict, list[dict]]:
    root = Path(directory)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    settings = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    with (root / "runs.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != summary["n"] or [int(row["seed"]) for row in rows] != settings["seeds"]:
        raise ValueError("Benchmark rows and seed list disagree")
    for row in rows:
        metrics, config, _, _, _ = load_results(root / row["output_dir"])
        if config["seed"] != int(row["seed"]) or config["dataset"] != settings["dataset"]:
            raise ValueError("Per-seed run configuration mismatch")
        if config["split_sha256"] != row["split_sha256"]:
            raise ValueError("Per-seed split mismatch")
        for name in ("OA", "AA", "kappa"):
            if not math.isclose(metrics[name], float(row[name]), abs_tol=1e-12):
                raise ValueError(f"Per-seed {name} mismatch")
    for name in ("OA", "AA", "kappa"):
        values = [float(row[name]) for row in rows]
        if not math.isclose(statistics.fmean(values), summary["metrics"][name]["mean"],
                            abs_tol=1e-12):
            raise ValueError(f"Aggregate {name} mean mismatch")
        if not math.isclose(statistics.stdev(values), summary["metrics"][name]["std"],
                            abs_tol=1e-12):
            raise ValueError(f"Aggregate {name} std mismatch")
    return summary, settings, rows


def log_benchmark(directory: str | Path, *, project: str = "ag-meta-hsi",
                  entity: str = "") -> str:
    import wandb

    root = Path(directory)
    summary, settings, rows = load_benchmark(root)
    marker = root / "wandb_summary_url.txt"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip():
        return marker.read_text(encoding="utf-8").strip()
    wandb.login()
    with wandb.init(
        project=project, entity=entity or None,
        name=f"{settings['baseline']}-{settings['dataset']}-{summary['n']}runs-"
             f"{settings['seeds'][0]}-{settings['seeds'][-1]}",
        group=f"{settings['baseline']}-{settings['dataset']}",
        job_type="benchmark-summary",
        config={**settings, "experiment_dir": str(root),
                "std_definition": summary["std_definition"]},
    ) as run:
        table_rows = []
        for row in rows:
            table_rows.append([
                int(row["seed"]), 100 * float(row["OA"]),
                100 * float(row["AA"]), 100 * float(row["kappa"]),
                row["split_sha256"], row["output_dir"],
            ])
        run.log({"benchmark/per_seed_metrics": wandb.Table(
            columns=["seed", "OA_percent", "AA_percent", "kappa_x100",
                     "split_sha256", "output_dir"], data=table_rows)})
        for name in ("OA", "AA", "kappa"):
            values = summary["metrics"][name]
            run.summary[f"benchmark/{name}_mean_percent"] = values["mean_percent"]
            run.summary[f"benchmark/{name}_std_percent"] = values["std_percent"]
        run.summary["benchmark/n"] = summary["n"]
        url = run.url
    if url:
        marker.write_text(url + "\n", encoding="utf-8")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project", default="ag-meta-hsi")
    parser.add_argument("--entity", default="")
    args = parser.parse_args()
    print(log_benchmark(args.output, project=args.project, entity=args.entity))


if __name__ == "__main__":
    main()
