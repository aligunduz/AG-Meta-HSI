"""QMTN: the SSARN network trained with the paper's QLOML twin loop.

The query gradient is evaluated at the support-adapted network and applied
directly to the separate twin. No gradient is taken through SGD updates.
"""

from __future__ import annotations

import csv
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
from .train import (ROOT, classification_metrics, cpu_tree, evaluate,
                    resolve_device, write_json)


def resolve_meta_config(dataset: str, k: int, supplied: dict | None) -> dict:
    """Keep the published settings and expose the unspecified support split."""
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict):
        raise ValueError("method_config must be a dictionary")
    allowed = {"ways", "support_shots", "tasks_per_epoch", "inner_steps", "inner_lr"}
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError(f"Unknown QMTN settings: {sorted(unknown)}")
    classes = dataset_spec(dataset)[-1]
    config = {"ways": 5 if dataset == "UP" else 8,
              "support_shots": 3, "tasks_per_epoch": 16,
              "inner_steps": 3, "inner_lr": 0.01}
    config.update(supplied)
    for name in ("ways", "support_shots", "tasks_per_epoch", "inner_steps"):
        if isinstance(config[name], bool) or not isinstance(config[name], int):
            raise ValueError(f"{name} must be an integer")
    if not 1 <= config["ways"] <= classes:
        raise ValueError("ways must be between 1 and the dataset's class count")
    if not 1 <= config["support_shots"] < k:
        raise ValueError("support_shots must be between 1 and k-1")
    if config["tasks_per_epoch"] < 1 or config["inner_steps"] < 1:
        raise ValueError("tasks_per_epoch and inner_steps must be positive")
    if not isinstance(config["inner_lr"], (int, float)) or not math.isfinite(config["inner_lr"]) or config["inner_lr"] <= 0:
        raise ValueError("inner_lr must be positive and finite")
    return config


def sample_task(train_records: np.ndarray, *, classes: int, ways: int,
                support_shots: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw a disjoint N-way support/query partition from training pixels."""
    chosen = rng.choice(np.arange(1, classes + 1), size=ways, replace=False)
    support, query = [], []
    for class_id in chosen:
        records = train_records[train_records[:, 2] == class_id]
        order = rng.permutation(len(records))
        support.extend(records[order[:support_shots]])
        query.extend(records[order[support_shots:]])
    return np.asarray(support, dtype=np.int64), np.asarray(query, dtype=np.int64)


def query_step(model: SSARN, twin: SSARN, optimizer: torch.optim.Optimizer,
               x: torch.Tensor, y: torch.Tensor) -> float:
    """Apply the query loss gradient at theta to theta_twin (paper Eq. 14)."""
    model.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    if not torch.isfinite(loss).item():
        raise RuntimeError("Nonfinite query loss")
    loss.backward()
    for source, destination in zip(model.parameters(), twin.parameters()):
        if source.grad is None:
            raise RuntimeError("Missing query gradient")
        if not torch.isfinite(source.grad).all().item():
            raise RuntimeError("Nonfinite query gradient")
        destination.grad = source.grad.detach().clone()
    optimizer.step()
    # BatchNorm running statistics are buffers rather than gradients.
    # Carry the adapted model's statistics along with the twin parameters.
    with torch.no_grad():
        for source, destination in zip(model.buffers(), twin.buffers()):
            destination.copy_(source)
    return float(loss.item())


def run_baseline(*, baseline: str = "QMTN", dataset: str = "UP",
                 data_dir: str | Path = "data", split_path: str | Path | None = None,
                 output_dir: str | Path | None = None, seed: int = 93, k: int = 5,
                 epochs: int = 300, learning_rate: float = 0.002,
                 batch_size: int = 15, test_batch_size: int = 32,
                 device: str = "gpu", mode: str = "check",
                 method_config: dict | None = None) -> dict:
    baseline, dataset = baseline.upper(), dataset.upper()
    if baseline != "QMTN":
        raise ValueError("This runner implements QMTN")
    if mode not in ("check", "smoke", "train"):
        raise ValueError("mode must be check, smoke, or train")
    if k < 2 or epochs < 1 or batch_size < 2 or test_batch_size < 1:
        raise ValueError("Invalid k, epochs, or batch size")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Outer learning rate must be positive and finite")
    meta = resolve_meta_config(dataset, k, method_config)
    bands, classes = dataset_spec(dataset)[-2:]
    target = resolve_device(device)
    scene = load_scene(dataset, data_dir)
    split_file = (Path(split_path) if split_path is not None else
                  ROOT / "splits" / f"{dataset.lower()}_seed{seed}_k{k}.tsv")
    if not split_file.is_file():
        raise FileNotFoundError(f"Fixed split does not exist: {split_file}")
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
    twin = SSARN(bands, classes).to(target)
    twin.load_state_dict(model.state_dict())
    padded = padded_cube(scene)
    model.eval()
    with torch.inference_mode():
        xcheck = torch.from_numpy(patch_batch(padded, split.train[:1])).to(target)
        logits = model(xcheck)
    if logits.shape != (1, classes) or not torch.isfinite(logits).all().item():
        raise RuntimeError("Invalid QMTN forward check")
    print(f"QMTN forward check: {tuple(xcheck.shape)} -> {tuple(logits.shape)}; device={target}", flush=True)
    if mode == "check":
        return {"split_sha256": split_digest, "device": str(target), "meta_config": meta}

    # Only the fixed training pixels enter tasks; full test labels are held out.
    xtrain = torch.from_numpy(patch_batch(padded, split.train))
    ytrain = torch.from_numpy(split.train[:, 2].copy() - 1)
    lookup = {(int(row), int(col)): i for i, (row, col, _) in enumerate(split.train)}
    rng = np.random.default_rng(seed)
    # Section 4.2 constructs a distribution of 16 diverse tasks once.
    task_pool = [sample_task(split.train, classes=classes, ways=meta["ways"],
                             support_shots=meta["support_shots"], rng=rng)
                 for _ in range(meta["tasks_per_epoch"])]
    support_optimizer = torch.optim.SGD(model.parameters(), lr=meta["inner_lr"])
    twin_optimizer = torch.optim.Adam(twin.parameters(), lr=learning_rate)

    def task_tensors(records: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        indices = [lookup[(int(row), int(col))] for row, col, _ in records]
        return xtrain[indices].to(target), ytrain[indices].to(target)

    def run_task(task: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
        support, query = task
        sx, sy = task_tensors(support)
        qx, qy = task_tensors(query)
        model.train()
        support_loss = 0.0
        for _ in range(meta["inner_steps"]):
            support_optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(sx), sy)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite support loss")
            loss.backward()
            support_optimizer.step()
            support_loss += float(loss.item())
        return support_loss / meta["inner_steps"], query_step(model, twin, twin_optimizer, qx, qy)

    if mode == "smoke":
        before = [p.detach().clone() for p in twin.parameters()]
        support_loss, query_loss = run_task(task_pool[0])
        changed = sum(torch.count_nonzero(a != b).item() for a, b in zip(before, twin.parameters()))
        if not changed:
            raise RuntimeError("Query step changed no twin parameters")
        return {"support_loss": support_loss, "query_loss": query_loss,
                "changed_twin_parameters": changed}

    destination = (Path(output_dir) if output_dir is not None else
                   ROOT / "outputs" / f"qmtn_{dataset.lower()}_seed{seed}")
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    destination.mkdir(parents=True)
    cube_file, gt_file, *_ = dataset_spec(dataset)
    config = {
        "model": "QMTN-SSARN-QLOML-v1", "baseline": baseline,
        "method_config": {} if method_config is None else method_config,
        "effective_meta_config": meta, "framework": "PyTorch", "dataset": dataset,
        "seed": seed, "k": k, "split_seed": split.seed,
        "split_sha256": split_digest, "split_source": str(split_file),
        "epochs": epochs, "learning_rate": learning_rate,
        "optimizer": "inner SGD; outer Adam", "batch_size": batch_size,
        "task_batching": "full support/query sets; batch_size retained for runner compatibility",
        "test_batch_size": test_batch_size, "train_count": len(split.train),
        "test_count": len(split.test), "patch_size": [9, 9, bands],
        "classes": list(range(1, classes + 1)),
        "preprocessing": "raw Float32; zero padding; no augmentation",
        "selection": "fixed final epoch; test once after checkpoint",
        "query_gradient": "gradient at support-adapted SSARN copied to twin; no second-order graph",
        "epoch_transfer": "twin parameters copied to SSARN at start of next epoch",
        "task_pool": "sampled once from fixed train split; shuffled each epoch",
        "support_shots_note": "Paper varies K from 1 to 4 without fixing the reported setting; default 3 is an explicit implementation choice",
        "python_version": __import__("sys").version.split()[0],
        "torch_version": str(torch.__version__),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "device": "CUDA" if target.type == "cuda" else "CPU",
        "checkpoint": "checkpoint.pt",
        "data_sha256": {name: file_sha256(Path(data_dir) / name) for name in (cube_file, gt_file)},
        "source_sha256": {name: file_sha256(ROOT / "pytorch" / name)
                          for name in ("data.py", "ssarn.py", "train.py", "qmtn.py")},
    }
    if target.type == "cuda":
        config["gpu_name"] = torch.cuda.get_device_name(target)
    write_json(destination / "config.json", config)
    shutil.copyfile(split_file, destination / "split.tsv")
    with (destination / "training.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["epoch", "train_cross_entropy", "support_cross_entropy", "query_cross_entropy"])
        for epoch in range(1, epochs + 1):
            if epoch > 1:
                model.load_state_dict(twin.state_dict())
            losses = [run_task(task_pool[int(index)]) for index in rng.permutation(len(task_pool))]
            support_mean = float(np.mean([pair[0] for pair in losses]))
            query_mean = float(np.mean([pair[1] for pair in losses]))
            writer.writerow([epoch, query_mean, support_mean, query_mean])
            handle.flush()
            print(f"Epoch {epoch}/{epochs}; support CE={support_mean:.6f}; query CE={query_mean:.6f}", flush=True)
    if file_sha256(split_file) != split_digest:
        raise RuntimeError("Split changed during training")
    # The paper tests f_theta, not the twin. Save both states for auditing.
    torch.save({"model_state_dict": cpu_tree(model.state_dict()),
                "twin_state_dict": cpu_tree(twin.state_dict()),
                "support_optimizer_state_dict": cpu_tree(support_optimizer.state_dict()),
                "twin_optimizer_state_dict": cpu_tree(twin_optimizer.state_dict()),
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
                             results["correct"][class_id - 1], results["per_class"][class_id - 1]])
    with (destination / "confusion.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["truth/prediction", *range(1, classes + 1)])
        for class_id, row in enumerate(confusion, start=1):
            writer.writerow([class_id, *row.tolist()])
    print(f"Saved QMTN checkpoint and test metrics to {destination}", flush=True)
    return metrics
