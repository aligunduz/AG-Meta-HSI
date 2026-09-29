"""First-order Proto-MAML using SSARN embeddings and QMTN's task distribution."""

from __future__ import annotations

import copy
import csv
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import (dataset_spec, file_sha256, load_pixel_split, load_scene,
                   padded_cube, patch_batch, validate_pixel_split)
from .qmtn import resolve_meta_config, sample_task
from .ssarn import SSARN
from .train import (ROOT, classification_metrics, cpu_tree, evaluate,
                    resolve_device, write_json)


SOURCE_FILES = ("data.py", "ssarn.py", "train.py", "qmtn.py", "foprotomaml.py")


def ssarn_embedding(net: SSARN, x: torch.Tensor) -> torch.Tensor:
    """Follow SSARN.forward exactly, stopping at the 32-D pooled features."""
    x = net.stem(x)
    residual = x + net.spectral_first(x)
    x = x + net.spectral_second(residual)
    x = F.leaky_relu(x, negative_slope=0.01)
    x = net.collapse(x)
    x = F.leaky_relu(x, negative_slope=0.01)
    x = net.collapse_bn(x).squeeze(2)
    x = net.spatial_residual(x)
    x = F.leaky_relu(x, negative_slope=0.01)
    return x.mean(dim=(-2, -1))


class PrototypeHead(nn.Module):
    """Task-specific linear head initialized from Euclidean prototypes."""

    def __init__(self, prototypes: torch.Tensor) -> None:
        super().__init__()
        detached = prototypes.detach()
        self.weight = nn.Parameter((2 * detached).clone())
        self.bias = nn.Parameter((-detached.square().sum(dim=1)).clone())

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.linear(embeddings, self.weight, self.bias)


def prototype_head(backbone: SSARN, support_x: torch.Tensor,
                   local_y: torch.Tensor, ways: int) -> PrototypeHead:
    with torch.no_grad():
        embeddings = ssarn_embedding(backbone, support_x)
        prototypes = torch.stack([embeddings[local_y == index].mean(dim=0)
                                  for index in range(ways)])
    if not torch.isfinite(prototypes).all().item():
        raise RuntimeError("Nonfinite or empty support prototype")
    return PrototypeHead(prototypes)


def backbone_parameters(net: SSARN) -> list[nn.Parameter]:
    return [parameter for name, parameter in net.named_parameters()
            if not name.startswith("classifier.")]


def resolve_config(dataset: str, k: int, supplied: dict | None) -> dict:
    supplied = {} if supplied is None else supplied
    if not isinstance(supplied, dict):
        raise ValueError("method_config must be a dictionary")
    allowed = {"ways", "support_shots", "tasks_per_epoch", "inner_steps",
               "inner_lr", "test_adapt_steps", "test_adapt_lr"}
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError(f"Unknown FOPROTOMAML settings: {sorted(unknown)}")
    meta = resolve_meta_config(dataset, k, {key: supplied[key] for key in supplied
                                            if key not in ("test_adapt_steps", "test_adapt_lr")})
    meta.pop("epoch_transfer")
    steps = supplied.get("test_adapt_steps", meta["inner_steps"])
    lr = supplied.get("test_adapt_lr", meta["inner_lr"])
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("test_adapt_steps must be a positive integer")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("test_adapt_lr must be positive and finite")
    return {**meta, "test_adapt_steps": steps, "test_adapt_lr": lr}


def make_task_pool(train_records: np.ndarray, *, classes: int, meta: dict,
                   rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    return [sample_task(train_records, classes=classes, ways=meta["ways"],
                        support_shots=meta["support_shots"], rng=rng)
            for _ in range(meta["tasks_per_epoch"])]


def adapt_task(meta_model: SSARN, support_x: torch.Tensor, local_y: torch.Tensor,
               *, ways: int, steps: int, lr: float) -> tuple[SSARN, PrototypeHead,
                                                               torch.optim.Optimizer, float]:
    adapted = copy.deepcopy(meta_model)
    adapted.train()
    head = prototype_head(adapted, support_x, local_y, ways)
    optimizer = torch.optim.SGD([*backbone_parameters(adapted), *head.parameters()], lr=lr)
    total = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(ssarn_embedding(adapted, support_x)), local_y)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite support loss")
        loss.backward()
        for parameter in (*backbone_parameters(adapted), *head.parameters()):
            if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                raise RuntimeError("Missing or nonfinite support gradient")
        optimizer.step()
        total += float(loss.item())
    return adapted, head, optimizer, total / steps


def meta_task_step(meta_model: SSARN, meta_optimizer: torch.optim.Optimizer,
                   support_x: torch.Tensor, support_y: torch.Tensor,
                   query_x: torch.Tensor, query_y: torch.Tensor,
                   *, ways: int, inner_steps: int, inner_lr: float
                   ) -> tuple[float, float, SSARN]:
    """Adapt a fresh copy and apply its query gradient to the meta backbone."""
    meta_optimizer.zero_grad(set_to_none=True)
    adapted, head, _, support_loss = adapt_task(
        meta_model, support_x, support_y, ways=ways, steps=inner_steps, lr=inner_lr)
    adapted.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    query_loss = F.cross_entropy(head(ssarn_embedding(adapted, query_x)), query_y)
    if not torch.isfinite(query_loss).item():
        raise RuntimeError("Nonfinite query loss")
    query_loss.backward()
    source = dict(adapted.named_parameters())
    for name, parameter in meta_model.named_parameters():
        if name.startswith("classifier."):
            continue
        gradient = source[name].grad
        if gradient is None or not torch.isfinite(gradient).all().item():
            raise RuntimeError(f"Missing or nonfinite query gradient: {name}")
        parameter.grad = gradient.detach().clone()
    meta_optimizer.step()
    with torch.no_grad():
        for meta_buffer, adapted_buffer in zip(meta_model.buffers(), adapted.buffers()):
            meta_buffer.copy_(adapted_buffer)
    return support_loss, float(query_loss.item()), adapted


class AdaptedClassifier(nn.Module):
    def __init__(self, backbone: SSARN, head: PrototypeHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(ssarn_embedding(self.backbone, x))


def run_baseline(*, baseline: str = "FOPROTOMAML", dataset: str = "UP",
                 data_dir: str | Path = "data", split_path: str | Path | None = None,
                 output_dir: str | Path | None = None, seed: int = 93, k: int = 5,
                 epochs: int = 300, learning_rate: float = 0.002,
                 batch_size: int = 15, test_batch_size: int = 32,
                 device: str = "gpu", mode: str = "check",
                 method_config: dict | None = None) -> dict:
    baseline, dataset = baseline.upper(), dataset.upper()
    if baseline != "FOPROTOMAML":
        raise ValueError("This runner implements FOPROTOMAML")
    if mode not in ("check", "smoke", "train"):
        raise ValueError("mode must be check, smoke, or train")
    if k < 2 or epochs < 1 or batch_size < 2 or test_batch_size < 1:
        raise ValueError("Invalid k, epochs, or batch size")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Outer learning rate must be positive and finite")
    meta = resolve_config(dataset, k, method_config)
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
    padded = padded_cube(scene)
    model.eval()
    with torch.inference_mode():
        xcheck = torch.from_numpy(patch_batch(padded, split.train[:1])).to(target)
        embedding = ssarn_embedding(model, xcheck)
    if embedding.shape != (1, 32) or not torch.isfinite(embedding).all().item():
        raise RuntimeError("Invalid FOPROTOMAML embedding check")
    print(f"FOPROTOMAML embedding check: {tuple(xcheck.shape)} -> {tuple(embedding.shape)}; device={target}", flush=True)
    if mode == "check":
        return {"split_sha256": split_digest, "device": str(target), "meta_config": meta}

    xtrain = torch.from_numpy(patch_batch(padded, split.train))
    ytrain = torch.from_numpy(split.train[:, 2].copy() - 1)
    lookup = {(int(row), int(col)): i for i, (row, col, _) in enumerate(split.train)}
    rng = np.random.default_rng(seed)
    task_pool = make_task_pool(split.train, classes=classes, meta=meta, rng=rng)
    meta_optimizer = torch.optim.Adam(backbone_parameters(model), lr=learning_rate)

    def task_tensors(records: np.ndarray, class_ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        indices = [lookup[(int(row), int(col))] for row, col, _ in records]
        labels = np.searchsorted(class_ids, records[:, 2])
        return xtrain[indices].to(target), torch.from_numpy(labels.copy()).to(target)

    def run_task(task: tuple[np.ndarray, np.ndarray]) -> tuple[float, float, SSARN]:
        support, query = task
        class_ids = np.unique(support[:, 2])
        sx, sy = task_tensors(support, class_ids)
        qx, qy = task_tensors(query, class_ids)
        return meta_task_step(model, meta_optimizer, sx, sy, qx, qy,
                              ways=meta["ways"], inner_steps=meta["inner_steps"],
                              inner_lr=meta["inner_lr"])

    if mode == "smoke":
        before = [p.detach().clone() for p in backbone_parameters(model)]
        support_loss, query_loss, _ = run_task(task_pool[0])
        changed = sum(torch.count_nonzero(a != b).item()
                      for a, b in zip(before, backbone_parameters(model)))
        if not changed:
            raise RuntimeError("Meta optimizer changed no backbone parameters")
        return {"support_loss": support_loss, "query_loss": query_loss,
                "changed_meta_parameters": changed}

    destination = (Path(output_dir) if output_dir is not None else
                   ROOT / "outputs" / f"foprotomaml_{dataset.lower()}_seed{seed}")
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    destination.mkdir(parents=True)
    cube_file, gt_file, *_ = dataset_spec(dataset)
    config = {
        "model": "FOProtoMAML-SSARN-v1", "baseline": baseline,
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
        "test_protocol": "deepcopy final meta backbone; initialize C-way prototypes from all split.train pixels; adapt backbone and head with SGD; eval running statistics; classify split.test once",
        "bn_protocol": "task copy starts with meta buffers; support and query forwards update copy in train mode; copy buffers to meta after each task; test copy adapts in train mode and evaluates in eval mode",
        "task_pool": "sampled once from fixed train split with QMTN sample_task; shuffled each epoch",
        "python_version": __import__("sys").version.split()[0],
        "torch_version": str(torch.__version__),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "device": "CUDA" if target.type == "cuda" else "CPU",
        "checkpoint": "checkpoint.pt",
        "data_sha256": {name: file_sha256(Path(data_dir) / name)
                        for name in (cube_file, gt_file)},
        "source_sha256": {name: file_sha256(ROOT / "pytorch" / name)
                          for name in SOURCE_FILES},
    }
    if target.type == "cuda":
        config["gpu_name"] = torch.cuda.get_device_name(target)
    write_json(destination / "config.json", config)
    shutil.copyfile(split_file, destination / "split.tsv")
    with (destination / "training.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["epoch", "train_cross_entropy", "support_cross_entropy", "query_cross_entropy"])
        for epoch in range(1, epochs + 1):
            losses = [run_task(task_pool[int(index)])[:2]
                      for index in rng.permutation(len(task_pool))]
            support_mean = float(np.mean([pair[0] for pair in losses]))
            query_mean = float(np.mean([pair[1] for pair in losses]))
            writer.writerow([epoch, query_mean, support_mean, query_mean])
            handle.flush()
            print(f"Epoch {epoch}/{epochs}; support CE={support_mean:.6f}; query CE={query_mean:.6f}", flush=True)

    if file_sha256(split_file) != split_digest:
        raise RuntimeError("Split changed during training")
    adapted, head, test_optimizer, _ = adapt_task(
        model, xtrain.to(target), ytrain.to(target), ways=classes,
        steps=meta["test_adapt_steps"], lr=meta["test_adapt_lr"])
    torch.save({"model_state_dict": cpu_tree(adapted.state_dict()),
                "meta_model_state_dict": cpu_tree(model.state_dict()),
                "test_head_state_dict": cpu_tree(head.state_dict()),
                "optimizer_state_dict": cpu_tree(meta_optimizer.state_dict()),
                "test_optimizer_state_dict": cpu_tree(test_optimizer.state_dict()),
                "epoch": epochs, "config": config}, destination / "checkpoint.pt")
    classifier = AdaptedClassifier(adapted, head)
    confusion = evaluate(classifier, padded, split.test, batch_size=test_batch_size,
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
    print(f"Saved FOPROTOMAML checkpoint and test metrics to {destination}", flush=True)
    return metrics
