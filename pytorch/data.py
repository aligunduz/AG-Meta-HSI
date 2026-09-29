"""Hyperspectral scenes, fixed pixel splits, and zero-padded SSARN patches."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DATASET_SPECS = {
    "UP": ("PaviaU.mat", "PaviaU_gt.mat", "paviaU", "paviaU_gt", 103, 9),
    "SA": ("Salinas_corrected.mat", "Salinas_gt.mat", "salinas_corrected", "salinas_gt", 204, 16),
    "IP": ("Indian_pines_corrected.mat", "Indian_pines_gt.mat", "indian_pines_corrected", "indian_pines_gt", 200, 16),
}


@dataclass(frozen=True)
class Scene:
    cube: np.ndarray  # height, width, bands; raw Float32
    labels: np.ndarray  # height, width; 0 is background


@dataclass(frozen=True)
class PixelSplit:
    seed: int
    k: int
    train: np.ndarray  # 1-based row, 1-based col, class ID
    test: np.ndarray


def dataset_spec(dataset: str) -> tuple[str, str, str, str, int, int]:
    key = dataset.upper()
    if key not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset {dataset}; choose UP, SA, or IP")
    return DATASET_SPECS[key]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scene(dataset: str, data_dir: str | Path) -> Scene:
    cube_file, gt_file, cube_key, gt_key, bands, classes = dataset_spec(dataset)
    directory = Path(data_dir)
    cube_path, gt_path = directory / cube_file, directory / gt_file
    for path in (cube_path, gt_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")
    cube_data, gt_data = loadmat(cube_path), loadmat(gt_path)
    if cube_key not in cube_data or gt_key not in gt_data:
        raise ValueError(f"Expected MATLAB variables {cube_key} and {gt_key}")
    cube = np.asarray(cube_data[cube_key], dtype=np.float32)
    labels = np.asarray(gt_data[gt_key], dtype=np.int64)
    if cube.ndim != 3 or labels.ndim != 2 or cube.shape[:2] != labels.shape:
        raise ValueError("Expected matching height × width × bands cube and 2-D labels")
    if cube.shape[2] != bands or np.any(labels < 0):
        raise ValueError(f"Invalid spectral band count or negative label for {dataset}")
    if not np.array_equal(np.unique(labels[labels != 0]), np.arange(1, classes + 1)):
        raise ValueError(f"Expected class IDs 1:{classes} for {dataset}")
    return Scene(cube, labels)


def make_pixel_split(labels: np.ndarray, *, seed: int = 93, k: int = 5) -> PixelSplit:
    if k <= 0:
        raise ValueError("k must be positive")
    rng = np.random.default_rng(seed)
    train, test = [], []
    for class_id in np.unique(labels):
        if class_id == 0:
            continue
        coordinates = np.argwhere(labels == class_id)
        if len(coordinates) <= k:
            raise ValueError(f"Class {class_id} must have more than {k} pixels")
        shuffled = coordinates[rng.permutation(len(coordinates))]
        train.extend((int(row + 1), int(col + 1), int(class_id)) for row, col in shuffled[:k])
        test.extend((int(row + 1), int(col + 1), int(class_id)) for row, col in shuffled[k:])
    order = lambda record: (record[2], record[0], record[1])
    split = PixelSplit(seed, k, np.asarray(sorted(train, key=order), dtype=np.int64),
                       np.asarray(sorted(test, key=order), dtype=np.int64))
    validate_pixel_split(labels, split)
    return split


def validate_pixel_split(labels: np.ndarray, split: PixelSplit) -> None:
    if split.k <= 0 or split.train.ndim != 2 or split.test.ndim != 2:
        raise ValueError("Invalid split structure")
    if split.train.shape[1] != 3 or split.test.shape[1] != 3:
        raise ValueError("Expected row, col, class columns")
    labeled_count = int(np.count_nonzero(labels))
    records = np.concatenate((split.train, split.test))
    if len(records) != labeled_count:
        raise ValueError("Split must cover every labeled pixel exactly once")
    rows, cols, class_ids = records.T
    height, width = labels.shape
    if np.any(rows < 1) or np.any(rows > height) or np.any(cols < 1) or np.any(cols > width):
        raise ValueError("Split coordinate outside image")
    if np.any(class_ids <= 0) or not np.array_equal(labels[rows - 1, cols - 1], class_ids):
        raise ValueError("Split label does not match ground truth")
    if len(np.unique((rows - 1) * width + cols - 1)) != labeled_count:
        raise ValueError("Duplicate pixel in split")
    classes = np.unique(labels[labels != 0])
    counts = np.bincount(split.train[:, 2], minlength=int(classes.max()) + 1)
    if np.any(counts[classes] != split.k) or np.count_nonzero(counts) != len(classes):
        raise ValueError("Train split must contain k pixels from each class")


def save_pixel_split(path: str | Path, split: PixelSplit) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"# seed={split.seed} k={split.k}\nset\tclass\trow\tcol\n")
        for group, records in (("train", split.train), ("test", split.test)):
            for row, col, class_id in records:
                handle.write(f"{group}\t{class_id}\t{row}\t{col}\n")


def load_pixel_split(path: str | Path) -> PixelSplit:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("Incomplete split file")
    header = re.fullmatch(r"# seed=(-?\d+) k=(\d+)", lines[0])
    if header is None or lines[1] != "set\tclass\trow\tcol":
        raise ValueError("Invalid split TSV header")
    groups: dict[str, list[tuple[int, int, int]]] = {"train": [], "test": []}
    for line in lines[2:]:
        fields = line.split("\t")
        if len(fields) != 4 or fields[0] not in groups:
            raise ValueError(f"Invalid split TSV row: {line}")
        group, class_id, row, col = fields
        groups[group].append((int(row), int(col), int(class_id)))
    return PixelSplit(int(header[1]), int(header[2]),
                      np.asarray(groups["train"], dtype=np.int64).reshape(-1, 3),
                      np.asarray(groups["test"], dtype=np.int64).reshape(-1, 3))


def padded_cube(scene: Scene, patch_size: int = 9) -> np.ndarray:
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("patch_size must be positive and odd")
    radius = patch_size // 2
    return np.pad(scene.cube, ((radius, radius), (radius, radius), (0, 0)))


def patch_batch(padded: np.ndarray, records: np.ndarray, patch_size: int = 9) -> np.ndarray:
    bands = padded.shape[2]
    batch = np.empty((len(records), 1, bands, patch_size, patch_size), dtype=np.float32)
    for index, (row, col, _) in enumerate(records):
        # TSV coordinates are 1-based, and padding shifts the upper-left crop.
        top, left = int(row - 1), int(col - 1)
        batch[index, 0] = padded[top:top + patch_size, left:left + patch_size].transpose(2, 0, 1)
    return batch
