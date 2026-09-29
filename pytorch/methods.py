"""Method registry used by the single-run and 10-run entry points.

Each runner accepts the run_baseline keyword arguments, including method_config,
and writes the standard
config, metrics, split, training, class-accuracy, confusion, and checkpoint
artifacts. Its config must include baseline, method_config, and source_sha256
for the declared source_files. Register a new method here; the notebooks do
not need code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable


@dataclass(frozen=True)
class Method:
    name: str
    module: str
    function: str
    source_files: tuple[str, ...]

    def runner(self) -> Callable:
        return getattr(import_module(self.module), self.function)


METHODS = {
    "SSARN": Method("SSARN", "pytorch.train", "run_baseline",
                    ("data.py", "ssarn.py", "train.py")),
    "QMTN": Method("QMTN", "pytorch.qmtn", "run_baseline",
                   ("data.py", "ssarn.py", "train.py", "qmtn.py")),
}


def available_methods() -> tuple[str, ...]:
    return tuple(sorted(METHODS))


def get_method(name: str) -> Method:
    key = name.strip().upper()
    if key not in METHODS:
        raise ValueError(
            f"Method {name!r} is not implemented; available: {', '.join(available_methods())}")
    return METHODS[key]
