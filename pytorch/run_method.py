"""Run one registered method on a fixed split."""

from __future__ import annotations

import argparse
import json

from .methods import get_method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--check", action="store_true")
    parser.add_argument("--method", "--baseline", dest="method", default="SSARN")
    parser.add_argument("--method-config", default="{}", help="JSON object of method-specific settings")
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
    method = get_method(args.method)
    method_config = json.loads(args.method_config)
    if not isinstance(method_config, dict):
        parser.error("--method-config must be a JSON object")
    method.runner()(
        baseline=method.name, dataset=args.dataset, data_dir=args.data,
        split_path=args.split, output_dir=args.output, seed=args.seed, k=args.k,
        epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size,
        test_batch_size=args.test_batch_size, device=args.device,
        mode="train" if args.train else "smoke" if args.smoke else "check",
        method_config=method_config)


if __name__ == "__main__":
    main()
