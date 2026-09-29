"""Edit the settings below, then use PyCharm's Run action on this file."""

from datetime import datetime, timezone
from pathlib import Path

from pytorch.methods import get_method


METHOD = "SSARN"
METHOD_CONFIG = {}  # Method-specific settings; SSARN uses none.
DATASET = "UP"  # UP, SA, IP
SEED = 93
K = 5
EPOCHS = 300
LEARNING_RATE = get_method(METHOD).default_learning_rate
BATCH_SIZE = 15
TEST_BATCH_SIZE = 32
DEVICE = "gpu"  # GPU required by default; use "cpu" only when requested.
SPLIT_PATH = None  # The stored UP/seed93 split is found automatically.
LOG_WANDB = True
WANDB_PROJECT = "ag-meta-hsi"
WANDB_ENTITY = ""  # Empty uses your personal W&B account.


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    method = get_method(METHOD)
    output_dir = root / "outputs" / f"pytorch_{method.name.lower()}_{DATASET.lower()}_seed{SEED}_{run_id}"
    print(f"Training {method.name} on {DATASET} using {DEVICE}; output: {output_dir}", flush=True)
    metrics = method.runner()(
        baseline=method.name, dataset=DATASET, data_dir=root / "data",
        split_path=SPLIT_PATH, output_dir=output_dir,
        seed=SEED, k=K, epochs=EPOCHS, learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE,
        device=DEVICE, mode="train", method_config=METHOD_CONFIG,
    )
    print(f"OA={metrics['OA']:.2%} AA={metrics['AA']:.2%} kappa={metrics['kappa']:.4f}")
    if LOG_WANDB:
        from pytorch.wandb_log import log_output
        print("W&B:", log_output(output_dir, project=WANDB_PROJECT, entity=WANDB_ENTITY))
