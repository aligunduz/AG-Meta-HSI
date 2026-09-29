"""Run one registered method on ten fixed splits; rerun to resume."""

from pathlib import Path

from pytorch.benchmark import run_benchmark


METHOD = "SSARN"
METHOD_CONFIG = {}  # Method-specific settings; SSARN uses none.
DATASET = "UP"  # UP, SA, IP
SEED_START = 90
RUNS = 10
K = 5
EPOCHS = 300
LEARNING_RATE = 0.002 if METHOD == "QMTN" else 0.001
BATCH_SIZE = 15
TEST_BATCH_SIZE = 64
DEVICE = "gpu"
EXPERIMENT_NAME = ""  # Same name resumes; change it for different training settings
LOG_WANDB = True
WANDB_PROJECT = "ag-meta-hsi"
WANDB_ENTITY = ""


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    last_seed = SEED_START + RUNS - 1
    default_name = (
        f"benchmark_{METHOD.lower()}_{DATASET.lower()}_"
        f"seeds{SEED_START}-{last_seed}_k{K}_e{EPOCHS}")
    assert not EXPERIMENT_NAME or all(
        char.isalnum() or char in "-_" for char in EXPERIMENT_NAME)
    output_dir = root / "outputs" / (EXPERIMENT_NAME or default_name)
    summary = run_benchmark(
        baseline=METHOD, dataset=DATASET, data_dir=root / "data",
        output_dir=output_dir, seed_start=SEED_START, runs=RUNS, k=K,
        epochs=EPOCHS, learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE,
        device=DEVICE, method_config=METHOD_CONFIG,
    )
    print(f"Experiment: {output_dir}")
    for name in ("OA", "AA", "kappa"):
        values = summary["metrics"][name]
        print(f"{name}: {values['mean_percent']:.2f} +/- {values['std_percent']:.2f}")
    if LOG_WANDB:
        from pytorch.wandb_benchmark import log_benchmark
        print("W&B summary:",
              log_benchmark(output_dir, project=WANDB_PROJECT, entity=WANDB_ENTITY))
