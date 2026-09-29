"""Build the Colab notebook that clones the GitHub repository."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks"
DEST.mkdir(exist_ok=True)


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


cells = [
    markdown("""# AG-Meta-HSI — Colab baseline koşusu

Bu defter Julia/Lux ile **SSARN** supervised baseline'ını çalıştırır. `DATASET` için **UP**, **SA** veya **IP** seçebilirsiniz. Her sınıftan `K=5` eğitim merkezi seçilir; kalan etiketli pikseller test edilir. Koşu tamamlandıktan sonra OA, AA, kappa, sınıf doğrulukları ve eğitim kaybı W&B'ye yazılır.

**İlk kullanım:** Defter proje kodunu [AG-Meta-HSI GitHub deposundan](https://github.com/aligunduz/AG-Meta-HSI) klonlar. Seçtiğiniz veri kümesinin iki `.mat` dosyasını hazırlayın. Veri dosyalarını [UPV/EHU kaynağından](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) indirebilirsiniz. `DATA_SOURCE="drive"` seçerseniz dosyaları `DRIVE_DATA_DIR` klasörüne koyun; `"upload"` seçerseniz yükleme penceresi açılır. Sonuçlar Google Drive'da saklanır.

| DATASET | Gerekli dosyalar | Bant / sınıf |
|---|---|---|
| UP | `PaviaU.mat`, `PaviaU_gt.mat` | 103 / 9 |
| SA | `Salinas_corrected.mat`, `Salinas_gt.mat` | 204 / 16 |
| IP | `Indian_pines_corrected.mat`, `Indian_pines_gt.mat` | 200 / 16 |

`BASELINE` şimdilik yalnızca `SSARN` kabul eder. Kod CPU üzerinde çalışır; Colab'da GPU seçmek bu sürümü hızlandırmaz. İlk Julia paket kurulumu ve derlemesi uzun sürebilir. W&B giriş anahtarı deftere kaydedilmez.

Önceden tamamlanmış bir koşuyu yüklemek için çıktı klasörünü Drive'a kopyalayın; `RUN_TRAINING=False` ve `EXISTING_OUTPUT_DIR` değerini ayarlayın. Bu durumda Julia kurulumu ve veri dosyaları gerekmez.
"""),
    code("""#@title Koşu ayarları
BASELINE = "SSARN" #@param {type:"string"}
DATASET = "UP" #@param ["UP", "SA", "IP"]
SEED = 93 #@param {type:"integer"}
K = 5 #@param {type:"integer"}
EPOCHS = 300 #@param {type:"integer"}
LEARNING_RATE = 0.001 #@param {type:"number"}
BATCH_SIZE = 15 #@param {type:"integer"}
TEST_BATCH_SIZE = 32 #@param {type:"integer"}

DATA_SOURCE = "drive" #@param ["drive", "upload"]
DRIVE_DATA_DIR = "/content/drive/MyDrive/AG-Meta-HSI/data" #@param {type:"string"}
DRIVE_RUNS_DIR = "/content/drive/MyDrive/AG-Meta-HSI/runs" #@param {type:"string"}
WANDB_PROJECT = "ag-meta-hsi" #@param {type:"string"}
WANDB_ENTITY = "" #@param {type:"string"}

# İsterseniz daha önce tamamlanmış bir çıktı klasörünü W&B'ye gönderebilirsiniz.
RUN_TRAINING = True #@param {type:"boolean"}
EXISTING_OUTPUT_DIR = "" #@param {type:"string"}

BASELINE = BASELINE.strip().upper()
assert BASELINE == "SSARN", "Desteklenen baseline: SSARN"
assert DATASET in {"UP", "SA", "IP"}
assert K > 0 and EPOCHS > 0 and BATCH_SIZE >= 2 and TEST_BATCH_SIZE > 0
assert LEARNING_RATE > 0 and WANDB_PROJECT.strip()
"""),
    code("""#@title Drive ve proje kodu
from google.colab import drive, files
from pathlib import Path
from datetime import datetime, timezone
import shutil, subprocess, sys, uuid

drive.mount("/content/drive")
repo_dir = Path("/content/AG-Meta-HSI")
repo_url = "https://github.com/aligunduz/AG-Meta-HSI.git"
repo_commit = None
if RUN_TRAINING:
    if not repo_dir.exists():
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
    else:
        assert (repo_dir / ".git").is_dir(), f"{repo_dir} bir Git deposu değil"
        origin_url = subprocess.check_output(
            ["git", "-C", str(repo_dir), "remote", "get-url", "origin"], text=True).strip()
        assert origin_url.rstrip("/").removesuffix(".git") == repo_url.removesuffix(".git"), f"Beklenmeyen GitHub deposu: {origin_url}"
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=True)
    assert (repo_dir / "scripts" / "run_baseline.jl").is_file(), "Yeni baseline kodu bulunamadı; GitHub deposunu güncelleyin."
    repo_commit = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    print("Proje:", repo_dir, "commit:", repo_commit)
"""),
    code("""#@title Julia ve W&B kurulumu
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wandb"], check=True)
if RUN_TRAINING:
    julia = shutil.which("julia")
    if julia is None:
        subprocess.run(["bash", "-lc", "curl -fsSL https://install.julialang.org | sh -s -- -y"], check=True)
        julia = str(Path.home() / ".juliaup" / "bin" / "julia")
    assert Path(julia).exists(), "Julia kurulamadı"
    subprocess.run([julia, "--version"], check=True)
    subprocess.run([julia, f"--project={repo_dir}", "-e", "using Pkg; Pkg.instantiate()"], cwd=repo_dir, check=True)
"""),
    code("""#@title Seçilen veri kümesini hazırla
required = {
    "UP": ("PaviaU.mat", "PaviaU_gt.mat"),
    "SA": ("Salinas_corrected.mat", "Salinas_gt.mat"),
    "IP": ("Indian_pines_corrected.mat", "Indian_pines_gt.mat"),
}[DATASET]
if RUN_TRAINING:
    if DATA_SOURCE == "drive":
        data_dir = Path(DRIVE_DATA_DIR).expanduser()
    else:
        data_dir = Path("/content/hsi-data")
        data_dir.mkdir(parents=True, exist_ok=True)
        if not all((data_dir / name).is_file() for name in required):
            print("Bu dosyaları yükleyin:", ", ".join(required))
            uploaded = files.upload()
            for name in required:
                if name in uploaded:
                    (data_dir / name).write_bytes(uploaded[name])
    missing = [name for name in required if not (data_dir / name).is_file()]
    assert not missing, f"Eksik veri dosyaları: {missing}; aranan klasör: {data_dir}"
    print(DATASET, "verisi hazır:", data_dir)
else:
    data_dir = None
"""),
    code("""#@title SSARN baseline koşusu
output_root = Path(DRIVE_RUNS_DIR).expanduser()
output_root.mkdir(parents=True, exist_ok=True)
if RUN_TRAINING:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output_dir = output_root / BASELINE.lower() / DATASET.lower() / run_id
    command = [
        julia, f"--project={repo_dir}", str(repo_dir / "scripts" / "run_baseline.jl"),
        "--train", "--baseline", BASELINE, "--dataset", DATASET,
        "--data", str(data_dir), "--output", str(output_dir),
        "--seed", str(SEED), "--k", str(K), "--epochs", str(EPOCHS),
        "--lr", str(LEARNING_RATE), "--batch-size", str(BATCH_SIZE),
        "--test-batch-size", str(TEST_BATCH_SIZE),
    ]
    print("Çıktı klasörü:", output_dir, flush=True)
    subprocess.run(command, cwd=repo_dir, check=True)
else:
    output_dir = Path(EXISTING_OUTPUT_DIR).expanduser()
    assert EXISTING_OUTPUT_DIR.strip() and output_dir.is_dir(), "Geçerli bir EXISTING_OUTPUT_DIR girin"
assert (output_dir / "metrics.toml").is_file(), "Koşu tamamlanmadı; metrics.toml bulunamadı"
print("Tamamlanan koşu:", output_dir)
"""),
    code("""#@title Tamamlanan sonuçları W&B'ye kaydet
import csv, tomllib, wandb

with (output_dir / "metrics.toml").open("rb") as handle:
    metrics = tomllib.load(handle)
with (output_dir / "config.toml").open("rb") as handle:
    run_config = tomllib.load(handle)
with (output_dir / "training.tsv").open(newline="") as handle:
    history = list(csv.DictReader(handle, delimiter=chr(9)))
with (output_dir / "class_accuracy.tsv").open(newline="") as handle:
    classes = list(csv.DictReader(handle, delimiter=chr(9)))
with (output_dir / "confusion.tsv").open(newline="") as handle:
    confusion = list(csv.reader(handle, delimiter=chr(9)))

assert history and classes and confusion
assert int(history[-1]["epoch"]) == int(metrics["epoch"])
assert sum(int(row["support"]) for row in classes) == int(run_config["test_count"])
assert abs(sum(int(row["correct"]) for row in classes) /
           int(run_config["test_count"]) - metrics["OA"]) < 1e-10
assert metrics["split_sha256"] == run_config["split_sha256"]

wandb.login()
saved_baseline = run_config.get("baseline", "SSARN")
saved_dataset = run_config.get("dataset", "UP")
wb_config = {
    "baseline": saved_baseline, "dataset": saved_dataset,
    "seed": run_config["seed"], "k": run_config.get("k", 5),
    "epochs": run_config["epochs"], "learning_rate": run_config["learning_rate"],
    "batch_size": run_config["batch_size"], "test_batch_size": run_config["test_batch_size"],
    "train_count": run_config["train_count"], "test_count": run_config["test_count"],
    "bands": run_config["patch_size"][2], "class_count": len(run_config["classes"]),
    "split_sha256": run_config["split_sha256"], "output_dir": str(output_dir),
    "source_sha256": run_config["source_sha256"],
    "data_sha256": run_config["data_sha256"],
}
if repo_commit:
    wb_config["git_commit"] = repo_commit
with wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY or None,
                name=f"{saved_baseline}-{saved_dataset}-seed{run_config['seed']}-{output_dir.name}",
                group=f"{saved_baseline}-{saved_dataset}",
                job_type="baseline", config=wb_config) as run:
    run.define_metric("epoch")
    run.define_metric("train/cross_entropy", step_metric="epoch")
    for row in history:
        run.log({"epoch": int(row["epoch"]), "train/cross_entropy": float(row["train_cross_entropy"])})
    run.log({"test/OA": metrics["OA"], "test/AA": metrics["AA"],
             "test/kappa": metrics["kappa"]})
    run.summary.update({"test/OA_percent": 100 * metrics["OA"],
                        "test/AA_percent": 100 * metrics["AA"],
                        "test/kappa_x100": 100 * metrics["kappa"]})
    class_rows = []
    for row in classes:
        class_id = int(row["class"])
        accuracy = float(row["accuracy"])
        class_rows.append([class_id, int(row["support"]), int(row["correct"]), accuracy])
        run.summary[f"test/class_{class_id:02d}_accuracy"] = accuracy
    run.log({"test/class_accuracy_table": wandb.Table(
        columns=["class", "support", "correct", "accuracy"], data=class_rows)})
    run.log({"test/confusion_table": wandb.Table(
        columns=confusion[0], data=[[int(x) for x in row] for row in confusion[1:]])})
    wandb_url = run.url
print("W&B koşusu:", wandb_url)
print("OA: {:.2%} | AA: {:.2%} | kappa: {:.4f}".format(
    metrics["OA"], metrics["AA"], metrics["kappa"]))
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "SSARN_colab.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(DEST / "SSARN_colab.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

print(DEST / "SSARN_colab.ipynb")
