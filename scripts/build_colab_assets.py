"""Generate the Python/PyTorch Google Colab notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "SSARN_colab.ipynb"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


cells = [
    markdown("""# AG-Meta-HSI — PyTorch yöntem koşusu

Bu defter, [GitHub deposunu](https://github.com/aligunduz/AG-Meta-HSI) klonlayıp `METHOD` ile seçilen kayıtlı PyTorch yöntemini koşar. `DATASET` olarak **UP**, **SA** veya **IP** seçin. Sınıf başına `K=5` eğitim pikseli, kalan bütün etiketli pikseller test için kullanılır. Son epoch checkpoint'i kaydedildikten sonra OA, AA, kappa ve sınıf doğrulukları hesaplanıp W&B'ye yazılır. Yeni bir yöntem kodu kayıt sistemine eklendiğinde defter değiştirilmez.

**Başlamadan önce:** Colab'da *Runtime → Change runtime type → T4 GPU* (veya başka bir NVIDIA GPU) seçin. İki `.mat` dosyasını Google Drive'daki `DRIVE_DATA_DIR` içine koyun veya `DATA_SOURCE="upload"` seçin. Defter GitHub'dan klonladığı için **yerel değişiklikleriniz ancak siz commit ve push ettikten sonra Colab'a ulaşır**. Defter otomatik commit/push yapmaz.

| DATASET | Veri küpü | Etiketler | Bant / sınıf |
|---|---|---|---|
| UP | `PaviaU.mat` | `PaviaU_gt.mat` | 103 / 9 |
| SA | `Salinas_corrected.mat` | `Salinas_gt.mat` | 204 / 16 |
| IP | `Indian_pines_corrected.mat` | `Indian_pines_gt.mat` | 200 / 16 |

Veri: [UPV/EHU hyperspectral scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes). `SPLIT_PATH` boşken UP/seed90–99/k5 için projedeki resmi `splits/` dosyaları kullanılır. Başka bir hazır TSV ayrımı kullanmak için dosyayı Drive'a kopyalayıp `SPLIT_PATH` olarak verin. Dosya bulunmazsa eğitim hata verir; split otomatik üretilmez.

`WANDB_ENTITY` kişisel hesapta boş kalabilir; bir takım workspace'ine yazacaksanız takımın entity adını girin. W&B API anahtarını deftere yazmayın; giriş penceresi açılır. Çıktılar Drive'da kalır; veri ve checkpoint W&B'ye yüklenmez.
"""),
    code("""#@title Koşu ayarları
METHOD = "SSARN" #@param {type:"string"}
METHOD_CONFIG_JSON = "{}" #@param {type:"string"}
DATASET = "UP" #@param ["UP", "SA", "IP"]
SEED = 93 #@param {type:"integer"}
K = 5 #@param {type:"integer"}
EPOCHS = 300 #@param {type:"integer"}
LEARNING_RATE = 0.001 #@param {type:"number"}
BATCH_SIZE = 15 #@param {type:"integer"}
TEST_BATCH_SIZE = 32 #@param {type:"integer"}
DEVICE = "gpu" #@param ["gpu", "cpu"]

DATA_SOURCE = "drive" #@param ["drive", "upload"]
DRIVE_DATA_DIR = "/content/drive/MyDrive/DOKTORA/meta_learning/AG-Meta-HSI/data" #@param {type:"string"}
DRIVE_RUNS_DIR = "/content/drive/MyDrive/DOKTORA/meta_learning/AG-Meta-HSI/runs" #@param {type:"string"}
SPLIT_PATH = "" #@param {type:"string"}
WANDB_PROJECT = "ag-meta-hsi" #@param {type:"string"}
WANDB_ENTITY = "" #@param {type:"string"}

# Tamamlanan bir PyTorch koşusunu sonradan W&B'ye göndermek için:
RUN_TRAINING = True #@param {type:"boolean"}
EXISTING_OUTPUT_DIR = "" #@param {type:"string"}

METHOD = METHOD.strip().upper()
DATASET = DATASET.strip().upper()
assert METHOD, "METHOD boş olamaz."
import json
METHOD_CONFIG = json.loads(METHOD_CONFIG_JSON)
assert isinstance(METHOD_CONFIG, dict), "METHOD_CONFIG_JSON bir JSON nesnesi olmalı"
assert DATASET in {"UP", "SA", "IP"}
assert K > 0 and EPOCHS > 0 and BATCH_SIZE >= 2 and TEST_BATCH_SIZE > 0
assert LEARNING_RATE > 0 and DEVICE in {"gpu", "cpu"}
assert DATA_SOURCE in {"drive", "upload"} and WANDB_PROJECT.strip()
"""),
    code("""#@title Google Drive ve GitHub kodu
from google.colab import drive, files
from pathlib import Path
from datetime import datetime, timezone
import subprocess, sys, uuid

drive.mount("/content/drive")
repo_dir = Path("/content/AG-Meta-HSI")
repo_url = "https://github.com/aligunduz/AG-Meta-HSI.git"
repo_commit = None
if RUN_TRAINING:
    if not repo_dir.exists():
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
    else:
        assert (repo_dir / ".git").is_dir(), f"{repo_dir} bir Git deposu değil"
        origin = subprocess.check_output(
            ["git", "-C", str(repo_dir), "remote", "get-url", "origin"], text=True).strip()
        assert origin.rstrip("/").removesuffix(".git") == repo_url.removesuffix(".git"), (
            f"Beklenmeyen depo: {origin}")
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=True)
    assert (repo_dir / "pytorch" / "train.py").is_file(), (
        "PyTorch kodu GitHub deposunda yok. PyCharm'daki yerel değişiklikleri "
        "siz commit ve push ettikten sonra Colab'ı yeniden çalıştırın.")
    sys.path.insert(0, str(repo_dir))
    from pytorch.methods import available_methods
    assert METHOD in available_methods(), (
        f"Yöntem kayıtlı değil: {METHOD}. Kullanılabilir: {available_methods()}")
    repo_commit = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    print("Proje:", repo_dir, "commit:", repo_commit)
"""),
    code("""#@title Python paketleri ve GPU kontrolü
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scipy", "wandb"], check=True)
if RUN_TRAINING:
    import torch
    print("PyTorch:", torch.__version__, "| CUDA mevcut:", torch.cuda.is_available())
    if DEVICE == "gpu":
        assert torch.cuda.is_available(), (
            "GPU bulunamadı. Colab → Runtime → Change runtime type → GPU seçin.")
        print("GPU:", torch.cuda.get_device_name(0))
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
            print("Yüklenecek dosyalar:", ", ".join(required))
            uploaded = files.upload()
            for name in required:
                if name in uploaded:
                    (data_dir / name).write_bytes(uploaded[name])
    missing = [name for name in required if not (data_dir / name).is_file()]
    assert not missing, f"Eksik veri dosyaları: {missing}; aranan klasör: {data_dir}"
    if SPLIT_PATH.strip():
        assert Path(SPLIT_PATH).expanduser().is_file(), f"Split bulunamadı: {SPLIT_PATH}"
    print(DATASET, "verisi hazır:", data_dir)
"""),
    code("""#@title PyTorch yöntem koşusu
if RUN_TRAINING:
    output_root = Path(DRIVE_RUNS_DIR).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output_dir = output_root / METHOD.lower() / DATASET.lower() / run_id
    command = [
        sys.executable, "-m", "pytorch.run_method",
        "--train", "--method", METHOD, "--method-config", METHOD_CONFIG_JSON,
        "--dataset", DATASET,
        "--data", str(data_dir), "--output", str(output_dir),
        "--seed", str(SEED), "--k", str(K), "--epochs", str(EPOCHS),
        "--lr", str(LEARNING_RATE), "--batch-size", str(BATCH_SIZE),
        "--test-batch-size", str(TEST_BATCH_SIZE), "--device", DEVICE,
    ]
    if SPLIT_PATH.strip():
        command += ["--split", str(Path(SPLIT_PATH).expanduser())]
    print("Çıktı klasörü:", output_dir, flush=True)
    subprocess.run(command, cwd=repo_dir, check=True)
else:
    assert EXISTING_OUTPUT_DIR.strip(), "EXISTING_OUTPUT_DIR girin"
    output_dir = Path(EXISTING_OUTPUT_DIR).expanduser()
    assert output_dir.is_dir(), f"Çıktı klasörü yok: {output_dir}"
assert (output_dir / "metrics.json").is_file(), "Koşu tamamlanmadı; metrics.json bulunamadı"
print("Tamamlanan koşu:", output_dir)
"""),
    code("""#@title Sonuçları W&B'ye kaydet
import csv, json, hashlib, wandb

metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
run_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
with (output_dir / "training.tsv").open(encoding="utf-8", newline="") as handle:
    history = list(csv.DictReader(handle, delimiter="\t"))
with (output_dir / "class_accuracy.tsv").open(encoding="utf-8", newline="") as handle:
    classes = list(csv.DictReader(handle, delimiter="\t"))
with (output_dir / "confusion.tsv").open(encoding="utf-8", newline="") as handle:
    confusion = list(csv.reader(handle, delimiter="\t"))

assert history and classes and confusion
assert int(history[-1]["epoch"]) == int(metrics["epoch"])
assert sum(int(row["support"]) for row in classes) == int(run_config["test_count"])
assert abs(sum(int(row["correct"]) for row in classes) /
           int(run_config["test_count"]) - metrics["OA"]) < 1e-10
split_digest = hashlib.sha256((output_dir / "split.tsv").read_bytes()).hexdigest()
assert split_digest == metrics["split_sha256"] == run_config["split_sha256"]

wandb.login()
saved_baseline = run_config["baseline"]
saved_dataset = run_config["dataset"]
wb_config = dict(run_config)
wb_config["output_dir"] = str(output_dir)
if repo_commit:
    wb_config["git_commit"] = repo_commit
with wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY.strip() or None,
                name=f"{saved_baseline}-{saved_dataset}-seed{run_config['seed']}-{output_dir.name}",
                group=f"{saved_baseline}-{saved_dataset}",
                job_type="baseline", config=wb_config) as run:
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
        class_rows.append([class_id, int(row["support"]), int(row["correct"]), accuracy])
        run.summary[f"test/class_{class_id:02d}_accuracy"] = accuracy
    run.log({"test/class_accuracy_table": wandb.Table(
        columns=["class", "support", "correct", "accuracy"], data=class_rows)})
    run.log({"test/confusion_table": wandb.Table(
        columns=confusion[0], data=[[int(value) for value in row] for row in confusion[1:]])})
    wandb_url = run.url
print("W&B koşusu:", wandb_url)
print("OA: {:.2%} | AA: {:.2%} | kappa: {:.4f}".format(
    metrics["OA"], metrics["AA"], metrics["kappa"]))
"""),
]

for index, cell in enumerate(cells, start=1):
    cell["id"] = f"ssarn-{index}"

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
NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(NOTEBOOK)
