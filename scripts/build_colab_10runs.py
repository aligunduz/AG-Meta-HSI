"""Generate the reusable 10-seed method comparison Colab notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "AG-Meta-HSI.ipynb"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


cells = [
    markdown("""# AG-Meta-HSI — 10 sabit split ile yöntem değerlendirmesi

Bu defter `METHOD` ile seçilen kayıtlı PyTorch yöntemini `splits/` altında önceden sabitlenmiş 10 split üzerinde çalıştırır. Varsayılan UP seed seti **90–99**, sınıf başına `k=5` eğitim pikselidir. Yöntem ve optimizer her koşuda yeniden başlatılır. OA, AA ve κ için **10 koşunun aritmetik ortalaması ± örnek standart sapması** hesaplanır. Her seed'in split dosyası, checkpoint'i ve ölçütleri ayrı saklanır. Aynı deney klasöründe defter yeniden çalıştırılırsa tamamlanmış seed'ler tekrar eğitilmez; yarım kalan seed için yeni bir attempt açılır. Yeni bir yöntem kodu `pytorch/methods.py` kayıt sistemine eklendiğinde bu defter değiştirilmez.

Makalenin UP Tablo 2 **SSARN** sütunu: OA **83,49±3,46**, AA **87,23±2,47**, κ×100 **78,74±4,25**. Makalede her sınıftan 5 rastgele örnek, 9×9 yama ve 10 denemenin ortalaması kullanılmıştır. Buradaki kod bağımsız bir PyTorch uygulamasıdır; mimari ayrıntılarının, ön işlemenin ve optimizasyon ayarlarının makale koduyla birebir aynı olduğu doğrulanmadığı için sayılar bir **referans karşılaştırmasıdır**, kesin yeniden üretim iddiası değildir.

`METHOD="QMTN"` seçildiğinde aynı SSARN üzerinde QLOML ikiz ağ eğitimi çalışır. UP Tablo 2 QMTN referansı OA **87,24±4,18**, AA **91,69±1,47**, κ×100 **83,70±5,13** değerleridir. Makale görev başına destek örneği sayısını sabitlemediği için bu uygulamanın varsayılanı 3'tür; `METHOD_CONFIG_JSON` ile değiştirilebilir. `LEARNING_RATE=0` seçilen yöntemin makaledeki varsayılanını kullanır.

Colab'da GPU çalışma ortamı seçin. Seçtiğiniz veri kümesinin iki `.mat` dosyasını `DRIVE_DATA_DIR` içine koyun veya `DATA_SOURCE="upload"` seçin. Defter kodu [GitHub deposundan](https://github.com/aligunduz/AG-Meta-HSI) çeker; yeni yerel kodları Colab'ın görmesi için **siz commit ve push etmelisiniz**. Defter commit/push yapmaz. W&B giriş anahtarını hücrelere yazmayın.
"""),
    code("""#@title Deney ayarları
import json

METHOD = "QMTN" #@param {type:"string"}
METHOD_CONFIG_JSON = "{}" #@param {type:"string"}
DATASET = "UP" #@param ["UP", "SA", "IP"]
SEED_START = 90 #@param {type:"integer"}
RUNS = 10 #@param {type:"integer"}
K = 5 #@param {type:"integer"}
EPOCHS = 300 #@param {type:"integer"}
LEARNING_RATE = 0.0 #@param {type:"number"} (0: method default)
BATCH_SIZE = 15 #@param {type:"integer"}
TEST_BATCH_SIZE = 64 #@param {type:"integer"}
DEVICE = "gpu" #@param ["gpu", "cpu"]

DATA_SOURCE = "drive" #@param ["drive", "upload"]
DRIVE_DATA_DIR = "/content/drive/MyDrive/DOKTORA/meta_learning/AG-Meta-HSI/data" #@param {type:"string"}
DRIVE_RUNS_DIR = "/content/drive/MyDrive/DOKTORA/meta_learning/AG-Meta-HSI/runs" #@param {type:"string"}
EXPERIMENT_NAME = "" #@param {type:"string"}
WANDB_PROJECT = "ag-meta-hsi" #@param {type:"string"}
WANDB_ENTITY = "" #@param {type:"string"}
LOG_WANDB = True #@param {type:"boolean"}

# Tamamlanmış bir deneyin sonuçlarını W&B'ye sonradan yüklemek için:
RUN_TRAINING = True #@param {type:"boolean"}
EXISTING_OUTPUT_DIR = "" #@param {type:"string"}

METHOD = METHOD.strip().upper()
DATASET = DATASET.strip().upper()
if LEARNING_RATE == 0:
    LEARNING_RATE = 0.002 if METHOD == "QMTN" else 0.001
assert METHOD and DATASET in {"UP", "SA", "IP"}
METHOD_CONFIG = json.loads(METHOD_CONFIG_JSON)
assert isinstance(METHOD_CONFIG, dict), "METHOD_CONFIG_JSON bir JSON nesnesi olmalı"
assert RUNS >= 2 and SEED_START >= 0 and K > 0 and EPOCHS > 0
assert BATCH_SIZE >= 2 and TEST_BATCH_SIZE > 0 and LEARNING_RATE > 0
assert DEVICE in {"gpu", "cpu"} and DATA_SOURCE in {"drive", "upload"}
assert WANDB_PROJECT.strip()
assert not EXPERIMENT_NAME or all(
    char.isalnum() or char in "-_" for char in EXPERIMENT_NAME), (
    "EXPERIMENT_NAME yalnız harf, rakam, tire veya alt çizgi içermeli")
SEEDS = list(range(SEED_START, SEED_START + RUNS))
print("Seed'ler:", SEEDS)
"""),
    code("""#@title Drive ve GitHub projesi
from google.colab import drive, files
from pathlib import Path
import subprocess, sys, json, csv

drive.mount("/content/drive")
repo_dir = Path("/content/AG-Meta-HSI")
repo_url = "https://github.com/aligunduz/AG-Meta-HSI.git"
if not repo_dir.exists():
    subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
else:
    assert (repo_dir / ".git").is_dir(), f"{repo_dir} bir Git deposu değil"
    origin = subprocess.check_output(
        ["git", "-C", str(repo_dir), "remote", "get-url", "origin"], text=True).strip()
    assert origin.rstrip("/").removesuffix(".git") == repo_url.removesuffix(".git"), (
        f"Beklenmeyen depo: {origin}")
    subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=True)
assert (repo_dir / "pytorch" / "benchmark.py").is_file(), (
    "10 koşu kodu henüz GitHub'da yok. PyCharm'daki değişiklikleri siz commit "
    "ve push ettikten sonra Colab'ı yeniden çalıştırın.")
import sys
sys.path.insert(0, str(repo_dir))
from pytorch.methods import available_methods
assert METHOD in available_methods(), (
    f"Yöntem kayıtlı değil: {METHOD}. Kullanılabilir: {available_methods()}")
repo_commit = subprocess.check_output(
    ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
print("Kullanılan Git commit:", repo_commit)
"""),
    code("""#@title Paketler ve GPU
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scipy", "wandb"], check=True)
if RUN_TRAINING:
    import torch
    print("PyTorch:", torch.__version__, "| CUDA mevcut:", torch.cuda.is_available())
    if DEVICE == "gpu":
        assert torch.cuda.is_available(), (
            "GPU bulunamadı. Colab → Runtime → Change runtime type → GPU seçin.")
        print("GPU:", torch.cuda.get_device_name(0))
"""),
    code("""#@title Veri dosyalarını hazırla
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
    print(DATASET, "verisi hazır:", data_dir)
"""),
    code("""#@title 10 seed koşusu (yarıda kalırsa aynı hücreyi yeniden çalıştırın)
if RUN_TRAINING:
    experiment_name = (EXPERIMENT_NAME.strip() or
                       f"{METHOD.lower()}_{DATASET.lower()}_seeds{SEEDS[0]}-{SEEDS[-1]}_k{K}")
    experiment_dir = (Path(DRIVE_RUNS_DIR).expanduser() / "benchmarks" /
                      experiment_name)
    command = [
        sys.executable, "-m", "pytorch.benchmark",
        "--method", METHOD, "--method-config", METHOD_CONFIG_JSON,
        "--dataset", DATASET,
        "--data", str(data_dir), "--output", str(experiment_dir),
        "--seed-start", str(SEED_START), "--runs", str(RUNS),
        "--k", str(K), "--epochs", str(EPOCHS), "--lr", str(LEARNING_RATE),
        "--batch-size", str(BATCH_SIZE),
        "--test-batch-size", str(TEST_BATCH_SIZE), "--device", DEVICE,
    ]
    print("Deney klasörü:", experiment_dir, flush=True)
    subprocess.run(command, cwd=repo_dir, check=True)
else:
    assert EXISTING_OUTPUT_DIR.strip(), "EXISTING_OUTPUT_DIR girin"
    experiment_dir = Path(EXISTING_OUTPUT_DIR).expanduser()
assert (experiment_dir / "summary.json").is_file(), (
    "10 koşu tamamlanmadı; summary.json henüz yok. Aynı deney klasöründe devam edin.")
print("Tamamlanan deney:", experiment_dir)
"""),
    code("""#@title OA / AA / κ: 10 koşunun ortalaması ± standart sapması
summary = json.loads((experiment_dir / "summary.json").read_text(encoding="utf-8"))
with (experiment_dir / "runs.tsv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
assert len(rows) == summary["n"]
print("Seed | OA (%) | AA (%) | κ×100")
for row in rows:
    print(f"{row['seed']:>4} | {100*float(row['OA']):6.2f} | "
          f"{100*float(row['AA']):6.2f} | {100*float(row['kappa']):6.2f}")
print("\\nOrtalama ± örnek std (n={})".format(summary["n"]))
for name in ("OA", "AA", "kappa"):
    values = summary["metrics"][name]
    print(f"{name}: {values['mean_percent']:.2f} ± {values['std_percent']:.2f}")
if METHOD == "SSARN" and DATASET == "UP":
    print("Makale Tablo 2 SSARN: OA 83.49±3.46 | AA 87.23±2.47 | κ×100 78.74±4.25")
elif METHOD == "QMTN" and DATASET == "UP":
    print("Makale Tablo 2 QMTN: OA 87.24±4.18 | AA 91.69±1.47 | κ×100 83.70±5.13")
print("Ayrıntılı dosyalar:", experiment_dir)
"""),
    code("""#@title 10 koşu özetini tek W&B koşusuna kaydet
if LOG_WANDB:
    sys.path.insert(0, str(repo_dir))
    from pytorch.wandb_benchmark import log_benchmark
    wandb_url = log_benchmark(
        experiment_dir, project=WANDB_PROJECT, entity=WANDB_ENTITY.strip())
    print("W&B özet koşusu:", wandb_url)
else:
    print("LOG_WANDB=False; yerel ve Drive sonuçları hazır.")
"""),
]

for index, cell in enumerate(cells, start=1):
    cell["id"] = f"ssarn-10runs-{index}"

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "AG-Meta-HSI.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
print(NOTEBOOK)
