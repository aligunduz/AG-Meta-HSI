"""Generate the native Julia Colab notebook (the notebook itself has only Julia code)."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks" / "SSARN_colab_julia.ipynb"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


cells = [
    markdown("""# AG-Meta-HSI — Julia ile Colab baseline koşusu

Bu defterin **bütün kod hücreleri Julia** dilindedir. Colab'da **Runtime → Change runtime type → Julia** seçin ve hücreleri sırayla çalıştırın. [Proje GitHub'dan](https://github.com/aligunduz/AG-Meta-HSI) indirilir. Şu an yalnızca `BASELINE = "SSARN"` desteklenir. `DATASET = "UP"`, `"SA"` veya `"IP"` seçebilirsiniz.

| DATASET | Gerekli iki dosya |
|---|---|
| UP | `PaviaU.mat`, `PaviaU_gt.mat` |
| SA | `Salinas_corrected.mat`, `Salinas_gt.mat` |
| IP | `Indian_pines_corrected.mat`, `Indian_pines_gt.mat` |

Verileri [UPV/EHU sayfasından](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) temin edin. Varsayılan `DATA_SOURCE = "upload"` için iki dosyayı Colab'ın sol **Files** panelinden `/content/hsi-data` klasörüne yükleyin (gerekirse klasörü oluşturun). Dosyalar mevcutsa yüklemeyi tekrarlamanız gerekmez. `DATA_SOURCE = "drive"` seçerseniz Drive'ın **önceden bağlanmış** olması ve `DRIVE_DATA_DIR` yolunun var olması gerekir. Julia çalışma ortamında Python'daki `google.colab.drive.mount` hücresi kullanılmaz; Drive bağlanamıyorsa `upload` kullanın.

Çıktılar varsayılan olarak `/content/ag-meta-hsi-runs` altındadır. Kalıcı saklama için Colab Files panelinden indirin. Drive bağlıysa `OUTPUT_SOURCE = "drive"` seçebilirsiniz. W&B'ye OA, AA, kappa, sınıf doğrulukları, karışıklık tablosu ve epoch başına eğitim kaybı kaydedilir; `.mat` verileri ve checkpoint W&B'ye gönderilmez. İlk Julia/CUDA paket kurulumu ve derlemesi zaman alabilir. Colab'da **Julia + GPU** çalışma ortamını seçin. `DEVICE = "gpu"` varsayılandır; GPU yoksa eğitim hata verir.

**Önemli:** Bu yeni defter GitHub'da görünmeden önce yerel değişiklikleri sizin commit edip push etmeniz gerekir. W&B anahtarını deftere yazmayın; aşağıdaki giriş hücresi gerektiğinde sizden ister. `WANDB_ENTITY = ""` kişisel hesabın varsayılan entity'sini kullanır.
"""),
    code("""# Koşu ayarları — Julia
BASELINE = "SSARN"
DATASET = "UP"                 # UP, SA, IP
SEED = 93
K = 5
EPOCHS = 300
LEARNING_RATE = 0.001
BATCH_SIZE = 15
TEST_BATCH_SIZE = 32
DEVICE = "gpu"                 # gpu veya cpu; Colab için gpu

DATA_SOURCE = "upload"        # upload veya drive
OUTPUT_SOURCE = "local"       # local veya drive
LOCAL_DATA_DIR = "/content/hsi-data"
LOCAL_RUNS_DIR = "/content/ag-meta-hsi-runs"
DRIVE_DATA_DIR = "/content/drive/MyDrive/DOKTORA/meta_learning/AG-Meta-HSI/data"
DRIVE_RUNS_DIR = "/content/drive/MyDrive/DOKTORA/meta_learning/AG-Meta-HSI/runs"

WANDB_PROJECT = "ag-meta-hsi"
WANDB_ENTITY = ""              # Boş bırakılabilir; takım projesiyse entity adını yazın.
RUN_TRAINING = true
EXISTING_OUTPUT_DIR = ""       # Sadece eski sonuçları W&B'ye göndermek için.

BASELINE = uppercase(strip(BASELINE))
DATASET = uppercase(strip(DATASET))
BASELINE == "SSARN" || error("Desteklenen baseline: SSARN")
DATASET in ("UP", "SA", "IP") || error("DATASET UP, SA veya IP olmalı")
DATA_SOURCE in ("upload", "drive") || error("DATA_SOURCE upload veya drive olmalı")
OUTPUT_SOURCE in ("local", "drive") || error("OUTPUT_SOURCE local veya drive olmalı")
DEVICE in ("gpu", "cpu") || error("DEVICE gpu veya cpu olmalı")
K > 0 && EPOCHS > 0 && BATCH_SIZE >= 2 && TEST_BATCH_SIZE > 0 || error("Geçersiz eğitim ayarı")
isfinite(LEARNING_RATE) && LEARNING_RATE > 0 || error("LEARNING_RATE pozitif ve sonlu olmalı")
!isempty(strip(WANDB_PROJECT)) || error("WANDB_PROJECT boş olamaz")
println("Julia ", VERSION, " | ", BASELINE, " / ", DATASET, " | istenen cihaz: ", DEVICE)
"""),
    code("""# Projeyi GitHub'dan çek ve Julia bağımlılıklarını kur.
using Pkg

repo_url = "https://github.com/aligunduz/AG-Meta-HSI.git"
repo_dir = "/content/AG-Meta-HSI"
if !isdir(repo_dir)
    run(`git clone $repo_url $repo_dir`)
else
    isdir(joinpath(repo_dir, ".git")) || error("$repo_dir bir Git deposu değil")
    origin = strip(read(`git -C $repo_dir remote get-url origin`, String))
    replace(rstrip(origin, '/'), r"\\.git$" => "") == replace(repo_url, r"\\.git$" => "") ||
        error("Beklenmeyen GitHub deposu: $origin")
    run(`git -C $repo_dir pull --ff-only`)
end
isfile(joinpath(repo_dir, "scripts", "run_baseline.jl")) || error("Baseline kodu eksik; depoyu güncelleyin")
if RUN_TRAINING && DEVICE == "gpu"
    project_text = read(joinpath(repo_dir, "Project.toml"), String)
    occursin(r"(?m)^CUDA\\s*=", project_text) ||
        error("GPU kodu henüz GitHub deposunda değil. Yerel değişiklikleri commit edip push edin.")
end
repo_commit = strip(read(`git -C $repo_dir rev-parse HEAD`, String))
Pkg.activate(repo_dir)
Pkg.instantiate()
using AGMetaHSI, CUDA, LuxCUDA
if RUN_TRAINING && DEVICE == "gpu"
    CUDA.functional(true) || error("CUDA GPU bulunamadı. Colab'da Julia + GPU runtime seçin.")
    println("CUDA GPU: ", CUDA.name(CUDA.device()))
end
println("Proje: $repo_dir | commit: $repo_commit")
"""),
    code("""# Veri dosyalarını kontrol et.
required = Dict(
    "UP" => ("PaviaU.mat", "PaviaU_gt.mat"),
    "SA" => ("Salinas_corrected.mat", "Salinas_gt.mat"),
    "IP" => ("Indian_pines_corrected.mat", "Indian_pines_gt.mat"),
)[DATASET]
data_dir = DATA_SOURCE == "drive" ? DRIVE_DATA_DIR : LOCAL_DATA_DIR
if RUN_TRAINING
    if DATA_SOURCE == "drive"
        isdir("/content/drive/MyDrive") || error("Drive bağlı değil. Files panelinden bağlayın veya DATA_SOURCE değerini upload yapın.")
    else
        mkpath(data_dir)
    end
    missing = filter(name -> !isfile(joinpath(data_dir, name)), required)
    isempty(missing) || error("Eksik dosyalar: $(join(missing, ", ")). Files panelinden $data_dir klasörüne yükleyin.")
    println("Veri hazır: $data_dir")
end
"""),
    code("""# SSARN eğitimini doğrudan Julia içinde çalıştır.
using Dates, Random

output_root = OUTPUT_SOURCE == "drive" ? DRIVE_RUNS_DIR : LOCAL_RUNS_DIR
if RUN_TRAINING
    OUTPUT_SOURCE == "drive" && !isdir("/content/drive/MyDrive") &&
        error("Drive bağlı değil. OUTPUT_SOURCE değerini local yapın veya Drive'ı bağlayın.")
    mkpath(output_root)
    split_dir = joinpath(output_root, "splits")
    mkpath(split_dir)
    split_path = joinpath(split_dir, "$(lowercase(DATASET))_split_seed$(SEED)_k$(K).tsv")
    if !isfile(split_path)
        scene = AGMetaHSI.load_hsi_dataset(DATASET, data_dir)
        split = AGMetaHSI.make_pixel_split(scene.labels; k=K, seed=SEED)
        AGMetaHSI.save_pixel_split(split_path, split)
        println("Split oluşturuldu: $split_path")
    end
    run_id = Dates.format(Dates.now(), dateformat"yyyymmddTHHMMSS") * "_" * Random.randstring(8)
    output_dir = joinpath(output_root, lowercase(BASELINE), lowercase(DATASET), run_id)
    println("Çıktı klasörü: $output_dir")
    result = AGMetaHSI.run_supervised(; dataset=DATASET, data_dir=data_dir,
        split_path=split_path, output_dir=output_dir, seed=SEED, k=K,
        epochs=EPOCHS, learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE,
        device=Symbol(DEVICE), train=true)
else
    isempty(strip(EXISTING_OUTPUT_DIR)) && error("EXISTING_OUTPUT_DIR belirtin")
    output_dir = EXISTING_OUTPUT_DIR
    isdir(output_dir) || error("Çıktı klasörü bulunamadı: $output_dir")
end
isfile(joinpath(output_dir, "metrics.toml")) || error("Koşu tamamlanmadı: metrics.toml yok")
println("Tamamlanan koşu: $output_dir")
"""),
    code("""# W&B için ayrı Julia ortamı: projenin Project.toml dosyasını değiştirmez.
wandb_env = "/content/ag-meta-hsi-wandb-env"
Pkg.activate(wandb_env)
if !isfile(joinpath(wandb_env, "Project.toml"))
    Pkg.add(Pkg.PackageSpec(name="Wandb", version="0.5.6"))
else
    Pkg.instantiate()
end
using Wandb, TOML

# IJulia'nın gizli giriş alanı anahtarı defter çıktısında göstermez.
# Colab Secrets Julia oturumundan doğrudan erişilebilir olmayabilir.
if isdefined(Main, :IJulia) && !haskey(ENV, "WANDB_API_KEY")
    Wandb.login(key=Main.IJulia.readprompt("W&B API key: ", password=true))
else
    Wandb.login()
end
"""),
    code("""# Tamamlanan koşunun tüm genel metriklerini W&B'ye gönder.
metrics = TOML.parsefile(joinpath(output_dir, "metrics.toml"))
run_config = TOML.parsefile(joinpath(output_dir, "config.toml"))
history_lines = readlines(joinpath(output_dir, "training.tsv"))
class_lines = readlines(joinpath(output_dir, "class_accuracy.tsv"))
confusion_lines = readlines(joinpath(output_dir, "confusion.tsv"))
length(history_lines) > 1 || error("Eğitim geçmişi boş")
length(class_lines) > 1 || error("Sınıf doğrulukları boş")

wb_config = Dict{String,Any}(
    "baseline" => run_config["baseline"], "dataset" => run_config["dataset"],
    "seed" => run_config["seed"], "k" => run_config["k"],
    "epochs" => run_config["epochs"], "learning_rate" => run_config["learning_rate"],
    "batch_size" => run_config["batch_size"], "test_batch_size" => run_config["test_batch_size"],
    "device" => run_config["device"],
    "train_count" => run_config["train_count"], "test_count" => run_config["test_count"],
    "bands" => run_config["patch_size"][3], "class_count" => length(run_config["classes"]),
    "split_sha256" => run_config["split_sha256"], "output_dir" => output_dir,
    "source_sha256" => run_config["source_sha256"], "data_sha256" => run_config["data_sha256"],
)
haskey(run_config, "gpu_name") && (wb_config["gpu_name"] = run_config["gpu_name"])
RUN_TRAINING && (wb_config["git_commit"] = repo_commit)
name = "$(run_config["baseline"])-$(run_config["dataset"])-seed$(run_config["seed"])-$(basename(output_dir))"
entity = isempty(strip(WANDB_ENTITY)) ? nothing : strip(WANDB_ENTITY)
wb = Wandb.WandbLogger(; project=WANDB_PROJECT, entity=entity, name=name,
    group="$(run_config["baseline"])-$(run_config["dataset"])",
    job_type="baseline", config=wb_config, settings=Wandb.wandb.Settings())
wandb_url = string(wb.wrun.url)
try
    for line in history_lines[2:end]
        fields = split(line, '\t')
        Wandb.log(wb, Dict("epoch" => parse(Int, fields[1]),
                           "train/cross_entropy" => parse(Float64, fields[2])))
    end
    nclasses = length(class_lines) - 1
    class_table = Matrix{Any}(undef, nclasses, 4)
    class_scalars = Dict{String,Any}()
    for (i, line) in enumerate(class_lines[2:end])
        fields = split(line, '\t')
        class_id = parse(Int, fields[1])
        accuracy = parse(Float64, fields[4])
        class_table[i, :] = [class_id, parse(Int, fields[2]), parse(Int, fields[3]), accuracy]
        class_scalars["test/class_$(lpad(class_id, 2, '0'))_accuracy"] = accuracy
    end
    confusion_header = split(confusion_lines[1], '\t')
    confusion_table = Matrix{Any}(undef, length(confusion_lines)-1, length(confusion_header))
    for (i, line) in enumerate(confusion_lines[2:end])
        confusion_table[i, :] = parse.(Int, split(line, '\t'))
    end
    test_log = Dict{String,Any}(
        "test/OA" => metrics["OA"], "test/AA" => metrics["AA"],
        "test/kappa" => metrics["kappa"],
        "test/OA_percent" => 100 * metrics["OA"],
        "test/AA_percent" => 100 * metrics["AA"],
        "test/kappa_x100" => 100 * metrics["kappa"],
        "test/class_accuracy_table" => Wandb.Table(; data=class_table,
            columns=["class", "support", "correct", "accuracy"]),
        "test/confusion_table" => Wandb.Table(; data=confusion_table,
            columns=confusion_header),
    )
    merge!(test_log, class_scalars)
    Wandb.log(wb, test_log)
finally
    close(wb)
end
println("W&B koşusu: $wandb_url")
println("OA: $(round(100 * metrics["OA"], digits=2))% | AA: $(round(100 * metrics["AA"], digits=2))% | kappa: $(round(metrics["kappa"], digits=4))")
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": DEST.name},
        "kernelspec": {"display_name": "Julia", "language": "julia", "name": "julia"},
        "language_info": {"name": "julia"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
DEST.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(DEST)
