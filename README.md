# AG-Meta-HSI — PyTorch yöntem karşılaştırması

Bu proje, hiperspektral görüntüler için kayıtlı PyTorch yöntemlerini aynı
sabit split'lerde karşılaştırır. **Supervised SSARN** ve makaledeki
**SSARN + QLOML = QMTN** ayrı baseline'lar olarak uygulanmıştır.
`METHOD="SSARN"` veya `METHOD="QMTN"`; `DATASET` olarak `UP` (Pavia
University), `SA` (Salinas) veya `IP` (Indian Pines) seçilebilir.

## PyCharm'da çalıştırma

Proje klasörünü PyCharm'da açın. Interpreter olarak
`.venv\Scripts\python.exe` seçin. [`run_pytorch.py`](run_pytorch.py)
dosyasının üstündeki `METHOD`, `DATASET`, `DEVICE` ve diğer ayarları
düzenleyin; ardından dosyaya sağ tıklayıp **Run 'run_pytorch'** seçin.
Varsayılan ayarlar `UP`, seed `93`, sınıf başına `5` eğitim pikseli,
`300` epoch ve GPU'dur. Eğitim tamamlandığında `LOG_WANDB=True` ile
genel ölçütler ve eğitim eğrisi W&B'ye kaydedilir.
`WANDB_ENTITY=""` kişisel hesap için uygundur.

Terminalde kontrol veya tam eğitim:

```powershell
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv\Scripts\python -m pytorch.run_method --check --method SSARN --dataset UP --device gpu
.venv\Scripts\python -m pytorch.run_method --smoke --method SSARN --dataset UP --device gpu
.venv\Scripts\python -m pytorch.run_method --train --method SSARN --dataset UP --device gpu --epochs 300 --seed 93 --k 5 --output outputs/pytorch_ssarn_up_seed93
.venv\Scripts\python -m pytorch.run_method --check --method QMTN --dataset UP --device gpu
.venv\Scripts\python -m pytorch.run_method --smoke --method QMTN --dataset UP --device gpu
.venv\Scripts\python -m pytorch.run_method --train --method QMTN --dataset UP --device gpu --epochs 300 --seed 93 --k 5 --output outputs/pytorch_qmtn_up_seed93
```

`--check` tek yamanın ileri geçişini; `--smoke` bir Adam güncellemesini
doğrular. Eğitim yalnızca açık `--train` ile başlar.
`--device gpu` CUDA kullanılamıyorsa hata verir; CPU'ya sessizce geçmez.
Bu bilgisayardaki `.venv` ortamında CUDA destekli PyTorch kuruludur.
Başka bilgisayarda kurulum için
[PyTorch kurulum sayfasından](https://pytorch.org/get-started/locally/)
uygun GPU paketini seçin; `numpy`, `scipy` ve `wandb` de gereklidir.

## 10 seed ile ortalama ± standart sapma

PyCharm'da [`run_10_seeds.py`](run_10_seeds.py) dosyasını açıp **Run**
seçin. Varsayılan seed'ler **90–99**; her biri için `splits/` altındaki
sabit TSV, sıfırdan model ve optimizer kullanılır. `UP` için her koşuda
45 eğitim, 42.731 test merkezi vardır. İsterseniz
`SEED_START`, `RUNS`, `DATASET` ve diğer ayarları dosyanın başından
değiştirin. Öğrenme hızı veya batch boyutunu değiştirerek yeni bir deney
başlatırken `EXPERIMENT_NAME` için yeni bir ad girin.

Komut satırı karşılığı:

```powershell
.venv\Scripts\python -m pytorch.benchmark --method SSARN --dataset UP --seed-start 90 --runs 10 --k 5 --epochs 300 --device gpu --data data --output outputs/benchmark_ssarn_up_seeds90-99_k5_e300
.venv\Scripts\python -m pytorch.benchmark --method QMTN --dataset UP --seed-start 90 --runs 10 --k 5 --epochs 300 --device gpu --data data --output outputs/benchmark_qmtn_up_seeds90-99_k5_e300
```

Çıktı klasöründe `experiment.json`, seed başına `splits/seed_N.tsv`,
`runs/seed_N/attempt_001/` koşu dosyaları, `runs.tsv`,
`summary.json` ve `summary.tsv` bulunur. OA/AA/κ için aritmetik
ortalama ve **örnek standart sapması (ddof=1)** hem 0–1 hem yüzde
ölçeğinde saklanır. Aynı ayarlarla yeniden çalıştırıldığında bitmiş
seed'ler doğrulanıp atlanır; yarıda kalan seed için yeni attempt açılır.
`LOG_WANDB=True` yalnızca toplu özeti tek bir W&B koşusuna yollar;
seed bazındaki sayılar o koşunun tablosunda bulunur. Karşılaştırma için
ortalama OA/AA/κ önceki tek koşularla aynı `test/OA`, `test/AA`,
`test/kappa` ve yüzde alanlarına yazılır; standart sapmalar
`test/OA_std_percent`, `test/AA_std_percent` ve
`test/kappa_std_x100` alanlarında görünür.
Eğitim tamamlandıktan sonra W&B yüklemesini ayrıca tekrarlamak için
`.venv\Scripts\python -m pytorch.wandb_benchmark --output outputs/<deney_klasörü>`
komutunu kullanın. Yüklenen özetin bağlantısı kaydedilir; aynı deney
yeniden çalıştırıldığında mevcut W&B koşusu güncellenir.

Colab'da [`AG-Meta-HSI.ipynb`](notebooks/AG-Meta-HSI.ipynb)
kullanın. `METHOD` alanı kayıtlı yöntemler arasında seçim yapar; defterin
kodunu her yöntem için değiştirmeye gerek yoktur. GitHub'dan kodları çeker ve
çıktıları Drive'da tutar.
`RUN_TRAINING=True` iken `EXISTING_OUTPUT_DIR=""` boş kalabilir.
Yarıda kesilirse aynı seed/ayarlarla defteri yeniden çalıştırın.
Epoch, öğrenme hızı veya batch boyutunu değiştirerek yeni deney
başlatırken `EXPERIMENT_NAME` için yeni bir ad girin; aynı klasör yalnızca
aynı ayarlarla kaldığı yerden devam eder.
`RUN_TRAINING=False` yalnızca tamamlanmış bir deneyin klasörünü
`EXISTING_OUTPUT_DIR` olarak verip W&B'ye yüklemek içindir.

Yeni yöntem eklemek için yönteme ait çalıştırıcıyı yazıp
[`pytorch/methods.py`](pytorch/methods.py) dosyasına kaydedin. Çalıştırıcı
`run_baseline` ile aynı argümanları kabul etmeli ve benchmark'ın okuduğu
`config.json`, `metrics.json`, `split.tsv`, `training.tsv`,
`class_accuracy.tsv`, `confusion.tsv` ile checkpoint'i üretmelidir.
`source_files` listesi koşunun `config.json` dosyasındaki `source_sha256`
anahtarlarıyla eşleşmelidir. Yeni yöntem kaydedilince aynı defterde yalnız
`METHOD` değeri ve gerekiyorsa `METHOD_CONFIG_JSON` içindeki yöntem ayarları
değiştirilir.

Zhu ve arkadaşlarının *Pattern Recognition* 172 (2026) makalesinde
Tablo 2'nin **SSARN** sütunu UP için OA **83,49±3,46**,
AA **87,23±2,47**, κ×100 **78,74±4,25** raporlar.
Makale de sınıf başına 5 rastgele örnek, 9×9 yama ve 10 deneme
kullanır. Buradaki PyTorch SSARN bağımsız bir uygulamadır;
ön işleme, ağırlık başlatma ve optimizasyon ayrıntılarının makale
koduyla aynı olduğu doğrulanmadığı için 10 koşu özeti
**referans karşılaştırmasıdır**, birebir yeniden üretim değildir.

## QMTN / QLOML baseline

QMTN mevcut SSARN ağını kullanır; yalnızca eğitim yöntemi farklıdır.
Her meta görevde eğitim split'inden rastgele sınıflar seçilir. Sınıf başına
5 eğitim örneğinin bir kısmı destek, kalanı sorgu kümesidir. 16 görevlik
havuz bir kez kurulur; her epoch'ta görev sırası karıştırılır. Destek
kaybıyla asıl SSARN, SGD ile üç kez güncellenir. Uyarlanmış SSARN'da
hesaplanan sorgu kaybının gradyanı, ayrı ikiz ağın Adam güncellemesine
aktarılır; ikinci derece türev hesaplanmaz. İkiz ağın parametreleri sonraki
epoch başında SSARN'a aktarılır. Son epoch'taki **asıl SSARN** test edilir;
ikiz ağın durumu da checkpoint'te saklanır. Test pikselleri meta görevlere
girmez. `QMTN-DA` gürültü artırımı bu baseline'a dahil değildir.

Makalenin açık ayarları varsayılandır: UP için 5-way, IP/SA için 8-way,
epoch başına 16 görev, 3 destek güncellemesi, iç SGD öğrenme hızı 0.01,
dış Adam öğrenme hızı 0.002 ve 300 epoch. Makale görev başına destek
örneği sayısını 1–4 arasında inceler ama Tablo 2 için hangisini
kullandığını açıkça belirtmez. Bu uygulama **3-shot** varsayar;
`--method-config '{"support_shots":2}'` ile değiştirilebilir. Aynı JSON
içinde `ways`, `tasks_per_epoch`, `inner_steps`, `inner_lr` ayarlanabilir.
`--lr` dış Adam hızıdır; yöntem belirtilince varsayılan otomatik seçilir.
`batch_size` QMTN'de arayüz uyumluluğu için tutulur; görevlerin destek ve
sorgu kümeleri tam olarak işlenir.

Makalenin UP Tablo 2 **QMTN** referansı OA **87,24±4,18**,
AA **91,69±1,47**, κ×100 **83,70±5,13** değerleridir. Makaledeki
görev örnekleme ve ikiz ağın buffer aktarımı ayrıntıları tam belirtilmediği
için bu bağımsız uygulama birebir yeniden üretim iddiası taşımaz.

## Veri ve sabit ayrım

Veri dosyaları `data/` klasöründe olmalıdır. Kaynak:
[UPV/EHU hyperspectral scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

| Dataset | Veri küpü | Etiket dosyası | Bant / sınıf |
|---|---|---|---|
| `UP` | `PaviaU.mat` | `PaviaU_gt.mat` | 103 / 9 |
| `SA` | `Salinas_corrected.mat` | `Salinas_gt.mat` | 204 / 16 |
| `IP` | `Indian_pines_corrected.mat` | `Indian_pines_gt.mat` | 200 / 16 |

Resmi UP split seti `splits/up_seed90_k5.tsv`–`up_seed99_k5.tsv` dosyalarıdır;
her biri **45 eğitim** ve **42.731 test** merkezini sabitler. PyCharm ve Colab
aynı dosyaları kullanır. Eski Julia seed 93 split'i
`splits/legacy_up_seed93_julia_k5.tsv` adıyla ayrı tutulur. Başka bir
`--split` yolu verilebilir; dosya bulunmazsa eğitim hata verir. Benchmark
da eksik split için hata verir, dosya üretmez. Arka plan sınıfı 0
değerlendirmeye girmez. Komşu eğitim/test
yamaları uzamsal olarak örtüşebilir.

Yamalar `9×9`, sınırda sıfır padding ve ham `Float32` değerlerle
hazırlanır; normalizasyon, PCA ve veri artırma uygulanmaz. Üç spektral
`Conv3d` bloğunu spektral residual, spektral daraltma, uzamsal residual
ve Q/K/V attention izler. Başlık global ortalama havuzlama ve Dense
katmanından oluşur. Eğitimde Adam ve cross-entropy kullanılır.
Doğrulama kümesi, erken durdurma veya test sonucuyla checkpoint seçimi yoktur.
Son epoch checkpoint'i kaydedildikten sonra test kümesi bir kez değerlendirilir.

## Çıktılar ve W&B

Her eğitim koşusu şu dosyaları üretir: `config.json`, `split.tsv`,
`training.tsv`, `checkpoint.pt`, `metrics.json`,
`class_accuracy.tsv`, `confusion.tsv`. OA, AA ve sınıf doğrulukları
0–1 aralığındadır; kappa ölçeklenmez. Ayrım, veri ve kaynak kodu için
SHA256 değerleri config içinde saklanır. Mevcut çıktı klasörünün üzerine
yazılmaz.

Tamamlanmış bir koşunun W&B kaydını sonradan yapmak için:

```powershell
.venv\Scripts\python -m pytorch.wandb_log --output outputs/<koşu_klasörü> --project ag-meta-hsi
```

W&B'ye epoch başına eğitim kaybı, OA/AA/kappa, sınıf doğrulukları ve
confusion tablosu yazılır. Veri dosyaları ve checkpoint yüklenmez.

## Python Colab defteri

[`notebooks/SSARN_colab.ipynb`](notebooks/SSARN_colab.ipynb) dosyasını
Colab'da Python 3 ve GPU çalışma ortamıyla açın. Defter
[GitHub deposunu](https://github.com/aligunduz/AG-Meta-HSI) klonlar,
`METHOD`/`DATASET` seçimlerine göre eğitir, koşu dosyalarını
Google Drive'a ve ölçütleri W&B'ye kaydeder. Defterin kaynağını
değiştirirseniz `python scripts/build_colab_assets.py` ile yeniden üretin.
Yerel değişiklikler GitHub'a siz commit ve push edene kadar Colab
tarafından görülemez.

## Doğrulama

`python -m unittest pytorch.test_ssarn pytorch.test_qmtn` modelin UP/SA/IP çıkış boyutlarını
ve sentetik veriyle bir epoch'luk eğitimin dosyalarını, iki farklı
split'i, ortalama/standart sapmayı ve kesinti sonrası devamı kontrol eder.
Gerçek UP verisinde CUDA ileri geçişi, tek optimizer güncellemesi ve
1 epoch'luk tam değerlendirme doğrulandı. Bu kısa koşunun ölçütleri
nihai 300 epoch deney sonucu değildir.
