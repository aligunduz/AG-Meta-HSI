# AG-Meta-HSI

Julia/Lux.jl ile Pavia University (UP) için **9 sınıflı supervised SSARN baseline**.
QMTN, QLOML, twin network, episodic eğitim ve veri artırma uygulanmamıştır.

## Kurulum ve çalıştırma

Julia 1.11 veya üzeri; geliştirme ortamı Julia 1.13.1, Lux 1.31.4.
Komutları repo kökünde çalıştırın:

```powershell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl
```

`data/PaviaU.mat` (`paviaU`) ve `data/PaviaU_gt.mat` (`paviaU_gt`) gereklidir.
Veri kaynağı: [UPV/EHU](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).
**Mevcut `outputs/up_split_seed93.tsv` dosyasını koruyun; yeniden üretmeyin.**
`prepare_up.jl` önceki veri hazırlama aracı olarak durur, eğitim tarafından çağrılmaz.

Yalnızca mevcut split ve bir gerçek eğitim yamasında ileri geçiş kontrolü:

```powershell
julia --project=. scripts/ssarn_up.jl --check
```

Argümansız çalıştırma da kontrol modudur. Epoch, optimizer güncellemesi,
test tahmini veya deney çıktı dosyası üretmez. `9×9×103` yama Lux için
`9×9×103×1×1` biçimine getirilir; `9×1` sonlu logit üretmesi doğrulanır.
Aynı kontrol eğitim komutunda da ilk güncellemeden önce çalışır.

### Tek adımlık smoke kontrolü

```powershell
julia --project=. scripts/ssarn_up.jl --smoke
```

Kayıtlı 45 eğitim pikselinden seed ile seçilen 15 örnekte bir forward/backward
ve **tam olarak bir Adam güncellemesi** yapar. Aynı batch'te güncelleme sonrası
ikinci forward ile loss kontrol edilir. İki loss'un ve gradyanların sonlu
olması, en az bir parametrenin değişmesi zorunludur; loss düşüşü zorunlu değildir.
Her iki loss trainmode BatchNorm ile hesaplanır. Test yamaları/tahminleri,
OA/AA, checkpoint, sonuç dizini veya eğitim dosyası oluşturulmaz.
`--batch-size` smoke için 15 olmalıdır. `--smoke`, `--check`, `--train`
birbirini dışlar; birlikte kullanımları hatadır. Varsayılan mod hâlâ `--check`.

VS Code Run and Debug listesinden **Julia: SSARN smoke** seçip **Ctrl+F5**
ile çalıştırabilirsiniz. Bu seçenek yerel, Git tarafından yok sayılan
`.vscode/launch.json` dosyasına eklenmiştir.

Smoke argüman testleri ve gerçek UP CLI kontrolü (bir Adam adımı):

```powershell
julia --project=. test/smoke.jl
```

Eğitimi **siz başlatmak istediğinizde**:

```powershell
julia --project=. scripts/ssarn_up.jl --train --epochs 300 --lr 0.001 --seed 93 --batch-size 15 --test-batch-size 32 --output outputs/ssarn_seed93
```

`--data`, `--split`, `--output` yolları değiştirilebilir. Split metadata'sı
seed=93/k=5, sınıflar 1:9 ve bütün etiketli piksellerin tam kapsanması doğrulanır.
Başka geçerli bir split yolu vermek farklı bir deneydir; dosyanın SHA256 değeri
kaydedilir. Var olan çıktı dizinine yazmak reddedilir. CPU/Float32 kullanılır;
GPU desteği henüz yoktur. İlk çalıştırmada Julia derlemesi zaman alabilir.

### Colab, veri kümesi seçimi ve W&B

[`notebooks/SSARN_colab.ipynb`](notebooks/SSARN_colab.ipynb) dosyasını Google
Colab'da açın. Defter kodu doğrudan
[`aligunduz/AG-Meta-HSI`](https://github.com/aligunduz/AG-Meta-HSI) deposundan
klonlar; aynı Colab oturumunda yeniden çalıştırılırsa `git pull --ff-only`
ile günceller. Defter kaynağı değişince `python scripts/build_colab_assets.py`
komutuyla `.ipynb` dosyasını yeniden üretin.

Defterde `BASELINE="SSARN"`, `DATASET="UP"`, `"SA"` veya `"IP"` ayarlayın.
Şimdilik yalnızca SSARN uygulanmıştır; başka baseline adı açık hata verir.
Veri dosyalarını Google Drive'daki `AG-Meta-HSI/data` klasörüne yerleştirin
veya `DATA_SOURCE="upload"` seçin:

| Seçim | Veri küpü | Etiket dosyası | Bant / sınıf |
|---|---|---|---|
| `UP` | `PaviaU.mat` | `PaviaU_gt.mat` | 103 / 9 |
| `SA` | `Salinas_corrected.mat` | `Salinas_gt.mat` | 204 / 16 |
| `IP` | `Indian_pines_corrected.mat` | `Indian_pines_gt.mat` | 200 / 16 |

Kaynak: [UPV/EHU veri sayfası](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).
İlk koşuda her sınıftan `k=5` piksel için seeded split oluşturulur; mevcut
split asla değiştirilmez. Split kopyası, SHA256 ve veri SHA256 değerleri her
koşunun çıktı klasöründe saklanır. Çıktı klasörü Drive'da benzersiz bir ad
alır. Defter koşu **başarıyla tamamlandıktan sonra** W&B'ye giriş ister ve
epoch başına eğitim kaybını, OA/AA/kappa'yı, sınıf doğruluklarını ve confusion
tablosunu kaydeder. `.mat` verileri ile checkpoint W&B'ye yüklenmez.
Var olan bir koşuyu yalnızca W&B'ye göndermek için çıktı klasörünü Drive'a
kopyalayıp `RUN_TRAINING=false` ve `EXISTING_OUTPUT_DIR` değerini ayarlayın.

Julia CLI ile aynı seçimi yapmak için örnek:

```powershell
julia --project=. scripts/run_baseline.jl --baseline SSARN --dataset SA --train --data data --seed 93 --k 5 --epochs 300 --lr 0.001 --batch-size 15 --test-batch-size 32 --output outputs/ssarn_sa_seed93
```

Bu çoklu veri kümesi yolu SSARN supervised baseline'ıdır; QMTN meta-öğrenme
veya GPU eğitimi içermez. Makaledeki on koşulu deneyle karşılaştırma için
farklı seed'leri ayrı ayrı çalıştırın ve dağılımı ayrıca hesaplayın.

## Sabit veri ve değerlendirme protokolü

- Sınıf başına 5 eğitim merkezi: toplam 45. Geri kalan bütün etiketli merkezler
  test kümesidir (standart UP verisinde 42.731). Arka plan sınıf 0 dışlanır.
- Mevcut `HSIScene`, `load_pixel_split`, `validate_pixel_split` ve
  `extract_patch` kullanılır. Tam 103 bant, 9×9 yama, sınırda sıfır padding.
- Ham Float32 değerleri korunur: normalizasyon, PCA, bant seçimi yoktur.
  Bu ön işleme tercihinin makaleyle birebir aynı olduğu iddia edilmez.
- Merkezler ayrıdır; komşu train/test yamaları örtüşebilir. Mekânsal tampon yoktur.
- Adam, sabit lr=0.001, 300 epoch, batch=15, seed=93 bu supervised baseline
  için **önceden seçilmiş başlangıç ayarlarıdır**. QLOML iç/dış döngü öğrenme
  oranları supervised baseline'a aktarılmamıştır.
- Eğitim yalnızca 45 eğitim etiketini kullanır. Her epoch bu 45 örneğin bir
  geçişidir; sıra seeded MersenneTwister ile karıştırılır. Validation,
  erken durdurma, scheduler veya en iyi test checkpoint'i seçimi yoktur.
- Son epoch checkpoint'i kaydedildikten sonra tüm test pikselleri bir kez,
  mini-batch halinde değerlendirilir. BatchNorm `testmode` kullanır; testten
  istatistik öğrenmez. Hiperparametreler test sonuçlarına göre seçilmemelidir.

## Şekil 2'den SSARN katmanları

Kaynak: J. Zhu ve diğerleri, *A query-driven twin network framework with
optimization-based meta-learning for few-shot hyperspectral image classification*,
Pattern Recognition 172 (2026), 112331, kullanıcının
sağladığı QLOML PDF'si, s. 4, Şekil 2–3, Bölüm 3.2, Denklem (5)–(8).
PDF araştırma kaynağıdır; bu kod resmi uygulama veya birebir yeniden üretim değildir.

Makale eksenleri `(spektral,H,W)`, Lux eksenleri `(H,W,spektral,kanal,batch)`.
Tablodaki boyutlar batch ekseni hariç Lux sırasındadır.

| Bölüm | İşlem | Çıktı |
|---|---|---|
| Girdi | 9×9×103 yama, tek giriş kanalı | 9×9×103×1 |
| Spektral giriş 1 | Conv3D 1×1×7, stride 1×1×2, 1→32; BN; LeakyReLU | 9×9×49×32 |
| Spektral giriş 2 | Conv3D 1×1×7, stride 1×1×2, 32→32; BN; LeakyReLU | 9×9×22×32 |
| Spektral giriş 3 | Conv3D 1×1×7, stride 1×1×2, 32→32; BN; LeakyReLU | 9×9×8×32 |
| Spektral residual | İki F=Conv3D 1×1×7 → LeakyReLU → BN; `x + F2(x + F1(x))` | 9×9×8×32 |
| Spektral daraltma | LeakyReLU → Conv3D 3×3×8, 32→32 → LeakyReLU → BN; spektral ekseni kaldır | 7×7×32 |
| Uzamsal residual | F=Conv2D 3×3 → LeakyReLU → BN; `x + F(x) + Att(F(x))` | 7×7×32 |
| Başlık | LeakyReLU → global ortalama havuzlama → Dense 32→9 | 9 logit |

Attention: üç ayrı 1×1 Conv2D ile 32 kanallı Q/K/V, her biri `32×49`;
`A=softmax(QᵀK, dims=2)` ve `V*A`, sonra 7×7×32'ye dönüş.
Softmax son uzamsal eksen üzerindedir (Denklem 7). Transformer ölçekleme
faktörü, kanal daraltma veya öğrenilebilir attention çarpanı eklenmez.
Son katman softmax içermez; eğitimde kararlı log-softmax cross-entropy,
tahminde argmax kullanılır. [Lux API](https://lux.csail.mit.edu/stable/api/Lux/layers).

### Açık varsayımlar ve belirsizlikler

1. İlk üç spektral konvolüsyonun padding'i açık verilmediği için **valid/0**
   seçildi. Böylece 103→49→22→8 ve B′=8 olur. Daraltma kernel'i buna bağlıdır.
2. Spektral residual padding=3 metinde verilir. Boyutu korumak için stride=1
   seçildi; girişteki stride=2 burada devam ettirilmez.
3. Şekil 2'de ilk üç blok Conv→BN→LeakyReLU, residual bloklar
   Conv→LeakyReLU→BN sırasındadır; kod bu ayrımı korur. Residual toplamlarının
   yerleri Denklem 5 ve 8 ile belirlenir.
4. Daraltmada padding=0, stride=1; uzamsal Conv2D'de padding=1, stride=1.
   Bunlar metindeki H−2/W−2 ve boyut koruma ile uyumludur.
5. PDF'de açık olmayan LeakyReLU eğimi **0.01**, BN epsilon **1e-5**, momentum
   **0.1**, affine scale/bias ve çalışan istatistikler kullanılır.
   Conv/Dense bias açıktır; ağırlık/bias başlatma Lux sürümünün varsayılanıdır.
   Bunların yazarların koduyla aynı olduğu ileri sürülmez.
6. Dokuz çıkış tüm UP sınıfları içindir; 5-way episodic sınıflandırıcı kurulmaz.
   Dropout ve ekstra aktivasyon/katman eklenmez. Şekildeki ortalama havuzlama,
   metindeki 32×1 özellik çıktısına göre global ortalama olarak uygulanır.

## Kaydedilen dosyalar

`--train` tamamlandığında seçilen çıktı dizininde:

- `config.toml`: model kimliği, seed, epoch, lr, batch boyutları, protokol,
  veri/split/kaynak SHA256 değerleri, Julia sürümü ve başlangıç zamanı.
- `split.tsv`, `Project.toml`, `Manifest.toml`: split ve ortam kopyaları.
- `training.tsv`: her epoch eğitim cross-entropy'si; test eğrisi yoktur.
- `checkpoint.jls`: son epoch parametreleri, BN durumu, Adam durumu, RNG,
  epoch ve config. Julia Serialization kullanır; aynı Julia/bağımlılık
  ortamında, yalnızca güvenilen checkpoint'leri yükleyin.
- `metrics.toml`: OA, AA, kappa, seed, epoch, lr ve checkpoint yolu.
- `class_accuracy.tsv`: sınıf başına destek, doğru sayısı ve accuracy.
- `confusion.tsv`: satır=gerçek sınıf, sütun=tahmin edilen sınıf.

OA=tüm doğru/test toplamı; sınıf accuracy=doğru/sınıf desteği;
AA=dokuz sınıf accuracy'sinin eşit ağırlıklı ortalaması.
Kappa=`(OA-pe)/(1-pe)`; `pe` gerçek/tahmin marjinallerinden hesaplanır.
Accuracy değerleri **0–1 aralığında**, kappa ölçeklenmeden saklanır.

Checkpoint ile tek yama tahmini:

```julia
using AGMetaHSI, Lux, Serialization
saved = deserialize("outputs/ssarn_seed93/checkpoint.jls")
scene = load_pavia_university("data")
split = load_pixel_split("outputs/ssarn_seed93/split.tsv")
x = patch_batch(scene, split.test[1:1])
logits, _ = ssarn()(x, saved.ps, Lux.testmode(saved.st))
predicted_class = argmax(vec(logits))
```

## Kod ve doğrulama

- `src/AGMetaHSI.jl`: mevcut UP veri modülü ve yeni modüllerin yüklenmesi.
- `src/ssarn.jl`: Lux backbone, attention, yama batch'i ve çıktı kontrolü.
- `src/supervised.jl`: sabit split doğrulama, eğitim, checkpoint/ölçütler.
- `scripts/ssarn_up.jl`: varsayılan kontrol modu ve açık eğitim komutu.
- `test/runtests.jl`: veri testleri, ara boyutlar, attention matris hesabı,
  sentetik batch üzerinde gradyan kontrolü (optimizer güncellemesi yok),
  checkpoint okuma/yazma ve bilinen confusion matrisleriyle ölçüt kontrolü.

Bu değişiklik hazırlanırken UP eğitimi başlatılmamıştır; OA/AA/kappa sonucu
ve eğitilmiş checkpoint henüz yoktur.

Doğrulama: 39/39 test geçti. Gerçek UP yamasında `(9,9,103,1,1) → (9,1)`
sonlu çıktı kontrolü geçti; split 45 eğitim / 42.731 test pikseli içeriyor.
Split SHA256 (kontrol öncesi/sonrası aynı):
`c3a2d2aafa38885a62ef45a45e14c12351eca023b9c8e4c233ffcf594ae6e807`.
Tam eğitim ve son test değerlendirmesi çalıştırılmadığından bunların uçtan uca
çalışma süresi ve sınıflandırma başarımı henüz doğrulanmamıştır.
