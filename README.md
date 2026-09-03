<div align="center">

# 🫀 EKG-Tarayıcı: 12-Derivasyonlu EKG Sınıflandırma ve Karar Destek Sistemi
### TEKNOFEST 2026 — Sağlıkta Yapay Zeka Yarışması (Lise Kategorisi)
#### **Takım:** Devre181 | **Takım ID:** #987840 | **Başvuru ID:** #5218603

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TEKNOFEST](https://img.shields.io/badge/TEKNOFEST-2026-red.svg)](https://www.teknofest.org/)

**[🇹🇷 Türkçe](README.md) | [🇬🇧 English](README_EN.md)**

<p align="center">
  <b>12 derivasyonlu EKG sinyallerinden aritmileri ve iletim bozukluklarını tespit eden, tanımadığı patolojileri güvenle reddedebilen ("Safe-Fail / Unknown"), açıklanabilir hibrit derin öğrenme mimarili klinik karar destek sistemi.</b>
</p>

</div>

---

## 📌 İçindekiler
- [Proje Hakkında](#-proje-hakkında)
- [Takım Şeması](#-takım-şeması)
- [Model Performansı ve Sonuçlar](#-model-performansı-ve-sonuçlar)
- [Hedef Sınıflar ve Unknown Mantığı](#-hedef-sınıflar-ve-unknown-mantığı)
- [Model Mimarisi](#-model-mimarisi)
- [Alternatiflerin Elenme Gerekçeleri](#-alternatiflerin-elenme-gerekçeleri)
- [Kullanılan Veri Setleri ve Ön İşleme](#-kullanılan-veri-setleri-ve-ön-i̇şleme)
- [Veri Dengesizliği ve Veri Artırma](#-veri-dengesizliği-ve-veri-artırma)
- [Veri Bölme ve Deney Protokolü](#-veri-bölme-ve-deney-protokolü)
- [Kayıp Fonksiyonu (Asymmetric Loss)](#-kayıp-fonksiyonu-asymmetric-loss)
- [Karar Mekanizması ve Kalibrasyon](#-karar-mekanizması-ve-kalibrasyon)
- [Hiperparametreler ve Eğitim Ayarları](#-hiperparametreler-ve-eğitim-ayarları)
- [Teknik Evrim ve Mühendislik Kararları](#-teknik-evrim-ve-mühendislik-kararları)
- [Açıklanabilirlik (Explainability)](#-açıklanabilirlik-explainability)
- [Performans ve Donanım Gereksinimleri](#-performans-ve-donanım-gereksinimleri)
- [Proje Dizin Yapısı](#-proje-dizin-yapısı)
- [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
- [Referanslar ve Literatür](#-referanslar-ve-literatür)
- [Lisans](#-lisans)

---

## 📖 Proje Hakkında

**EKG-Tarayıcı**, TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması kapsamında **Devre181** takımı tarafından geliştirilmiş uçtan uca bir klinik karar destek sistemidir.

Geleneksel derin öğrenme tabanlı EKG modelleri sinyalleri 2D spektrogram veya görsel formatlara çevirerek yüksek parametre yükü ve gecikme oluşturur. Bu çalışmada, doğrudan **1 boyutlu zaman serisi (1D Time Series)** üzerinde çalışan **hibrit bir mimari (ResNet-1D + CBAM + Lead-Transformer)** geliştirilmiştir. 

Sistem; yüksek sınıflandırma başarısının yanı sıra, hastane koşullarında veya taşınabilir EKG cihazlarında (edge devices) **milisaniyeler düzeyinde çıkarım (inference)** yapabilmekte ve klinik güvenilirlik için belirsiz veya dağılım dışı (OOD) durumlarda **Safe-Fail (Unknown)** kararı üretebilmektedir.

---

## 👥 Takım Şeması

| Rol | Sorumluluk Alanı |
|:---|:---|
| **Danışman (Akademik Koordinatör)** | Takımın genel işleyişini denetleme, klinik bilgi akışı, validasyon ve test stratejilerinin bilimsel standartlara uyumu. |
| **Takım Kaptanı** | Eğitim sonuçları analizi, mantıksal hata giderme, Macro F1 optimizasyonu ve aşırı öğrenme (overfitting) yönetimi. |
| **1. Üye** | Uluslararası literatür taraması, teorik altyapı, 12 derivasyonlu EKG sinyallerinin kardiyak ritim/iletim bozukluğu klinik çözümlemesi. |
| **2. Üye** | Derin öğrenme mimari analizi, veri kalitesi kontrol protokolleri, eksik/gürültülü verilerin elenmesi ve bilimsel uyum denetimi. |
| **3. Üye** | Sinyal ön işleme, pencereleme, veri artırma boru hatları entegrasyonu, ilk model eğitim derlemeleri ve performans takibi. |
| **4. Üye** | Proje yazılım mimarisi, modüler kod yapısı, dış doğrulama test ortamı kararlılığı ve GPU/bellek optimizasyonları. |

---

## 📊 Model Performansı ve Sonuçlar

Modelin doğrulama kümesinde elde ettiği genel metrikler:

| Metrik | Skor |
|:---|:---:|
| **Genel Doğruluk (Accuracy)** | **%85.3** |
| **Macro F1-Score** | **0.768** |
| **Macro Precision** | **0.800** |
| **Macro Recall** | **0.800** |

### Sınıf Bazlı Performans Dağılımı

| Sınıf | Tanı | Precision | Recall | F1-Score |
|:---:|:---|:---:|:---:|:---:|
| **NORMAL** | Normal Sinüs Ritmi | 0.92 | 0.92 | **0.92** |
| **AFIB** | Atriyal Fibrilasyon | 0.84 | 0.81 | **0.82** |
| **AFL** | Atriyal Flutter | 0.83 | 0.81 | **0.82** |
| **RBBB** | Sağ Dal Bloğu (Right Bundle Branch Block) | 0.72 | 0.73 | **0.73** |
| **LBBB** | Sol Dal Bloğu (Left Bundle Branch Block) | 0.69 | 0.74 | **0.71** |
| **Ortalama** | **Macro Average** | **0.80** | **0.80** | **0.768** |

### Karışıklık Matrisi (Confusion Matrix)
```
Gerçek \ Tahmin     Normal    AFIB     AFL     RBBB    LBBB
Normal                458      12       6       14      10
AFIB                   18     203      22        4       3
AFL                     9      24     162        3       2
RBBB                    8       2       3      132      35
LBBB                    7       1       2       29     111
```
*Hata analizi incelendiğinde, nadir hataların morfolojik benzerliğe sahip LBBB–RBBB ve ritim özelliği yakın olan AFIB–AFL arasında gerçekleştiği saptanmıştır.*

---

## 🎯 Hedef Sınıflar ve Unknown Mantığı

Sistem, 10 saniyelik (2500 örnek @ 250 Hz) 12 derivasyonlu EKG kayıtlarını 5 ana sınıfa ayırır:
`NORMAL`, `AFIB`, `AFL`, `RBBB`, `LBBB`.

### 🛡️ Safe-Fail / Unknown (Bilinmeyen) Mekanizması
Klinik pratikte bir modelin tanımadığı bir patolojiyi (örn. Akut Miyokard Enfarktüsü - MI, WPW sendromu vb.) zorla 5 sınıftan birine ataması hastaya doğrudan zarar verebilir.
- **Negatif Çıpa (Negative Anchoring):** Eğitim ve ön eğitim aşamasında hedef 5 sınıf dışındaki kayıtlara hedef vektör olarak `[0, 0, 0, 0, 0]` atanmıştır. Model, tanımadığı morfolojilerde aktivasyonları sıfıra baskılamayı öğrenir.
- **OOD Doğrulaması:** Validasyon aşamasında hedef sınıfları içermeyen **4.000 kayıt** (3.000 ön eğitim + 1.000 fine-tuning) sisteme verilerek modelin düşük güven üretme yeteneği doğrulanmıştır.
- **Karar Kuralı:** Güven eşiğini aşamayan veya sınıflar arası skor marjı yetersiz olan sinyaller **Unknown** statüsüne çekilerek yanlış pozitif tanıların önüne geçilir.

---

## 🏗️ Model Mimarisi

Model, uçtan uca eğitilebilir üç aşamalı hibrit bir yapıdan oluşur:

```
Giriş: [Batch, 12, 2500] (12 Derivasyon × 2500 Zaman Noktası)
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  ResNet-1D (4 Residual Blok)                           │
│  - Her derivasyon bağımsız konvolüsyondan geçer        │
│  - Kernel: 7, Stride: 2, MaxPool, Residual bağlantılar │
│  - Çıkış: [Batch * 12, 256, T]                         │
└────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  CBAM 1D (Convolutional Block Attention Module)        │
│  - Kanal Dikkat: Tanısal açıdan kritik derivasyonları  │
│    öne çıkarır, artefaktlı/ölü kanalları bastırır      │
│  - Zaman Dikkat: QRS/P/T dalga aralıklarına odaklanır  │
└────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  Global Average Pooling (GAP)                          │
│  - Zaman boyutu (T) sıkıştırılır                       │
│  - Çıkış: [Batch, 12, 256]                             │
└────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  Pre-LN Transformer Encoder (2 Katman, 4 Başlık)       │
│  - Girişe öğrenilebilir CLS Token eklenir              │
│  - 12 derivasyonun uzaysal kalp ekseni korelasyonunu   │
│    ve elektriksel yayılım ilişkisini modeller          │
│  - Çıkış (CLS): [Batch, 256]                           │
└────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  Sınıflandırıcı Başlık (Classification Head)           │
│  - LayerNorm + Linear(256 -> 5)                        │
│  - Bağımsız Sigmoid çıkışları (Multi-label)           │
└────────────────────────────────────────────────────────┘
```

> **Neden Softmax Değil Sigmoid?**  
> Bir hastada aynı anda hem ritim bozukluğu (örn. AFIB) hem de iletim bozukluğu (örn. RBBB) bulunabilir. Softmax olasılıkları toplamsal olarak 1'e zorlayarak klinik gerçekliği bozar; bu nedenle bağımsız çoklu etiket (multi-label) öngören **Sigmoid** tercih edilmiştir.

---

## 🚫 Alternatiflerin Elenme Gerekçeleri

| Alternatif Yaklaşım | Elenme Nedeni |
|:---|:---|
| **Klasik Makine Öğrenmesi (SVM, Random Forest)** | Ham EKG'den manuel öznitelik çıkarımı gerektirmesi; P-QRS-T morfolojilerinin klinik çeşitliliğini ve derivasyonlar arası elektriksel ekseni modellemede yetersiz kalması. |
| **Saf Transformer (Pure Transformer)** | Düşük tümevarımsal yanlılık (inductive bias) nedeniyle sınırlı veri setlerinde yakınsama güçlüğü çekmesi ve aşırı öğrenme (overfitting) riskinin yüksek olması. |
| **Softmax Tabanlı Mimari** | Sınıfları birbirini dışlayan (mutually exclusive) kabul etmesi; oysa klinik pratikte aynı hastada hem ritim (AFIB) hem iletim (RBBB) bozukluğunun aynı anda bulunabilmesi. |
| **WeightedRandomSampler** | Verinin doğal öncül (prior) olasılık dağılımını bozması, modelin dağılım dışı (OOD) verilerde aşırı güvenli (overconfident) yanlış tahminler üretmesine yol açması. |

---

## 🔬 Kullanılan Veri Setleri ve Ön İşleme

### Veri Seti Dağılımı (PDR Tablosu)

| Veri Seti | Kaynak | Toplam Kayıt | Frekans / Süre | Kullanım Amacı | Format |
|:---|:---|:---:|:---:|:---:|:---|
| **TEKNOFEST** | TEKNOFEST | 5.000 | 500Hz / 10sn | Fine-tuning | .mat / .dat, .hea |
| **PTB-XL** | PhysioNet | 21.799 | 500Hz / 10sn | Pre-training | .dat, .hea |
| **ECG-Arrhythmia** | PhysioNet (Chapman) | 45.152 | 500Hz / 10sn | Pre-training | .mat, .hea |
| **G12EC** | PhysioNet Challenge (Georgia) | 10.344 | 500Hz / 5-10sn | Pre-training | .mat, .hea |

*Dört veri setinden hedef 5 sınıfa ait toplam **41.601 kayıt** derlenmiştir (NORMAL: 20.388, AFIB: 4.864, AFL: 9.319, RBBB: 4.946, LBBB: 2.084).*

### Sinyal Ön İşleme Hattı
1. **Bandpass Filtreleme (0.5 – 45 Hz):** Düşük frekanslı solunum salınımlarını ve yüksek frekanslı miyogram (kas) gürültülerini keserken klinik QRS morfolojisini korur.
2. **Notch Filtreleme (50 Hz):** Şebeke hattından kaynaklanan elektrik parazitlerini temizler.
3. **Derivasyon Bazlı Z-Score Normalizasyonu:** Her kanal sıfır ortalama ve birim varyansa getirilir; ölü kanallar ($std < 10^{-6}$) korunarak anomaliler engellenir.
4. **Veri Kalitesi ve Boyut Standardizasyonu:** 
   - 250 Hz örnekleme hızına yeniden örnekleme (10 saniye $\rightarrow$ 2500 örnek).
   - Eksik derivasyon içeren veya 8 saniyenin altındaki kayıtlar çıkarılmıştır.
   - Mutlak genliği $> 10\text{ mV}$ aşan veya örneklerinin $>\%5$'i ADC sınırlarında doymuş (clipping) sinyaller elenmiştir.
   - Fingerprinting ile 100 adet yinelenen (duplicate) kayıt temizlenmiştir.
5. **HDF5 RAM Önbellekleme:** Tüm sinyaller `float16` tensörleri halinde HDF5 dosyasında saklanır. PyTorch DataLoader worker'ları thread-local dosya işleyicileriyle doğrudan RAM'den okuma yapar; disk darboğazı giderilerek epoch süresi **12 dakikadan 2 dakikanın altına** indirilmiştir.

---

## ⚖️ Veri Dengesizliği ve Veri Artırma

"Normal" sınıfının tüm verilerde %49.01 oranında baskın olması nedeniyle modelin çoğunluk sınıfa yönelmesini önlemek için şu adımlar uygulanmıştır:

### Veri Dağılımı ve Ağırlık Tablosu

| Sınıf | Dahil Edilen Veri | Veri Artırma Sonucu | Sınıf Ağırlığı (Class Weight) |
|:---:|:---:|:---:|:---:|
| **NORMAL** | 18.000 | 18.000 | **0.45** |
| **AFIB** | 4.800 | 9.600 *(x2)* | **1.00** |
| **AFL** | 8.000 | 8.000 | **1.20** |
| **RBBB** | 4.900 | 9.800 *(x2)* | **0.85** |
| **LBBB** | 2.000 | 4.000 *(x2)* | **2.90** |
| **Toplam** | **37.700** | **49.400** | — |

*LBBB sınıfının teorik ters-frekans ağırlığı 4.7 iken, log-ölçekleme ile agresif katsayı 2.9 seviyesine dengelenmiştir.*

### Veri Artırma (Yalnızca Train Kümesi / Azınlık Sınıfları)
- **Gaussian Noise:** Sensör paraziti simülasyonu.
- **Baseline Wander:** Solunum kaynaklı 0.1–0.5 Hz taban çizgisi salınımı.
- **Genlik Ölçekleme:** Genel morfolojiyi bozmayacak 0.95–1.05 oranlarında genlik çeşitlendirmesi.
- **Time Shift:** ±300 ms aralığında zaman kaydırma.

---

## 🧪 Veri Bölme ve Deney Protokolü

- **Hasta Bazlı Ayrım (GroupShuffleSplit):** Aynı hastaya ait farklı kayıtların hem eğitim hem test kümesine düşerek veri sızıntısı (data leakage) yaratması engellenmiştir.
- **Ön Eğitim Bölümü:** PTB-XL, Chapman ve Georgia verileri **%80 Eğitim, %20 Doğrulama** olarak ayrılmıştır.
- **İnce Ayar (Fine-Tuning) Bölümü:** TEKNOFEST verisi **%70 Eğitim, %15 Doğrulama, %15 Test** olarak ayrılmıştır.
- **Safe-Fail / Unknown Validasyonu:** Hedef dışı 4.000 kayıt (3.000 ön eğitim + 1.000 fine-tuning) kullanılarak modelin bilinmeyen hastalıklarda doğru reddetme davranışı test edilmiştir.

---

## 📐 Kayıp Fonksiyonu (Asymmetric Loss)

Veri setlerindeki sınıf dengesizliğini ve tıbbi hataların asimetrik riskini yönetmek için **Asymmetric Loss (ASL)** kullanılmıştır:

$$\mathcal{L} = \sum_{k=1}^{K} - y_k (1 - p_k)^{\gamma_{pos}} \log(p_k) - (1 - y_k) (p_{m,k})^{\gamma_{neg}} \log(1 - p_{m,k})$$

- $\gamma_{pos} = 0$: Doğru pozitif tahminlerde kayıp baskılanmaz, aşırı güven (overconfidence) önlenir.
- $\gamma_{neg} = 2$: Yanlış negatiflere (özellikle LBBB gibi hayati risk taşıyan sınıfların kaçırılmasına) ağır ceza verilir.
- **Asymmetric Clipping ($clip = 0.05$):** Kolay negatifler tamamen sıfırlanarak gradyan zor örneklere odaklanır.

---

## 🚦 Karar Mekanizması ve Kalibrasyon

1. **Temperature Scaling (Sıcaklık Kalibrasyonu):**
   Validasyon kümesinde logitler $z / T$ formülüyle kalibre edilir; modelin %90 güven dediği bir tahminin gerçekte de %90 doğrulukta olması sağlanır.
2. **Class-Specific Thresholds:**
   0.50–0.95 aralığında 0.01 adımlarla grid search yapılarak her hastalık için F1 skorunu maksimize eden sınıfa özel eşikler optimize edilmiştir.
3. **Margin Kontrolü ve Safe-Fail Karar Kuralı:**
   ```
   En yüksek olasılık < Güven Eşiği          ───► [ UNKNOWN ]
   En yüksek olasılık >= 0.80               ───► [ KABUL ]
   Eşik <= En yüksek olasılık < 0.80:
      Fark (Top1 - Top2) < Marj Eşiği      ───► [ UNKNOWN (Kararsız / Belirsiz) ]
      Fark (Top1 - Top2) >= Marj Eşiği     ───► [ KABUL (Top1 Sınıfı) ]
   ```

---

## ⚙️ Hiperparametreler ve Eğitim Ayarları

| Parametre | Ön Eğitim (Pre-training) | İnce Ayar (Fine-tuning) |
|:---|:---:|:---:|
| **Optimizer** | AdamW (Weight Decay = $1\times 10^{-4}$) | AdamW (Weight Decay = $1\times 10^{-4}$) |
| **Öğrenme Oranı (LR)** | 5 Epoch Warmup $\rightarrow 1\times 10^{-3}$, Cosine Annealing $\rightarrow 1\times 10^{-5}$ | Cosine Scheduler $\rightarrow 2\times 10^{-4}$ |
| **Batch Boyutu** | 64 | 32 |
| **Epoch Sayısı** | 30 Epoch *(Teorik plan: 80–100)* | 20–30 Epoch |
| **Düzenlileştirme (Regularization)** | 0.2 Dropout, Gradient Clipping (max norm = 1.0) | 0.2 Dropout, Gradient Clipping (max norm = 1.0) |
| **Erken Durdurma (Early Stopping)** | 5 Epoch sabır (Validation Macro F1 takibi) | 5 Epoch sabır (Validation Macro F1 takibi) |
| **Hassasiyet (Precision)** | Otomatik Karışık Hassasiyet (AMP - FP16) | Otomatik Karışık Hassasiyet (AMP - FP16) |

---

## 🔄 Teknik Evrim ve Mühendislik Kararları

Sistem geliştirilirken kademeli olarak çözülen sorunlar ve evrim adımları:

1. **1. Aşama (Temel CNN):** Sadece CNN kullanıldığında model azınlık sınıflarında yetersiz kalıp çoğunluk sınıfına yöneldi; Macro-F1 **~0.59** seviyesinde kaldı. Bu nedenle mimari ResNet-1D + CBAM + Transformer hibrit omurgasına dönüştürüldü.
2. **2. Aşama (Sampler Deformasyonu):** Sınıf dengesizliğini çözmek için kullanılan `WeightedRandomSampler`'ın verinin doğal öncül (prior) dağılımını bozduğu, kalibrasyonu yıktığı ve dağılım dışı (OOD) sinyallere aşırı güvenle yanlış tanı koyduğu tespit edildi.
3. **Çözüm ve Entegrasyon:** Sampler sistemden tamamen çıkarıldı; dengeleme **Asymmetric Loss + Sınıf Ağırlıkları** yapısına devredildi. Bellek darboğazı **HDF5 RAM Caching** ile aşıldı. Safe-Fail karar motorunun eklenmesiyle nihai Macro-F1 **0.768** seviyesine ulaştı.

---

## 💡 Açıklanabilirlik (Explainability)

Klinik kararların hekim tarafından denetlenebilmesi için CBAM ve Transformer bloklarındaki dikkat ağırlıkları (**Attention Rollout**) analiz edilmiştir:
- **AFIB Açıklaması:** Modelin dikkat ağırlığının yaklaşık **%68'inin II, V1 ve aVF** derivasyonlarında yoğunlaştığı; düzensiz RR aralıkları ve P dalgası kayıplarının kararda belirleyici olduğu klinik olarak doğrulanmıştır.
- **LBBB Açıklaması:** Dikkat ağırlığının **V5, V6 ve I** derivasyonlarındaki genişlemiş QRS morfolojisine odaklandığı gözlemlenmiştir.
- **Hata Analizi:** Düşük sinyal kalitesine sahip kayıtlarda dikkat dağılımının zamana yayıldığı ve model güven skorunun belirgin şekilde düştüğü tespit edilmiştir.

---

## ⚡ Performans ve Donanım Gereksinimleri

- **Geliştirme Donanımı:** Intel Core i7-14700HX CPU (28 çekirdek), NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM), 32 GB RAM.
- **İşletim Sistemi ve Yazılım:** Windows 11, Python 3.12, PyTorch (CUDA hızlandırmalı), wfdb, h5py, scikit-learn, scipy.
- **GPU Bellek Kullanımı (VRAM):** 1D sinyal yaklaşımı sayesinde maksimum **3–5 GB VRAM** (8 GB GPU'larda rahatça çalışır).
- **Eğitim Süresi:**
  - *Ön Eğitim (~37.000 kayıt):* ~2 saat (30 Epoch, AMP enabled)
  - *İnce Ayar (5.000 kayıt):* ~20 dakika (20–30 Epoch)
  - *HDF5 Caching Katkısı:* Bir epoch süresi **12 dakikadan 2 dakikanın altına** düşürülmüştür.
- **Çıkarım Hızı (Inference Latency):**
  - GPU Üzerinde: **5 – 10 ms**
  - Modern CPU Üzerinde: **~50 ms** *(Standart hastane bilgisayarlarında ve taşınabilir EKG cihazlarında gerçek zamanlı çalışmaya uygundur)*

---

## 📂 Proje Dizin Yapısı

```bash
├── augment.py             # Veri artırma (Gürültü, Baseline Wander, Genlik, Shift)
├── dataset.py             # HDF5 destekli PyTorch Dataset implementasyonu
├── decision.py            # Temperature scaling, eşik taraması ve Safe-Fail karar mantığı
├── evaluate.py            # Değerlendirme metrikleri (Macro-F1, ROC-AUC, MCC, Confusion Matrix)
├── explainability.py      # Attention Rollout ve açıklanabilirlik fonksiyonları
├── genelsema.txt          # Sistem mimarisi ve teknik dokümantasyon
├── hdf5_builder.py        # WFDB/PhysioNet ham verilerini HDF5'e dönüştürme scripti
├── loss.py                # Sınıf ağırlıklı Asymmetric Loss (ASL) implementasyonu
├── model.py               # Hibrit ResNet-1D + CBAM + Transformer mimarisi
├── preprocess.py          # Bandpass (0.5-45Hz), Notch (50Hz), Z-score filtreleme
├── rapor_kaynaklar.txt    # Hesaplama kaynakları ve PDR rapor taslakları
├── train.py               # Pretraining ve Finetuning eğitim boru hattı
├── README.md              # Türkçe dokümantasyon
└── README_EN.md           # İngilizce dokümantasyon
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Ortam Kurulumu
```bash
git clone https://github.com/Devre181/EKG-Tarayici.git
cd EKG-Tarayici

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install wfdb h5py scikit-learn scipy numpy pandas
```

### 2. Veri Setlerini HDF5 Formatına Dönüştürme
```bash
python hdf5_builder.py
```

### 3. Modeli Eğitme
```bash
python train.py
```

---

## 📚 Referanslar ve Literatür

1. **Ribeiro, A. H., et al. (2020).** Automatic diagnosis of the 12-lead ECG using a deep neural network. *Nature Communications*, 11(1), 1760.
2. **Strodthoff, N., et al. (2021).** Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL. *IEEE JBHI*, 25(5), 1519–1528.
3. **Zhu, H., et al. (2020).** Automatic multilabel electrocardiogram diagnosis of heart rhythm or conduction abnormalities with deep learning. *The Lancet Digital Health*, 2(7), e348–e357.
4. **Zhou, F., & Fang, D. (2025).** Classification of multi-lead ECG based on multiple scales and hierarchical feature convolutional neural networks. *Scientific Reports*, 15, 16418.
5. **Najia, M., & Faouzi, B. (2025).** An Enhanced Hybrid Model Combining CNN, BiLSTM, and Attention Mechanism for ECG Segment Classification. *Biomedical Engineering and Computational Biology*, 16, 1–14.
6. **Alghieth, M. (2025).** DeepECG-Net: A Hybrid Transformer-Based Deep Learning Model for Real-Time ECG Anomaly Detection. *Scientific Reports*, 15(1), 20714.

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır. Klinik ve akademik amaçlı araştırmalara açıktır.

<div align="center">
  <b>TEKNOFEST 2026 — Devre181 Takımı 🚀</b>
</div>
