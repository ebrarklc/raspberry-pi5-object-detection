# Bölüm 4 — 🏆 Proje 2: Nesne Tespiti (SSD MobileNet & EfficientDet)

> **Kaynak:** Marcelo Rovai, *Edge AI Engineering — Hands-On with the Raspberry Pi*, s. 146–168
> **Ön koşul:** [Bölüm 2](https://github.com/ebrarklc/raspberry-pi5-image-classification-litert) (venv `~/tflite_env`, LiteRT, Jupyter) ve [Bölüm 3](https://github.com/ebrarklc/raspberry-pi5-custom-image-classifier) (Flask canlı akış)
> **Donanım:** Raspberry Pi 5 (4 GB), Camera Module v2 (IMX219)

---

## İçindekiler
1. [Teori: nesne tespiti nedir?](#1-teori-nesne-tespiti-nedir)
2. [IoU, NMS ve mAP](#2-iou-nms-ve-map)
3. [Kurulum](#3-kurulum)
4. [SSD MobileNet V1'in içi](#4-ssd-mobilenet-v1in-içi)
5. [İlk tespit ve kutu çizimi](#5-ilk-tespit-ve-kutu-çizimi)
6. [SSD MobileNet vs EfficientDet](#6-ssd-mobilenet-vs-efficientdet)
7. [Canlı nesne tespiti](#7-canlı-nesne-tespiti)
8. [Sorun giderme](#8-sorun-giderme)
9. [Terimler sözlüğü](#9-terimler-sözlüğü)
10. [Mülakat notları](#10-mülakat-notları)

---

## 1. Teori: nesne tespiti nedir?

| | Sınıflandırma | Nesne tespiti |
|---|---|---|
| Soru | "Bu resimde **ne** var?" | "**Neler** var ve **nerede**?" |
| Çıktı | 1 etiket + olasılık | Her nesne için: etiket + skor + **bounding box** |
| Örnek | `desk %38` | `keyboard %67 [kutu]`, `mouse %72 [kutu]`, `bottle %54 [kutu]` |

**Tespit = Lokalizasyon (nerede?) + Sınıflandırma (ne?)**

### Bounding box
`[ymin, xmin, ymax, xmax]` — sol üst ve sağ alt köşe, **normalize** (0–1, resim boyutundan bağımsız).
Piksele dönüşüm: `x = xmin × W`, `y = ymin × H`, `w = (xmax − xmin) × W`, `h = (ymax − ymin) × H`.

### İki yaklaşım
| | İki aşamalı (two-stage) | Tek aşamalı (single-stage) |
|---|---|---|
| Nasıl | ① Bölge öner ② her bölgeyi sınıflandır | Tek ileri geçişte tüm kutular + sınıflar |
| Örnek | R-CNN, Faster R-CNN | **SSD**, **EfficientDet**, YOLO, FOMO |
| Hız | Yavaş | ⚡ Hızlı → edge için ideal |

### SSD (Single Shot MultiBox Detector)
- Resmin üzerine farklı boyut/oranlarda önceden yerleştirilmiş **anchor (hazır) kutular** (SSD MobileNet V1 300×300'de ~1900 tane).
- Her anchor için: "kutuyu ne kadar kaydır/büyüt?" + "içinde hangi sınıf var?" tahmini.
- Farklı derinlikteki özellik haritalarından tahmin → farklı ölçekler.

### EfficientDet
- **EfficientNet** backbone + **BiFPN** (çift yönlü özellik piramidi) + bileşik ölçekleme.
- BiFPN: Erken katmanlar (yüksek çözünürlük, az anlam) ile derin katmanların (düşük çözünürlük, zengin anlam) bilgisini yukarı ve aşağı defalarca birleştirir → **küçük nesnelerde** daha başarılı.

### COCO
Nesne tespitinin "ImageNet'i": 200 binden fazla etiketli resim, **80 nesne sınıfı** (91 ID'lik orijinal listeden 11'i etiketlenmemiş).

---

## 2. IoU, NMS ve mAP

### IoU (Intersection over Union)
$$\text{IoU} = \frac{\text{Kesişim}}{\text{Birleşim}} = \frac{A \cap B}{A + B - A \cap B}$$

**Örnek:** Gerçek `[0,0,4,4]`, tahmin `[2,2,6,6]` → kesişim 2×2 = **4**, birleşim 16 + 16 − 4 = **28**, IoU = **0.14** → yanlış tespit (genelde eşik 0.5).
➡️ IoU acımasızdır: Kutunun biraz kayması bile skoru hızla düşürür.

### NMS (Non-Maximum Suppression)
1. Kutuları skora göre sırala.
2. En yüksek skorlu kutuyu tut.
3. Onunla **aynı sınıfta** olup IoU'su eşiği aşan kutuları sil.
4. Kalanlarla tekrarla → nesne başına tek kutu.

> Kedi ve köpek kutuları örtüşse de ikisi de kalır: NMS yalnızca **aynı sınıftaki** kutuları karşılaştırır.

### mAP (mean Average Precision)
Her sınıf için precision–recall eğrisinin altındaki alan (**AP**) → tüm sınıfların ortalaması (**mAP**). `mAP@0.5`: IoU ≥ 0.5 olan tespitler doğru sayılır.

### Güven eşiği
| Eşik | Etki |
|---|---|
| Düşük (0.2) | Daha çok tespit, daha çok yanlış pozitif (recall ↑, precision ↓) |
| Orta (0.5) | Dengeli başlangıç |
| Yüksek (0.8) | Az ama emin tespit, gerçek nesneler kaçabilir (precision ↑, recall ↓) |

---

## 3. Kurulum

```bash
source ~/tflite_env/bin/activate
mkdir -p ~/Documents/TFLITE/OBJ_DETECT/models ~/Documents/TFLITE/OBJ_DETECT/images
cd ~/Documents/TFLITE/OBJ_DETECT

BASE=https://raw.githubusercontent.com/Mjrovai/EdgeML-with-Raspberry-Pi/refs/heads/main/OBJ_DETEC
wget -P models $BASE/models/ssd-mobilenet-v1-tflite-default-v1.tflite
wget -P models $BASE/models/lite-model_efficientdet_lite0_detection_metadata_1.tflite
wget -P models $BASE/models/coco_labels.txt
wget -P images $BASE/images/cat_dog.jpeg
wget -P images $BASE/images/home-office.jpg
wget -P images $BASE/images/office.jpeg
```

| Dosya | Boyut |
|---|---|
| ssd-mobilenet-v1-tflite-default-v1.tflite | 4.0 MB |
| lite-model_efficientdet_lite0_detection_metadata_1.tflite | 4.4 MB |
| coco_labels.txt | 690 B |

### Etiket dosyası
```
person (0), bicycle (1), ..., streetsign (11), stopsign (12), ..., cat (16), dog (17), ...
```
`streetsign` COCO'nun 80 sınıfında **yok**; orijinal 91 ID'lik listede yer tutucu olarak duruyor ki indeksler kaymasın.

> ⚠️ **Off-by-one tuzağı:** Etiket dosyasında bir satır eksik/fazla olursa tüm etiketler kayar; model kedi bulur, ekrana "dog" yazılır.

### Jupyter (farklı ağlarda da çalışsın)
```bash
jupyter notebook --ip=0.0.0.0 --no-browser
```
`0.0.0.0` = Pi'nin sahip olduğu **her adreste** dinle. IP'yi öğrenmek için `hostname -I`.

---

## 4. SSD MobileNet V1'in içi

```
GİRİŞ: [  1 300 300   3] uint8 | quantization: (0.0078125, 128)

ÇIKIŞ SAYISI: 4
  [0] TFLite_Detection_PostProcess     shape=[ 1 10  4]  dtype=float32   → boxes
  [1] TFLite_Detection_PostProcess:1   shape=[ 1 10]     dtype=float32   → classes
  [2] TFLite_Detection_PostProcess:2   shape=[ 1 10]     dtype=float32   → scores
  [3] TFLite_Detection_PostProcess:3   shape=[1]         dtype=float32   → count
```
- Giriş quantization'ı Bölüm 2'deki MobileNet V2 ile **aynı** → ham 0–255 piksel verilir, model içeride [−1, 1]'e çevirir.
- En fazla **10** tespit.
- Çıkışlar float32: `TFLite_Detection_PostProcess` katmanı **anchor çözme + NMS**'i modelin içinde yapıyor.

---

## 5. İlk tespit ve kutu çizimi

```python
img = orig_img.resize((300, 300))                          # squash
input_data = np.expand_dims(np.array(img), axis=0)        # (1, 300, 300, 3) uint8
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
```

**cat_dog.jpeg (3915×3150) — ham çıktı (38.6 ms):**
```
0: cat          skor=0.73  kutu=[0.408 0.038 1.002 0.432]
1: dog          skor=0.70  kutu=[0.202 0.413 0.994 0.954]
2: dog          skor=0.27  ...
...
7: diningtable  skor=0.12  kutu=[0.749 -0.056 1.009 0.91]
```
- İlk iki tespit doğru; geri kalanı düşük skorlu kopyalar/yanlışlar → **eşik 0.5**.
- Koordinatlar 0–1 dışına taşabiliyor → **`np.clip(box, 0, 1)`**.
- Sınıflandırıcıya (1.9 ms) göre ~20 kat ağır: daha büyük giriş (300² vs 160²), ~1900 anchor × (kutu + 91 sınıf) tahmini, NMS.

Çizim (matplotlib):
```python
x, y = xmin * W, ymin * H
w, h = (xmax - xmin) * W, (ymax - ymin) * H
ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=3, edgecolor="red", facecolor="none"))
```

**Ofis resimleri (SSD, eşik 0.5):**
| Resim | Tespitler | Kaçırılanlar |
|---|---|---|
| home-office | laptop %82 | kupa, telefon, defter (hepsi COCO'da var) → **küçük nesne zayıflığı** |
| office | laptop, 2× bottle, chair, potted plant | çerçeveler (COCO'da "picture frame" yok → closed-set) |

---

## 6. SSD MobileNet vs EfficientDet

```
EfficientDet-Lite0 GİRİŞ: [1 320 320 3] uint8 (0.0078125, 127)
  [0] StatefulPartitionedCall:3  shape=[1 25 4]
  [1] StatefulPartitionedCall:2  shape=[1 25]
  [2] StatefulPartitionedCall:1  shape=[1 25]
  [3] StatefulPartitionedCall:0  shape=[1]
```
İsimler ve sıralama SSD'den farklı, en fazla **25** tespit. → Çıkışlar **içeriğe göre** tanınır:

```python
def read_outputs(itp):
    boxes = classes = scores = count = None
    for o in itp.get_output_details():
        t = itp.get_tensor(o['index'])[0] if len(o['shape']) > 1 else itp.get_tensor(o['index'])
        if t.ndim == 2 and t.shape[-1] == 4:  boxes = t
        elif t.size == 1:                     count = int(t.ravel()[0])
        elif np.allclose(t, np.round(t)):     classes = t     # sınıf numaraları tam sayıdır
        else:                                 scores = t
    return boxes, classes, scores, count
```

**Sonuçlar (Pi 5, 4 thread, eşik 0.5):**
| Resim | SSD MobileNet V1 | EfficientDet-Lite0 |
|---|---|---|
| cat_dog | 27.5 ms · cat %73, dog %70 | **21.8 ms** · cat **%79**, dog **%77** |
| home-office | 45.1 ms · laptop %82 | **30.5 ms** · laptop %69, **cell phone %59**, **cup %57** |
| office | 27.9 ms · laptop %63, bottle %61, chair %56, bottle %51, potted plant %50 | **24.6 ms** · chair **%84**, laptop %71, potted plant %59 |

**Yorum:**
- EfficientDet hem **daha hızlı** hem SSD'nin kaçırdığı **küçük nesneleri** (telefon, kupa) buldu → BiFPN.
- Ama office'te şişeleri kaçırdı → **hiçbir model her durumda en iyi değil**; kendi verinde test et.
- Süreler tek ölçüm (her biri warm-up sonrası); kesin karşılaştırma için çoklu ölçüm + ortalama ± std gerekir.

---

## 7. Canlı nesne tespiti

[`live_detect.py`](live_detect.py):
```bash
python live_detect.py          # EfficientDet-Lite0 (varsayılan)
python live_detect.py ssd      # SSD MobileNet V1
# → http://<PI_IP>:5000  (okul ağında açılmazsa Pi'de http://localhost:5000)
```

**Yapı:** Bölüm 3'teki canlı sınıflandırıcıyla aynı (kamera thread'i + lock + MJPEG), farkları:
| Özellik | Açıklama |
|---|---|
| `sys.argv[1]` | Model komut satırından seçilir |
| `read_outputs()` | İki modelde de çalışan içerik tabanlı çıkış okuma |
| Kamera 640×480 | Kutular ve yazılar net görünsün; model yine 300/320'ye küçültür |
| `color_for(class_id)` | Sınıf numarası rastgele üretecin **seed**'i → her sınıf her karede aynı renk |
| Etiket arka planı | `cv2.getTextSize` ile metin boyutu ölçülüp dolu dikdörtgen çizilir |

**Sonuçlar (EfficientDet-Lite0):**
| Sahne | Tespitler | Çıkarım | Sistem FPS |
|---|---|---|---|
| Tek nesne | mouse %69 | 20.5 ms | 15 |
| Çoklu nesne | mouse %72, keyboard %67, bottle %54 | 20.7 ms | 15 |

➡️ Model tek başına ~50 FPS'e yetecek hızda; sistem 15 FPS. Darboğaz: kamera yakalama + çizim + JPEG + akış zinciri (Bölüm 1–3'teki pipeline dersi).

---

## 8. Sorun giderme

| Hata | Sebep | Çözüm |
|---|---|---|
| `OSError: [Errno 99] Cannot assign requested address` (Jupyter) | `--ip=` ile verilen adres Pi'de yok (farklı ağ / IP değişti) | `hostname -I` ile yeni IP; ya da `--ip=0.0.0.0` |
| Bilgisayardan Pi'ye bağlanılamıyor (okul ağı) | Ağda **client isolation** olabilir | Pi'de `http://localhost:8888` / `:5000` |
| `No camera number 0 found` | Sistem kamerayı görmüyor; genelde gevşek şerit kablo (taşıma!) | Pi'yi kapat → kabloyu iki uçtan yeniden tak → `rpicam-hello --list-cameras`, `dmesg \| grep -i imx219` |
| `Device or resource busy` | Kamerayı başka süreç kullanıyor (ör. Jupyter kernel) | `pgrep -af python` → ilgili süreci kapat |
| Kutular resmin dışına taşıyor | Koordinatlar 0–1 dışında | `np.clip(box, 0, 1)` |
| Etiketler kaymış (kedi → "dog") | Etiket dosyası model indeksleriyle uyumsuz | Modelle birlikte gelen etiket dosyasını kullan |

---

## 9. Terimler sözlüğü

| Terim | Açıklama |
|---|---|
| **Nesne tespiti** | Bir resimdeki her nesnenin sınıfını ve konumunu (kutusunu) bulma. |
| **Bounding box** | Nesneyi saran dikdörtgen: `[ymin, xmin, ymax, xmax]`. |
| **Normalize koordinat** | Kutu koordinatlarının 0–1 arası oran olarak verilmesi. |
| **Lokalizasyon** | Nesnenin yerini bulma. |
| **Single-stage / two-stage** | Tek ileri geçişte tespit / önce bölge önerip sonra sınıflandırma. |
| **SSD** | Single Shot MultiBox Detector; anchor kutular üzerinden tek geçişte tespit. |
| **Anchor box** | Resme önceden yerleştirilmiş, modelin kaydırıp ölçeklediği hazır kutular. |
| **Backbone** | Özellik çıkaran gövde ağı (SSD'de MobileNet, EfficientDet'te EfficientNet). |
| **BiFPN** | Farklı çözünürlükteki özellikleri çift yönlü birleştiren piramit ağı. |
| **COCO** | 80 sınıflık nesne tespiti veri seti. |
| **IoU** | İki kutunun kesişim / birleşim oranı. |
| **NMS** | Aynı nesneye ait örtüşen kutulardan yalnızca en yüksek skorluyu bırakma. |
| **mAP** | Sınıf başına AP'lerin ortalaması; tespit modellerinin genel başarı ölçüsü. |
| **Güven eşiği** | Altındaki tespitlerin atıldığı skor değeri. |
| **False positive / negative** | Olmayan nesneyi bulmak / var olanı kaçırmak. |
| **Client isolation** | Ağdaki cihazların birbirini görmesinin engellenmesi (okul/kafe ağları). |
| **dmesg** | Linux çekirdek mesajları; donanım tanıma hatalarını gösterir. |
| **seed** | Rastgele sayı üretecinin başlangıç değeri; aynı seed → aynı sayılar. |

---

## 10. Mülakat notları

**"Sınıflandırma ile tespit farkı?"**
Sınıflandırma tüm resme tek etiket verir; tespit her nesne için sınıf + skor + kutu verir. Tespit = lokalizasyon + sınıflandırma.

**"IoU nedir, nasıl hesaplanır?"**
Kesişim / birleşim; birleşim = A + B − kesişim. Genelde IoU ≥ 0.5 doğru tespit sayılır.

**"NMS ne işe yarar?"**
Aynı nesne için üretilen örtüşen kutulardan, skora göre en iyisini tutup aynı sınıftaki yüksek IoU'lu kopyaları siler.

**"mAP@0.5 ne demek?"**
IoU ≥ 0.5 olan tespitlerin doğru sayıldığı, sınıf başına AP'lerin ortalaması.

**"Tek aşamalı dedektörler neden edge'de tercih edilir?"**
Tek ileri geçişte tüm tahminleri ürettikleri için hızlıdırlar (SSD, YOLO, EfficientDet, FOMO).

**"Küçük nesne tespiti neden zor, nasıl iyileştirilir?"**
Giriş küçültülünce birkaç piksele iner; derin katmanlarda kaybolur. Çözüm: çok ölçekli özellik birleştirme (FPN/BiFPN), daha yüksek giriş çözünürlüğü, tiling.

**"Model değiştirdiğinde kodun kırılmaması için?"**
Çıkışları indekse göre değil isim/şekil/içeriğe göre eşle; giriş tipini ve normalizasyonu modelden oku.

**"Eşik değerini nasıl seçersin?"**
Precision–recall takası: hiçbir şeyi kaçırmamak gerekiyorsa düşük, yanlış alarmın pahalı olduğu yerde yüksek eşik.

**"Saha sisteminde 'yazılım hatası' gibi görünen bir problemi nasıl ayıklarsın?"**
Önce donanım görülüyor mu (`--list-cameras`, `dmesg`, `lsusb`, `i2cdetect`), güç ve ısı normal mi; sonra koda bak.
