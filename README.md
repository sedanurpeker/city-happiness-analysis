<div align="center">

#  City Emotion Analysis with Grey Relational Analysis (GRA)
### Şehir Duygu Analizi ve Gri İlişkisel Analiz (GRA) ile Mutluluk Sıralaması

Deep learning based facial emotion recognition + Grey Relational Analysis to rank Turkish cities by "happiness."
Türkiye şehirlerini, derin öğrenme tabanlı yüz duygu tanıma ve Gri İlişkisel Analiz (GRA) yöntemiyle "mutluluk" skoruna göre sıralayan bir sistem.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-000000?logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)
![DeepFace](https://img.shields.io/badge/DeepFace-Emotion%20Recognition-orange)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-ORM-D71F00)
![License](https://img.shields.io/badge/License-Academic%20Project-lightgrey)

** Live Demo / Canlı Demo:** [sedanurpeker.github.io/docs](https://sedanurpeker.github.io/docs/)

</div>

---

##  English

###  Overview
This is a **graduation (thesis) project** that analyzes emotions detected in city walking-tour videos (or user-uploaded videos) using facial detection + emotion recognition, tracks unique individuals across frames, aggregates the results per city, and ranks cities by a **happiness score** computed with **Grey Relational Analysis (GRA)** — a multi-criteria decision-making method. Results are served through an interactive Flask web dashboard with maps, charts, and downloadable academic accuracy reports.

###  Key Features
-  **Video upload & offline analysis** — upload any city or general video for emotion analysis
-  **Real-time webcam emotion detection** with a calibration step (HSEmotion / ONNX)
-  **Face detection + tracking** — a custom multi-object tracker (IoU + motion + embedding re-identification) ensures each person is counted **only once**
-  **Robust filtering pipeline** — rejects masked faces, posters/screens, static/duplicate detections, and low-confidence predictions to keep results defensible
-  **Grey Relational Analysis (GRA)** — computes a multi-criteria happiness grade per city from 7 emotion classes (angry, disgust, fear, happy, sad, surprise, neutral)
-  **City happiness ranking** with a reliability factor based on sample size (unique people analyzed)
-  **Interactive map (Leaflet)** and **charts (Chart.js / Plotly)** for visual analytics
-  **Automated academic accuracy report** (methodology, confidence-margin distribution, temporal consistency, pipeline diagram) — since no ground-truth labels exist, the system defends its reliability through internal-consistency metrics instead of classic accuracy
-  **Time-of-day breakdown** (morning / afternoon / evening) per city
-  **Relational database** (SQLAlchemy ORM, MSSQL/SQLite) storing cities, videos, frames, faces, emotions, and GRA results

###  Tech Stack
| Layer | Technologies |
|---|---|
| Backend | Python, Flask |
| Computer Vision | OpenCV, DeepFace, HSEmotion (ONNX) |
| Data / ORM | SQLAlchemy, MSSQL / SQLite |
| Analysis | Pandas, NumPy, Grey Relational Analysis (custom implementation) |
| Frontend | Bootstrap 5, Chart.js, Plotly.js, Leaflet.js |
| Reporting | Matplotlib (PDF/PNG report generation) |

###  Project Structure (simplified)
```
├── app.py                     # Flask application & routes
├── db.py / models.py          # SQLAlchemy engine, session & ORM models
├── init_db.py / create_db.py  # DB bootstrap scripts
├── schema.sql                 # Raw SQL schema reference
├── tracker.py                 # Multi-person tracker (IoU + embedding re-ID)
├── video_to_emotions.py        # Core video → emotion pipeline (profiles: city/generic/realtime)
├── gra_core.py                 # GRA math (normalization, reference vector, grading)
├── gra.py / gra_analysis.py    # City-level & time-period GRA runners
├── accuracy_report.py          # Academic methodology & confidence report generator
├── realtime_emotion_webcam.py  # Live webcam emotion detector
├── templates/                  # index, upload, ranking, city, realtime, report pages
└── static/                     # style.css, images
```

###  Setup
```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/city-emotion-gra.git
cd city-emotion-gra

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Set MSSQL_DSN (or switch db.py to SQLite for local testing)

# 5. Initialize the database
python init_db.py

# 6. Run the web app
python app.py
```
Then open `http://localhost:5000` in your browser.

###  Methodology Note
Since there is no manually labeled ground-truth dataset for street-level emotion recognition, this project does **not** claim a classic "accuracy" number. Instead, reliability is defended through:
1. **Multi-frame temporal consistency** (same person re-evaluated across frames, time-weighted)
2. **Confidence margin** (gap between top-1 and top-2 emotion scores)
3. **Embedding-based re-identification** (prevents double-counting the same person)
4. **Strict quality filters** (size, occlusion, mask/poster detection, blur, skin-tone checks)

This is documented automatically in the generated **academic accuracy report**.

###  Known Limitations
- Distant/wide-angle city footage yields smaller faces → lower detection confidence
- Lighting conditions and camera angle affect emotion classification accuracy
- Occlusion (masks, sunglasses, hands) reduces usable samples
- Real-time performance depends on hardware (GPU strongly recommended for DeepFace)

###  Academic Use
This repository was built as part of a **graduation project** at **Sakarya University of Applied Sciences, Department of Computer Engineering (2025–2026)**, under the supervision of **Assoc. Prof. Dr. Selman HIZAL**. Contributions, forks, and educational reuse are welcome — please cite this repository if used in academic work.

 **Live demo / project site:** https://sedanurpeker.github.io/docs/

---

## 🇹🇷 Türkçe

###  Proje Hakkında
Bu proje, şehir gezinti videolarından (ya da kullanıcı tarafından yüklenen videolardan) yüz tespiti ve duygu tanıma yoluyla duygu analizi yapan, videodaki kişileri kare boyunca takip ederek **her kişiyi yalnızca bir kez** sayan, şehir bazlı sonuçları toplayan ve **Gri İlişkisel Analiz (GRA)** adlı çok kriterli karar verme yöntemiyle şehirleri **mutluluk skoruna** göre sıralayan bir **bitirme (lisans/yüksek lisans tez) projesidir**. Sonuçlar; harita, grafik ve indirilebilir akademik doğruluk raporlarıyla birlikte interaktif bir Flask web panelinde sunulur.

###  Öne Çıkan Özellikler
-  **Video yükleme ve çevrimdışı analiz** — şehir videosu veya genel video yükleyerek duygu analizi yapılabilir
-  **Kalibrasyon destekli gerçek zamanlı webcam duygu analizi** (HSEmotion / ONNX tabanlı)
-  **Yüz tespiti + takip sistemi** — özel geliştirilmiş çoklu nesne takipçisi (IoU + hareket + embedding tabanlı yeniden tanıma) sayesinde her kişi **yalnızca bir kez** sayılır
-  **Güçlü filtreleme hattı** — maskeli yüzler, poster/ekran görüntüleri, statik/tekrarlanan tespitler ve düşük güvenli tahminler elenerek sonuçların savunulabilirliği artırılır
-  **Gri İlişkisel Analiz (GRA)** — 7 duygu sınıfından (kızgın, iğrenme, korku, mutlu, üzgün, şaşkın, nötr) şehir bazlı çok kriterli mutluluk skoru hesaplar
-  **Şehir mutluluk sıralaması** — analiz edilen benzersiz kişi sayısına dayalı güvenilirlik katsayısıyla ağırlıklandırılır
-  **İnteraktif harita (Leaflet)** ve **grafikler (Chart.js / Plotly)** ile görsel analiz
-  **Otomatik akademik doğruluk raporu** — ground truth (referans etiket) verisi bulunmadığından, sistem klasik "accuracy" yerine güven marjı, zaman tutarlılığı ve pipeline şeması gibi iç tutarlılık metrikleriyle güvenilirliğini savunur
-  Şehir bazında **sabah / öğleden sonra / akşam** zaman dilimi karşılaştırması
-  **İlişkisel veritabanı** (SQLAlchemy ORM, MSSQL/SQLite) — şehirler, videolar, kareler, yüzler, duygular ve GRA sonuçları saklanır

###  Teknoloji Yığını
| Katman | Teknolojiler |
|---|---|
| Backend | Python, Flask |
| Bilgisayarlı Görü | OpenCV, DeepFace, HSEmotion (ONNX) |
| Veri / ORM | SQLAlchemy, MSSQL / SQLite |
| Analiz | Pandas, NumPy, Gri İlişkisel Analiz (özel implementasyon) |
| Frontend | Bootstrap 5, Chart.js, Plotly.js, Leaflet.js |
| Raporlama | Matplotlib (PDF/PNG rapor üretimi) |

###  Proje Yapısı (özet)
```
├── app.py                     # Flask uygulaması ve route'lar
├── db.py / models.py          # SQLAlchemy engine, session ve ORM modelleri
├── init_db.py / create_db.py  # Veritabanı kurulum betikleri
├── schema.sql                 # Ham SQL şeması referansı
├── tracker.py                 # Çoklu kişi takip sistemi (IoU + embedding yeniden tanıma)
├── video_to_emotions.py        # Ana video → duygu analizi hattı (city/generic/realtime profilleri)
├── gra_core.py                 # GRA matematiği (normalizasyon, referans vektör, skorlama)
├── gra.py / gra_analysis.py    # Şehir bazlı ve zaman dilimi bazlı GRA çalıştırıcıları
├── accuracy_report.py          # Akademik metodoloji ve güven raporu üretici
├── realtime_emotion_webcam.py  # Canlı webcam duygu analizi
├── templates/                  # index, upload, ranking, city, realtime, rapor sayfaları
└── static/                     # style.css, görseller
```

###  Kurulum
```bash
# 1. Depoyu klonla
git clone https://github.com/<kullanici-adiniz>/city-emotion-gra.git
cd city-emotion-gra

# 2. Sanal ortam oluştur
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Ortam değişkenlerini ayarla
cp .env.example .env
# MSSQL_DSN değerini gir (yerel test için db.py'yi SQLite'a çevirebilirsiniz)

# 5. Veritabanını başlat
python init_db.py

# 6. Uygulamayı çalıştır
python app.py
```
Ardından tarayıcıda `http://localhost:5000` adresini açın.

###  Metodoloji Notu
Sokak/şehir düzeyinde duygu tanıma için elle etiketlenmiş bir ground-truth veri seti bulunmadığından, bu proje klasik bir "accuracy" (doğruluk) değeri **iddia etmez**. Bunun yerine güvenilirlik şu şekilde savunulur:
1. **Çoklu kare zaman tutarlılığı** (aynı kişi, zaman ağırlıklı olarak birden fazla karede yeniden değerlendirilir)
2. **Güven marjı** (en yüksek ve ikinci en yüksek duygu skoru arasındaki fark)
3. **Embedding tabanlı yeniden tanıma** (aynı kişinin birden fazla kez sayılmasını engeller)
4. **Sıkı kalite filtreleri** (boyut, okluzyon, maske/poster tespiti, bulanıklık, cilt tonu kontrolü)

Bu bilgiler otomatik olarak üretilen **akademik doğruluk raporunda** belgelenir.

###  Bilinen Sınırlılıklar
- Uzak/geniş açılı şehir görüntülerinde yüzler küçük kalır → tespit güveni düşer
- Işıklandırma koşulları ve kamera açısı duygu sınıflandırma doğruluğunu etkiler
- Okluzyon (maske, güneş gözlüğü, el) kullanılabilir örnek sayısını azaltır
- Gerçek zamanlı performans donanıma bağlıdır (DeepFace için GPU şiddetle önerilir)

###  Akademik Kullanım
Bu depo, **Sakarya Uygulamalı Bilimler Üniversitesi, Bilgisayar Mühendisliği Bölümü (2025–2026)** kapsamında, **Doç. Dr. Selman HIZAL** danışmanlığında hazırlanan bir **bitirme projesidir**. Katkılar, fork'lar ve eğitim amaçlı yeniden kullanım memnuniyetle karşılanır — akademik çalışmalarda kullanılması durumunda lütfen bu depoyu kaynak gösterin.

 **Canlı demo / proje sitesi:** https://sedanurpeker.github.io/docs/

---

<div align="center">

**Author / Geliştiren:** Sedanur PEKER
**Advisor / Danışman:** Doç. Dr. Selman HIZAL
**Institution / Kurum:** Sakarya Uygulamalı Bilimler Üniversitesi — Bilgisayar Mühendisliği
**Project Type / Proje Türü:** Graduation Project 2025–2026 (Bitirme Projesi)
**Live Demo / Canlı Demo:** [sedanurpeker.github.io/docs](https://sedanurpeker.github.io/docs/)

</div>
