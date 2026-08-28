# OCR Aksara Jawa - Streamlit App Eksperimen Jurnal

Aplikasi web berbasis **Streamlit** untuk mendeteksi kata target pada citra manuskrip Sasradiningrat II dan mengenali transliterasi Latinnya menggunakan pipeline **YOLOv8n + CRNN + CTC greedy decoding**.

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)](https://javanese-script-ocr.streamlit.app/)

---

## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Arsitektur Pipeline](#arsitektur-pipeline)
- [Detail Model](#detail-model)
- [Struktur Folder](#struktur-folder)
- [Instalasi Lokal](#instalasi-lokal)
- [Deployment ke Streamlit Community Cloud](#deployment-ke-streamlit-community-cloud)
- [Cara Penggunaan](#cara-penggunaan)
- [Konfigurasi](#konfigurasi)
- [Ruang Lingkup](#ruang-lingkup)

---

## Gambaran Umum

Aplikasi ini merupakan paket deployment mandiri dari eksperimen jurnal. Seluruh kode inferensi, konfigurasi, dan bobot final sudah tersedia di dalam folder ini sehingga aplikasi tidak bergantung pada direktori lain di workspace penelitian.

### Fitur Utama

| Fitur | Deskripsi |
|---|---|
| **Deteksi kata target** | YOLOv8n clean split mendeteksi region kata yang sesuai ruang lingkup anotasi |
| **Pengenalan karakter** | CRNN VGG16-BiLSTM mengenali setiap crop menjadi urutan karakter Latin |
| **CTC decoding** | Greedy decoding menghapus blank dan menggabungkan prediksi berulang |
| **Urutan baca** | Baseline-center clustering V2 menyusun kata per baris dari kiri ke kanan |
| **Pratinjau dan visualisasi** | Menampilkan citra input, bounding box, nomor urut, dan hasil transliterasi |
| **Ekspor hasil** | Hasil dapat diunduh sebagai teks, CSV, dan gambar visualisasi |
| **Tampilan responsif** | Antarmuka disesuaikan untuk desktop dan ponsel |

---

## Arsitektur Pipeline

```text
Input citra halaman
        |
        v
YOLOv8n clean split
Deteksi bounding box kata target
        |
        v
Crop region tanpa margin tambahan
        |
        v
RGB + direct resize 32 x 128
Normalisasi ImageNet
        |
        v
CRNN: VGG16 Blocks 1-3
+ Adaptation Layers
+ BiLSTM dua arah
+ Linear 24 kelas
        |
        v
CTC greedy decoding
        |
        v
Baseline-center reading order V2
        |
        v
Transliterasi Latin per baris
```

---

## Detail Model

### 1. YOLOv8n Clean Split - Deteksi Kata

| Parameter | Nilai |
|---|---|
| Model | YOLOv8n pretrained yang dilatih ulang |
| Task | Object detection satu kelas: `kata` |
| Bobot | `models/yolov8n_clean_split_v1_best.pt` |
| Ukuran bobot | Sekitar 5,9 MB |
| Input inference | `imgsz=640` |
| Confidence default | 0,25 |
| NMS IoU | 0,70 |
| Maksimum deteksi | 300 |

Hasil test clean split pada 16 citra dan 452 bounding box:

| Metrik | Nilai |
|---|---:|
| Precision | 0,4224 |
| Recall | 0,3695 |
| mAP50 | 0,3482 |
| mAP50-95 | 0,2303 |

### 2. CRNN Direct Resize - Pengenalan Karakter

```text
Input RGB (3, 32, 128)
        |
VGG16 Blocks 1-3, frozen
        |
Feature map (256, 4, 16)
        |
Adaptation Layers + Height Pooling
        |
Sequence (16, 512)
        |
BiLSTM 2 layer, hidden 256 per arah
        |
Linear (512, 24)
        |
CTC greedy decoding
```

| Parameter | Nilai |
|---|---|
| Bobot | `models/crnn_direct_resize_v1_best.pt` |
| Ukuran bobot | Sekitar 63 MB |
| Preprocessing | RGB, direct resize 32 x 128, normalisasi ImageNet |
| Feature extractor | VGG16 Blocks 1-3 |
| Recurrent layer | BiLSTM dua layer, hidden size 256 per arah |
| Sequence length | 16 timestep |
| Charset | 23 karakter Latin |
| Output | 23 karakter + 1 blank CTC = 24 kelas |

Hasil pengujian pada 477 ground-truth crop yang independen:

| Metrik | Nilai |
|---|---:|
| Word Accuracy | 0,8910 |
| Word Error Rate | 0,1090 |
| Character Error Rate | 0,0349 |

### 3. Charset

Charset aktual tersedia pada `models/charset.txt`:

```text
abcdeghijklmnoprstuwyèê
```

---

## Struktur Folder

```text
upload github/
|-- app.py
|-- pipeline.py
|-- models.py
|-- preprocessing.py
|-- ctc_utils.py
|-- deployment_config.json
|-- requirements.txt
|-- packages.txt
|-- README.md
`-- models/
    |-- yolov8n_clean_split_v1_best.pt
    |-- crnn_direct_resize_v1_best.pt
    `-- charset.txt
```

### Penjelasan File

| File | Fungsi |
|---|---|
| `app.py` | Antarmuka Streamlit, upload, pratinjau, inferensi, visualisasi, tab hasil, dan unduhan |
| `pipeline.py` | Integrasi YOLOv8n, crop, batch recognition, reading order, dan penyusunan teks |
| `models.py` | Definisi CRNN VGG16-Adaptation-BiLSTM |
| `preprocessing.py` | Direct resize dan preprocessing CRNN |
| `ctc_utils.py` | Konversi charset dan CTC greedy decoding |
| `deployment_config.json` | Path bobot, checksum SHA-256, threshold, dan konfigurasi inference |
| `requirements.txt` | Dependensi Python untuk lokal dan Streamlit Cloud |
| `packages.txt` | Pustaka sistem `libgl1` dan GLib Trixie yang diperlukan OpenCV pada Streamlit Cloud |
| `models/` | Bobot final dan charset |

---

## Instalasi Lokal

### Prasyarat

- Python 3.11 atau 3.12 direkomendasikan
- `pip`
- RAM minimal sekitar 2 GB untuk pemuatan model dan inference CPU

### Langkah Instalasi

1. Masuk ke folder aplikasi:

   ```bash
   cd "upload github"
   ```

2. Buat virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Pasang dependensi:

   ```bash
   pip install -r requirements.txt
   ```

4. Jalankan aplikasi:

   ```bash
   streamlit run app.py
   ```

5. Buka `http://localhost:8501`.

---

## Deployment ke Streamlit Community Cloud

### 1. Unggah ke GitHub

Jadikan **isi folder `upload github/` sebagai root repository**, sehingga `app.py`, `requirements.txt`, dan folder `models/` langsung terlihat di halaman utama repository.

Bobot CRNN berukuran sekitar 63 MB. GitHub Web hanya menerima berkas sampai sekitar 25 MB, sehingga unggah repository menggunakan Git CLI, bukan tombol **Upload files** di browser. Ukuran setiap model masih di bawah batas 100 MB GitHub untuk push Git biasa.

Contoh:

```bash
cd "upload github"
git init
git add .
git commit -m "Deploy OCR Aksara Jawa"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPOSITORY.git
git push -u origin main
```

### 2. Buat Aplikasi Streamlit

1. Buka [Streamlit Community Cloud](https://share.streamlit.io/).
2. Pilih **Create app**.
3. Pilih repository dan branch `main`.
4. Isi **Main file path** dengan `app.py`.
5. Gunakan Python 3.11 atau 3.12 jika tersedia pada pengaturan lanjutan.
6. Klik **Deploy**.

Aplikasi tidak memerlukan `secrets.toml`, API key, database, atau layanan eksternal.

### 3. Isi Tautan Badge

Setelah URL Streamlit diperoleh, ubah bagian berikut:

```markdown
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)]()
```

menjadi:

```markdown
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)](https://nama-aplikasi.streamlit.app/)
```

---

## Cara Penggunaan

1. Pilih **Upload** dan masukkan gambar berformat JPG, PNG, BMP, atau TIF.
2. Atur confidence deteksi jika diperlukan. Nilai evaluasi resmi adalah 0,25.
3. Pilih **Proses OCR**.
4. Periksa transliterasi, urutan baca, detail deteksi, dan crop kata.
5. Unduh teks, CSV, atau gambar visualisasi bila diperlukan.

---

## Konfigurasi

Konfigurasi deployment disimpan pada `deployment_config.json` dan diverifikasi saat model dimuat.

| Parameter | Nilai default |
|---|---:|
| Confidence threshold | 0,25 |
| NMS IoU | 0,70 |
| Image size YOLOv8n | 640 |
| Crop margin | 0,00 |
| CRNN preprocessing | `direct_resize` |
| Reading order | `baseline_center_v2` |
| Line center tolerance | 0,60 |

Checksum model diperiksa sebelum inference. Deployment akan berhenti dengan pesan kesalahan apabila bobot hilang, rusak, atau tidak sesuai konfigurasi.

---

## Ruang Lingkup

- Model detector dilatih pada **kata target yang dianotasi secara terpilih**, bukan seluruh kata dalam setiap halaman.
- Aplikasi menerima citra halaman penuh, tetapi output tidak boleh diklaim sebagai transkripsi exhaustif seluruh halaman.
- Output CRNN berupa transliterasi Latin langsung, bukan Unicode aksara Jawa dan bukan terjemahan bahasa Indonesia.
- Reading order V2 membantu menyusun hasil, tetapi koreksi manusia masih diperlukan untuk bounding box ambigu atau merged crop.
- Aplikasi merupakan prototipe eksperimen jurnal dan belum dirancang sebagai sistem produksi multi-user.

---

## Lisensi dan Atribusi

Pastikan hak penggunaan citra manuskrip, bobot model, dan dataset telah diperiksa sebelum repository dibuat publik. YOLOv8 dijalankan melalui library Ultralytics; ketentuan lisensi Ultralytics juga perlu diperhatikan untuk bentuk distribusi yang dipilih.
