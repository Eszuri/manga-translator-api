# Manga Translator

Backend FastAPI dan extension browser untuk mendeteksi teks manga, menjalankan
OCR Jepang, menerjemahkan dialog, menghapus teks asli, dan render teks hasil
terjemahan pada gambar. <b>Project ini harus memiliki GPU pada komputer</b>.

## Struktur Project

```text
api/
  app/
    api/v1/              Endpoint dan validasi upload
    core/                Konfigurasi dan kebijakan GPU
    services/            Deteksi, OCR, terjemahan, inpainting, dan penataan teks
    assets/fonts/        Font untuk hasil render
  tools/                 Runner server dan builder gambar lokal
  requirements/          Dependency server lokal
  Images/
    original Images/     Gambar manga asli
    build Images/        Hasil build manga translator
  .env.example           sample environment
  .env                   konfigurasi API
extension/               Source extension browser
models/                  Model deteksi dan OCR lokal
assets/branding/         Icon extension
docs/                    Dokumentasi
scripts/
  manga_tui.py         Control Center TUI & unified runner Python
  run-tui.bat          Launcher tunggal Windows
dist/                    Paket hasil build extension
```

## Shortcut Windows & Argumen CLI

Semua fungsi (server, proses gambar, build extension) terpusat pada launcher tunggal:

| Keperluan | Perintah | Keterangan |
| --- | --- | --- |
| **Control Center TUI (Rekomendasi)** | `scripts\run-tui.bat` | Membuka antarmuka Textual interaktif |
| Jalankan Server langsung | `scripts\run-tui.bat --server` | Menjalankan backend server FastAPI Uvicorn |
| Proses gambar lokal langsung | `scripts\run-tui.bat --process-images` | Menjalankan pipeline pemrosesan gambar lokal |
| Build extension langsung | `scripts\run-tui.bat --build-ext` | Mengemas browser extension ke ZIP dan CRX |
| Bantuan | `scripts\run-tui.bat --help` | Menampilkan daftar argumen CLI yang tersedia |

## Menjalankan Backend

Siapkan `api/.env`, model, dan environment `.venv` terlebih dahulu. Buka TUI:

```powershell
scripts\run-tui.bat
```
*(Lalu tekan tombol **`1`** untuk memulai server, atau jalankan langsung `scripts\run-tui.bat --server`)*


Periksa API:

```powershell
http://127.0.0.1:8000/api/v1/health
```

Dokumentasi endpoint interaktif tersedia di `http://127.0.0.1:8000/docs`.

## Memasang Extension Browser

1. Buka `chrome://extensions` atau `edge://extensions`.
2. Aktifkan mode pengembang (*Developer mode*).
3. Pilih **Load unpacked** (muat ekstensi yang belum dikemas), lalu pilih folder `extension`.
4. Isi kolom `Server` pada popup, misalnya `http://127.0.0.1:8000`.
5. Pilih Google Translate atau LLM API, bahasa tujuan, dan arah baca.
6. Aktifkan manga translator untuk domain halaman yang sedang dibuka.

Untuk mem-build extension (.crx / .zip):

```powershell
scripts\run-tui.bat --build-ext
```

*(Atau buka `scripts\run-tui.bat` dan tekan tombol **`3`**).*
Hasil paket (`.zip` dan `.crx`) otomatis dibuat di folder `dist`.

## Bagaimana Gambar Diproses

- Gambar akan diproses ketika semua gambar pada web telah selesai dimuat dan memiliki dimensi asli minimal 500 × 700 piksel.
- Maksimal dua gambar yang diproses secara paralel per halaman web.
- Proses dilakukan bergantian sampai semua gambar pada halaman terselesaikan.
- Gambar dalam viewport diprioritaskan. Gambar di luar viewport tetap diproses setelah antrean prioritas viewport selesai.
- Hasil disimpan dalam IndexedDB browser dan dapat digunakan kembali saat refresh tanpa harus translate ulang.

## Pemrosesan GPU

Inferensi model deteksi dan OCR wajib memakai DirectML (`DmlExecutionProvider`) atau CUDA (`CUDAExecutionProvider`) pada Windows lokal.

Pipeline yang berjalan di CPU:
- Decoding gambar
- Pengolahan mask geometri
- Inpainting OpenCV Telea
- Penataan teks (typesetting)
- Node kontrol/shape ONNX tertentu

> **Catatan:** OCR yang digunakan ditujukan untuk teks Jepang. Bahasa selain Jepang tidak didukung secara optimal.

## Persiapan Lingkungan Lokal

Gunakan Python 3.11. Jika `.venv` belum ada:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r api\requirements\local.txt
```

Salin `api/.env.example` ke `api/.env` jika belum ada:

```powershell
copy api\.env.example api\.env
```

Folder `models` harus berisi:
- `comic-text-detector.onnx`
- Folder `manga-ocr` berisi encoder, decoder, tokenizer, serta file konfigurasi model.

## Build Gambar Lokal

Masukkan JPEG, PNG, atau WebP ke `api/Images/original Images`. Untuk memproses gambar:

```powershell
scripts\run-tui.bat --process-images
```

*(Atau buka `scripts\run-tui.bat` dan tekan tombol **`2`**).*

Hasil proses akan tersimpan di `api/Images/build Images`: `Box`, `Text Box`, `OCR`, `translate`, `inpainting`, dan `render`.

### Argumen CLI `tools.build_local`

Selain melalui TUI, builder gambar dapat dijalankan langsung menggunakan perintah Python dari folder `api` (`python -m tools.build_local [ARGUMEN]`):

| Argumen | Tipe / Pilihan | Nilai Bawaan | Deskripsi |
| --- | --- | --- | --- |
| `--input-dir` | Path | `Images/original Images` | Path folder gambar manga masukan |
| `--output-dir` | Path | `Images/build Images` | Path folder penyimpanan hasil tahapan proses |
| `--image` | String | *(semua)* | Memproses satu nama file tertentu saja (contoh: `009.jpg`) |
| `--limit` | Integer | `0` | Batas jumlah halaman yang diproses (`0` = proses semua halaman) |
| `--target-lang` | `id`, `en` | `id` | Bahasa target terjemahan (`id` = Indonesia, `en` = Inggris) |
| `--translator` | `google`, `llm` | `google` | Mesin penerjemah; `llm` memakai konfigurasi `api/.env` |
| `--font-scale` | Float | `1.0` | Pengali ukuran font teks pada hasil akhir render |

> **Catatan:** Komputasi perangkat otomatis di-hardcode ke **GPU** dan deteksi balon teks otomatis menggunakan model **Hybrid** (Comic Text Detector + Segmentasi Balon).

## Pengaturan API

Edit file `api/.env`:

LLM membaca seluruh OCR satu halaman bersama, menerjemahkan, lalu memeriksa makna dan pembagian teks per balon. Proses memakai dua panggilan LLM; respons tidak valid dapat diulang dalam batas waktu yang sama.

| Pengaturan | Fungsi |
| --- | --- |
| `API_HOST` | `127.0.0.1` agar API hanya bisa diakses dari PC server; `0.0.0.0` agar perangkat lain di jaringan lokal dapat mengakses |
| `API_PORT` | Port server lokal, default `8000` |
| `API_RELOAD` | `True` agar server otomatis restart saat kode berubah (development) |
| `LLM_API_KEY` | Kunci API layanan LLM; kosongkan jika menggunakan Google Translate atau Ollama lokal |
| `LLM_BASE_URL` | Alamat endpoint LLM, misalnya `http://127.0.0.1:11434/v1` untuk Ollama pada PC yang sama |
| `LLM_MODEL` | Model ID yang digunakan, contoh: `qwen2.5:3b` atau `gpt-4o-mini` |
| `LLM_TIMEOUT_SECONDS` | Batas waktu total terjemahan dan pemeriksaan LLM per halaman, termasuk retry (default: `90.0` detik) |
| `DEFAULT_TRANSLATOR` | Penerjemah bawaan: `google` atau `llm` |

Contoh Ollama lokal:

```dotenv
LLM_API_KEY=
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=qwen2.5:3b
DEFAULT_TRANSLATOR=llm
```

### Akses dari Perangkat Lain (LAN / Tailscale)

- Jika perangkat berada di jaringan Wi-Fi/LAN yang sama: gunakan `http://IP-PC:8000` (atur `API_HOST=0.0.0.0`).
- Jika berada di jaringan berbeda: hubungkan kedua perangkat melalui Tailscale, lalu gunakan `http://IP-TAILSCALE-PC:8000`.
- Pastikan Windows Firewall mengizinkan koneksi masuk pada port yang ditentukan (default `8000`).

## Parameter API Endpoint

Dokumentasi OpenAPI interaktif lengkap tersedia di `http://127.0.0.1:8000/docs`. Seluruh endpoint otomatis menggunakan akselerasi **GPU** dan deteksi **Hybrid**. Berikut rincian parameter yang tersedia untuk setiap endpoint:

### 1. `POST /api/v1/detect/bubbles`
Mendeteksi balon percakapan dan region teks pada satu halaman manga.
- **Content-Type**: `multipart/form-data`
- **Parameter Form**:
  - `file` *(UploadFile, Wajib)*: File gambar manga (JPEG, PNG, WebP).
  - `reading_direction` *(String, Opsional)*: Arah urutan baca, `'rtl'` (Manga Jepang) atau `'ltr'` (Manhwa Korea). Nilai bawaan: `'rtl'`.

### 2. `POST /api/v1/ocr/recognize`
Mendeteksi balon percakapan dan mengekstrak teks Jepang seluruh halaman menggunakan Manga-OCR.
- **Content-Type**: `multipart/form-data`
- **Parameter Form**:
  - `file` *(UploadFile, Wajib)*: File gambar manga halaman penuh.
  - `reading_direction` *(String, Opsional)*: `'rtl'` atau `'ltr'`. Nilai bawaan: `'rtl'`.

### 3. `POST /api/v1/ocr/recognize-crop`
Mengekstrak teks Jepang dari satu potongan gambar kecil (hasil seleksi kotak manual dari extension).
- **Content-Type**: `multipart/form-data`
- **Parameter Form**:
  - `file` *(UploadFile, Wajib)*: File gambar potongan teks.

### 4. `POST /api/v1/translate/page`
Pipeline terjemahan penuh (Deteksi &rarr; OCR &rarr; Terjemahan teks dialog).
- **Content-Type**: `multipart/form-data`
- **Parameter Form**:
  - `file` *(UploadFile, Wajib)*: File gambar manga.
  - `target_lang` *(String, Opsional)*: Kode bahasa target (`'id'` atau `'en'`). Nilai bawaan: `'id'`.
  - `reading_direction` *(String, Opsional)*: `'rtl'` atau `'ltr'`. Nilai bawaan: `'rtl'`.

### 5. `POST /api/v1/translate/dialogues`
Menerjemahkan array dialog terstruktur yang telah diekstrak sebelumnya melalui LLM.
- **Content-Type**: `application/json`
- **Body JSON**:
  - `target_lang` *(String, Wajib)*: Kode bahasa tujuan (contoh: `"id"`).
  - `dialogues` *(Array of Object, Wajib)*: Daftar item dialog `[{"id": 1, "text": "..."}]`.

### 6. `POST /api/v1/translate/inpaint-page`
Pipeline end-to-end lengkap dari deteksi, ekstraksi OCR, terjemahan, pembersihan teks asli (inpainting), hingga penataan font komik baru (typesetting).
- **Content-Type**: `multipart/form-data`
- **Parameter Form**:
  - `file` *(UploadFile, Wajib)*: File gambar manga.
  - `target_lang` *(String, Opsional)*: Kode bahasa target (`'id'` atau `'en'`). Nilai bawaan: `'id'`.
  - `translator` *(String, Opsional)*: Mesin penerjemah: `'llm'` atau `'google'`. Nilai bawaan dari konfigurasi `.env`.
  - `reading_direction` *(String, Opsional)*: `'rtl'` atau `'ltr'`. Nilai bawaan: `'rtl'`.
  - `typeset` *(Boolean, Opsional)*: `true` untuk render teks hasil terjemahan ke balon, `false` untuk hasil inpainting bersih tanpa teks baru. Nilai bawaan: `true`.
  - `font_scale` *(Float, Opsional)*: Pengali ukuran font teks. Nilai bawaan: `1.0`.
  - `all_caps` *(Boolean, Opsional)*: Format huruf kapital dialog komik. Nilai bawaan: `true`.
  - `return_format` *(String, Opsional)*: Format balasan: `'image'` (binary JPEG langsung) atau `'json'` (base64 image + metadata balon teks). Nilai bawaan: `'image'`.

### 7. `POST /api/v1/translate/inpaint-stream`
Menjalankan pipeline inpainting & typesetting dengan streaming progres per tahapan secara *real-time* via NDJSON.
- **Content-Type**: `multipart/form-data`
- **Format Respon**: `application/x-ndjson` (event tahapan: `detect`, `ocr`, `translate`, `inpaint`, `render`, `done`)
- **Parameter Form**:
  - `file` *(UploadFile, Wajib)*: File gambar manga.
  - `target_lang` *(String, Opsional)*: Nilai bawaan `'id'`.
  - `translator` *(String, Opsional)*: `'llm'` atau `'google'`.
  - `reading_direction` *(String, Opsional)*: `'rtl'` atau `'ltr'`.
  - `typeset` *(Boolean, Opsional)*: Nilai bawaan `true`.
  - `font_scale` *(Float, Opsional)*: Nilai bawaan `1.0`.
  - `all_caps` *(Boolean, Opsional)*: Nilai bawaan `true`.

## Troubleshooting / Pesan Error

Baca log pada terminal tempat server dijalankan:

| Pesan yang Muncul | Arti dan Solusi |
| --- | --- |
| `Required GPU provider ... is unavailable` | Provider GPU tidak ditemukan. Pastikan GPU mendukung DirectX 12 dan paket `onnxruntime-directml` terinstal. |
| `failed to activate ... Active providers: ...` | Sesi model ONNX gagal mengaktifkan GPU execution provider. Periksa log startup terminal. |
| `Missing model files ...` | File model ONNX di folder `models/` belum lengkap atau kosong. |
| HTTP `422` | Permintaan ditolak karena validasi parameter gagal. Periksa kembali form data yang dikirimkan. |
| `LLM is not configured` | Kunci API kosong untuk provider remote. Isi `LLM_API_KEY` di `api/.env`. |
| `Cannot connect to the LLM endpoint` | Server gagal menghubungi `LLM_BASE_URL`. Pastikan instance Ollama/LLM sudah menyala. |
| `LLM request timed out after ...s` | Panggilan LLM melewati batas waktu. Naikkan `LLM_TIMEOUT_SECONDS` di `api/.env`. |
| `LLM did not return a valid translation for bubble ...` | LLM tidak mengembalikan terjemahan untuk ID balon tersebut. Coba gunakan Google Translate atau model LLM lain. |
| `Google Translate failed (HTTP 429)` | Google membatasi jumlah request (rate limit). Tunggu beberapa saat atau beralih ke LLM. |
| `Request timed out ...` pada extension | Extension membatalkan request setelah timeout. Periksa beban GPU di log server. |
