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

## Shortcut Windows

Semua fungsi (server, proses gambar, build extension) terpusat pada launcher tunggal:

| Keperluan | Perintah |
| --- | --- |
| **Control Center TUI (Rekomendasi)** | `scripts\run-tui.bat` |
| Jalankan Server langsung | `scripts\run-tui.bat --server` |
| Proses gambar lokal langsung | `scripts\run-tui.bat --process-images` |
| Build extension langsung | `scripts\run-tui.bat --build-ext` |

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

Masukkan JPEG, PNG, atau WebP ke `api/Images/original Images`. Contoh memproses satu gambar:

```powershell
scripts\run-tui.bat --process-images --limit 1
```

Gunakan opsi `--image "nama.jpg"` untuk memilih file tertentu:

```powershell
scripts\run-tui.bat --process-images --image "001.jpg"
```

*(Atau buka `scripts\run-tui.bat` dan tekan tombol **`2`**).*

Hasil proses akan tersimpan di `api/Images/build Images`: `Box`, `Text Box`, `OCR`, `translate`, `inpainting`, dan `render`.

## Pengaturan API

Edit file `api/.env`:

| Pengaturan | Fungsi |
| --- | --- |
| `API_HOST` | `127.0.0.1` agar API hanya bisa diakses dari PC server; `0.0.0.0` agar perangkat lain di jaringan lokal dapat mengakses |
| `API_PORT` | Port server lokal, default `8000` |
| `API_RELOAD` | `True` agar server otomatis restart saat kode berubah (development) |
| `LLM_API_KEY` | Kunci API layanan LLM; kosongkan jika menggunakan Google Translate atau Ollama lokal |
| `LLM_BASE_URL` | Alamat endpoint LLM, misalnya `http://127.0.0.1:11434/v1` untuk Ollama pada PC yang sama |
| `LLM_MODEL` | Model ID yang digunakan, contoh: `qwen2.5:3b` atau `gpt-4o-mini` |
| `LLM_TIMEOUT_SECONDS` | Batas waktu tunggu respons LLM dalam detik (default: `30.0`) |
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
