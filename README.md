# Manga Translator

Backend FastAPI dan extension browser untuk mendeteksi teks manga, menjalankan
OCR Jepang, menerjemahkan dialog, menghapus teks asli, dan render teks hasil
terjemahan pada gambar. <b>Project ini harus memiliki GPU pada komputer</b>

## Struktur project

```text
api/
  app/
    api/v1/              Endpoint dan validasi upload
    core/                Konfigurasi dan kebijakan GPU
    services/            Deteksi, OCR, terjemahan, inpainting, dan penataan teks
    assets/fonts/        Font untuk hasil render
    models/              Model lokal
  tools/                 Runner server dan builder gambar
  requirements/          Dependency global, lokal, dan Docker
  Images/
    original Images/     Gambar manga asli
    build Images/        Hasil build manga translator
  .env.example           sample environment
  .env                   konfigurasi API, dan kawan kawan
  Dockerfile             Image aplikasi
  Dockerfile.runtime     Image dependency GPU
extension/               Source extension browser
assets/branding/         icon extension
docs/                    Dokumentasi
scripts/
  tasks/                 Builder package extension
  windows/
    common/              Pemeriksaan Docker Engine
    build/               Shortcut build gambar dan extension
    run/                 Shortcut server lokal dan Docker
compose.yaml             Deployment GPU
compose.dev.yaml         Override development dengan auto-reload
compose.runtime.yaml     Build runtime GPU
dist/                    ZIP, CRX, dan kunci extension
```

## Shortcut Windows


| Keperluan | File |
| --- | --- |
| Build gambar lokal | `scripts\windows\build\local-images.bat` |
| Build extension | `scripts\windows\build\package-extension.bat` |
| Server lokal Windows | `scripts\windows\run\local-server.bat` |
| Docker development (tanpa build, auto-reload) | `scripts\windows\run\docker-development.bat` |
| Build aplikasi dan jalankan Docker development | `scripts\windows\run\docker-development-rebuild.bat` |
| Build runtime GPU Docker | `scripts\windows\run\docker-runtime-rebuild.bat` |
| Build aplikasi dan jalankan Docker deployment | `scripts\windows\run\docker-deployment.bat` |
| Hentikan dan hapus container project | `scripts\windows\run\docker-stop.bat` |

## Menjalankan backend

Siapkan `api/.env` dan model terlebih dahulu. Tanpa Docker, setelah environment
`.venv-gpu` tersedia:

```powershell
scripts\windows\run\local-server.bat
```

Dengan Docker, untuk instalasi pertama atau build aplikasi:

```powershell
scripts\windows\run\docker-development-rebuild.bat
```

Script tersebut membangun runtime hanya jika image runtime belum tersedia.
Jika dependency runtime berubah, build ulang runtime secara terpisah.

Periksa API:

```powershell
/api/v1/health
```

Dokumentasi endpoint tersedia di `/docs`.
Server lokal dan Docker tidak dapat memakai port host yang sama secara bersamaan.

## Memasang extension

1. Buka `chrome://extensions` atau `edge://extensions`.
2. Aktifkan mode pengembang atau mode developer.
3. Pilih load extension yang belum dikemas, lalu pilih folder `extension`.
4. Isi kolom `Server` pada popup, misalnya `http://127.0.0.1:8000`.
5. Pilih Google Translate atau LLM API, bahasa tujuan, dan arah baca.
6. Aktifkan manga translator untuk domain halaman yang sedang dibuka (default false). 



Untuk build extension (.crx / .zip):

```powershell
scripts\windows\build\package-extension.bat
```

Script membutuhkan perintah `python` pada `PATH`.  
CRX terbuild jika browser Chromium (google chrome,dkk) ditemukan.
 Hasil berada di `dist`.

## bagaimana gambar diproses

- Gambar akan diproses ketika sudah terload semua gambar yg ada pada web dan memiliki dimensi asli minimal 500 × 700 piksel.
- Maksimal dua gambar yg diproses oleh manga translator per halaman web.
2 proses tersebut bergantian sampai semua gambar terselesaikan oleh manga translator di halaman web.
- Gambar dalam viewport diprioritaskan. Gambar di luar viewport tetap diproses
  setelah antrean prioritas viewport
- Hasil disimpan dalam IndexedDB browser dan dapat digunakan kembali saat refresh tanpa harus translate ulang.

## pemrosesan GPU

Inferensi model deteksi dan OCR wajib memakai CUDA pada Docker atau DirectML pada Windows lokal.

pipeline yg tidak berjalan di GPU: decoding gambar, pengolahan mask,
inpainting OpenCV Telea, penataan teks, dan beberapa node kontrol/shape ONNX
tetap memakai CPU.

OCR yang digunakan ditujukan untuk teks Jepang. Bahasa selain jepang mungkin tidak berfungsi sepenuhnya




## Persiapan lokal

Gunakan Python 3.11. Jika `.venv-gpu` belum ada:

```powershell
py -3.11 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r api\requirements\local.txt
```

Salin `api/.env.example` ke `api/.env` jika belum ada.
Model di `api/app/models` harus berisi `comic-text-detector.onnx` dan folder
`manga-ocr` dengan encoder, decoder, tokenizer, serta konfigurasi OCR.

## Build gambar lokal

Masukkan JPEG, PNG, atau WebP ke `api/Images/original Images`. Contoh satu gambar pertama:

```powershell
scripts\windows\build\local-images.bat --limit 1
```

Gunakan `--image "nama.jpg"` untuk memilih file tertentu.
Build memakai Google Translate dan arah baca kanan ke kiri.
Hasil ada di `api/Images/build Images`: `Box`, `Text Box`, `OCR`,
`translate`, `inpainting`, dan `render`. Build ulang menimpa hasil dengan nama sama.

## Pengaturan API

Edit `api/.env`:

| Pengaturan | Fungsi |
| --- | --- |
| `API_HOST` | `127.0.0.1` agar API hanya bisa diakses dari PC server; `0.0.0.0` agar perangkat lain juga bisa mengaksesnya |
| `API_PORT` | Port server lokal, default `8000` |
| `API_RELOAD` | `True` agar server lokal restart otomatis saat kode berubah |
| `LLM_API_KEY` | Kunci layanan LLM; Google Translate tidak memerlukannya |
| `LLM_BASE_URL` | Alamat utama layanan LLM. Dipakai jika alamat khusus lokal atau Docker tidak diisi |
| `LLM_BASE_URL_LOCAL` | Alamat layanan LLM saat backend dijalankan tanpa Docker. Jika diisi, alamat ini dipakai sebagai pengganti `LLM_BASE_URL` |
| `LLM_BASE_URL_DOCKER` | Alamat layanan LLM saat backend dijalankan dengan Docker. Jika diisi, alamat ini dipakai sebagai pengganti `LLM_BASE_URL` |
| `LLM_MODEL` | ID model yang tersedia pada layanan |



Contoh Ollama pada PC:

```dotenv
LLM_API_KEY=
LLM_BASE_URL_LOCAL=http://127.0.0.1:11434/v1
LLM_BASE_URL_DOCKER=http://host.docker.internal:11434/v1
LLM_MODEL=model-yang-sudah-tersedia (contoh= qwen2.5:3b)
```

Dalam contoh ini, backend lokal mengakses Ollama melalui `127.0.0.1`.
Backend Docker mengakses Ollama pada PC yang sama melalui `host.docker.internal`.

- Backend lokal memakai `LLM_BASE_URL_LOCAL`; 
  Docker memakai `LLM_BASE_URL_DOCKER`.
  Jika alamat tersebut kosong, backend memakai `LLM_BASE_URL`.
- Kosongkan `LLM_BASE_URL_LOCAL` dan `LLM_BASE_URL_DOCKER` jika keduanya tidak digunakan.
- Kunci boleh kosong untuk host lokal yang dikenali backend, termasuk `host.docker.internal`.

### Akses dari perangkat lain

- Jika perangkat berada di jaringan yang sama, gunakan `http://IP-PC:8000`.
  Ganti `IP-PC` dengan alamat IP komputer yang menjalankan API.
- Jika berada di jaringan berbeda, hubungkan kedua perangkat melalui Tailscale,
  lalu gunakan `http://IP-TAILSCALE-PC:8000`.
- Jangan memakai `127.0.0.1` untuk mengakses PC lain. Alamat ini berarti perangkat
  yang sedang digunakan.

Untuk memeriksa koneksi, buka `http://IP-PC:8000/api/v1/health` dari perangkat lain.
Jika memakai Tailscale, gunakan IP Tailscale PC pada alamat tersebut.
Pastikan firewall PC mengizinkan koneksi ke port API. 

## Setelah mengubah kode atau pengaturan konfigurasi

- Development Docker: perubahan `api/app` langsung reload.
- Deployment: jalankan kembali `docker-deployment.bat` setelah mengubah kode.
- Dependency Docker berubah: build runtime, lalu build aplikasi.
- `.env` berubah: restart server lokal. jika di Docker, buat ulang container:


## Jika terjadi error

Untuk Docker, baca log dengan `docker compose -f compose.yaml logs --tail 100 api`.
Untuk server lokal, baca pesan pada terminal tempat server dijalankan.


| Pesan yang muncul | Arti dan tindakan |
| --- | --- |
| `Docker Engine did not respond within 15 seconds` | Pemeriksaan Docker Engine melewati batas 15 detik. Jalankan `docker info` untuk melihat error koneksinya.
| `Required GPU provider ... is unavailable` atau `GPU-only backend requires ...` | Provider GPU yang dibutuhkan tidak tersedia. Daftarnya ditampilkan setelah `Available providers`. Lokal memerlukan `DmlExecutionProvider` Docker memerlukan `CUDAExecutionProvider` |
| `failed to activate ... Active providers: ...` | Sesi model gagal mengaktifkan provider GPU yang diminta. Baca error pemuatan ONNX sebelumnya pada log.|
| `Missing model files ...` | File yang disebut setelah pesan ini tidak ditemukan pada folder model container. Cocokkan daftar tersebut dengan file di `api/app/models` |
| HTTP `422` | API menolak parameter. `detail.loc` menunjukkan parameter yang salah dan `detail.msg` menjelaskan alasannya. Contoh: `body.file: Field required` berarti file tidak terkirim; `body.device: Input should be 'gpu'` berarti nilai perangkat ditolak |
| `LLM is not configured` | Pemeriksaan konfigurasi LLM gagal. Untuk layanan yang membutuhkan kunci, isi `LLM_API_KEY`. atau isi .env ada yg salah|
| `Cannot connect to the LLM endpoint` | Backend gagal membuat koneksi ke layanan LLM. Uji alamat yang benar-benar dipakai: `LLM_BASE_URL_LOCAL`, `LLM_BASE_URL_DOCKER`, atau `LLM_BASE_URL` jika alamat khusus kosong |
| `LLM request timed out after ...s` | Panggilan LLM melewati batas waktu yang diatur oleh `LLM_TIMEOUT_SECONDS`. Ini berbeda dari timeout extension |
| `LLM did not return a valid translation for bubble ...` | Hasil LLM tidak memuat terjemahan yang tidak kosong untuk ID dialog tersebut. Respons harus berisi ID dan teks terjemahan untuk setiap dialog atau kemungkinan menggunakan gambar yg bukan bahasa jepang|
| `Google Translate failed (HTTP 429)` | Google membatasi permintaan. Ini bukan error API key LLM |
| `Google Translate returned an unexpected number of results` | Jumlah hasil Google tidak sama dengan jumlah dialog yang dikirim. Backend menolak hasil tersebut agar dialog tidak tertukar |
| `Request timed out ...` pada extension | Extension membatalkan request setelah 30 detik, atau 60 detik untuk stream. Pesan ini tidak membuktikan server mati atau sibuk; lihat log backend untuk mengetahui tahap terakhir |

Pada respons stream, `stage=error` berarti proses gagal meskipun HTTP tetap 200.


<center><b>Selamat Mencoba</b></center>

## GUI server Windows

- Jalankan dari source: `scripts\windows\run\local-server-gui.bat`.
- Build paket Windows: `scripts\windows\build\package-server-gui.bat`.
- Hasil: `dist/MangaTranslatorServer/MangaTranslatorServer.exe`. Folder `_internal` harus tetap berada di sebelah executable.

Paket menyertakan Python dan dependency server; tidak membutuhkan instalasi Python atau Docker pada PC tujuan.
Model ONNX, gambar, `.env`, dan kunci API tidak ikut dipaketkan. Model tetap diperlukan:
pilih folder `api/app/models` yang sudah ada melalui **Settings**, atau salin isinya ke folder `models` di sebelah executable.
Model tidak diunduh otomatis. GPU dengan dukungan DirectX 12 dan driver yang sesuai tetap diperlukan.

GUI dibuka dengan server berhenti. Tekan **Start server** untuk menjalankannya. **Stop** saat pemuatan model membatalkan startup; **Force stop** menghentikan proses yang belum selesai ditutup. Tombol hanya mengontrol server milik GUI.
Jika port sudah dipakai server lokal atau Docker, hentikan server tersebut atau ubah **API port** pada GUI.

GUI tidak membaca atau mengubah `api/.env`. Atur LLM dan akses perangkat lain melalui **Settings**.
Google Translate memerlukan internet; LLM memerlukan layanan dan model yang sudah tersedia.
Pengaturan dan log disimpan di `%LOCALAPPDATA%\MangaTranslatorServer`; kunci API dilindungi Windows DPAPI.
Akses jaringan mati secara default. Aktifkan **Network** jika diperlukan; izin firewall tetap diatur sendiri.
