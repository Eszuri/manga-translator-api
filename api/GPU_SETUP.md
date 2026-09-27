# Menjalankan detektor dengan NVIDIA GPU (Windows)

Terverifikasi pada GTX 1660 Ti, driver 581.57, Python 3.11.15:
profil model mencatat `DmlExecutionProvider`, halaman 018 membutuhkan 0,160 detik
pada inferensi pertama dan 0,108 detik pada inferensi berikutnya. Batch tiga
halaman mencatat 120–176 ms per deteksi; 32 tes proyek lulus. Waktu dapat berbeda
tergantung gambar dan beban perangkat.

Jalankan PowerShell dari direktori utama proyek. Setup memakai virtual environment
agar paket Python aplikasi lain tidak berubah. Gunakan Python 3.11 yang terpasang.

```powershell
py -3.11 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r api\requirements-gpu.txt
.\.venv-gpu\Scripts\python.exe api\check_gpu.py
```

Jika launcher `py -3.11` tidak menemukan Python, jalankan perintah pertama dengan
path lengkap ke executable Python 3.11 (`py -0p` menampilkan daftar instalasi).

Setup utama menggunakan ONNX Runtime DirectML 1.24.4 pada Windows. DirectML
menjalankan model melalui GPU DirectX 12 (termasuk GTX 1660 Ti), tanpa unduhan
runtime CUDA/cuDNN. Log provider harus menampilkan `DmlExecutionProvider`.

`check_gpu.py` memeriksa provider aktif, menjalankan halaman 018 dua kali, dan
memastikan profil mencatat operator yang dieksekusi oleh GPU. Provider yang
sekadar muncul di `get_available_providers()` belum membuktikan GPU bekerja.
Beberapa operator pendukung dapat tetap berjalan pada CPU.

Untuk menjalankan batch:

```powershell
Set-Location api
..\.venv-gpu\Scripts\python.exe batch_test_folder.py --limit 30 --detector comic_text_detector --require-gpu
```

Gunakan executable di `.venv-gpu`, bukan `python` global yang masih memakai ONNX
Runtime CPU. Opsi `--require-gpu` menolak sesi tanpa GPU dan menonaktifkan
pengulangan inferensi otomatis dengan provider CPU saat GPU gagal.

Untuk API:

```powershell
..\.venv-gpu\Scripts\python.exe run_server.py
```

Pada request API, pilih `detector_type=comic_text_detector`; default API masih
`contour`. Jangan memasang `onnxruntime`, `onnxruntime-directml`, dan
`onnxruntime-gpu` bersama dalam satu environment: ketiganya menyediakan modul
Python yang sama.

## Alternatif: NVIDIA CUDA

Jalur ini membutuhkan unduhan pustaka NVIDIA sekitar 2 GB. Dari direktori utama
proyek, buat environment terpisah:

```powershell
py -3.11 -m venv .venv-cuda
.\.venv-cuda\Scripts\python.exe -m pip install -r api\requirements-cuda.txt
.\.venv-cuda\Scripts\python.exe api\check_gpu.py
```

Paket `[cuda,cudnn]` memasang runtime CUDA 12 dan cuDNN 9. Detektor memanggil
`onnxruntime.preload_dlls()` sebelum membuat sesi. CUDA Toolkit lengkap dan
PyTorch tidak diperlukan. Untuk jalur ini, log harus menunjukkan
`CUDAExecutionProvider` dan pemeriksaan profil harus lulus. Konfigurasi CUDA
disediakan sebagai alternatif; instalasi dan uji CUDA belum diselesaikan.

Referensi resmi:
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
