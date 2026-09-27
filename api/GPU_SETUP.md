# Running the Detector with GPU (Windows)

Verified on NVIDIA GTX 1660 Ti, driver 581.57, Python 3.11:
Model execution runs on `DmlExecutionProvider`, processing page `018.jpg` in ~0.160s on the first inference and ~0.110s on subsequent warm runs. Batch tests process pages in ~120–250ms per detection.

Run PowerShell from the root project directory. A virtual environment is used to keep dependencies isolated:

```powershell
py -3.11 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r api\requirements-gpu.txt
.\.venv-gpu\Scripts\python.exe api\check_gpu.py
```

If `py -3.11` does not locate Python, provide the full path to your Python 3.11 executable (`py -0p` lists available installations).

The primary setup utilizes **ONNX Runtime DirectML 1.24.4** on Windows. DirectML executes models via DirectX 12 hardware acceleration (compatible with modern AMD, Intel, and NVIDIA GPUs) without requiring CUDA/cuDNN toolkit downloads. Active providers must include `DmlExecutionProvider`.

`check_gpu.py` validates active providers, executes warm-up inference, and confirms GPU execution without fallback.

### Running Batch Tests:

```powershell
Set-Location api
# Batch folder test
.\test_batch_gpu.bat --limit 30

# Manga OCR batch test
.\test_ocr_gpu.bat --limit 30
```

### Running the API Server:

```powershell
Set-Location api
..\.venv-gpu\Scripts\python.exe run_server.py
```

API requests default to `detector_type=hybrid`. Do not install `onnxruntime`, `onnxruntime-directml`, and `onnxruntime-gpu` simultaneously in the same environment: all three provide the same `onnxruntime` Python package namespace.

---

## Alternative: NVIDIA CUDA

This setup requires downloading ~2 GB of NVIDIA CUDA 12 packages. From the project root, create a separate virtual environment:

```powershell
py -3.11 -m venv .venv-cuda
.\.venv-cuda\Scripts\python.exe -m pip install -r api\requirements-cuda.txt
.\.venv-cuda\Scripts\python.exe api\check_gpu.py
```

The `[cuda,cudnn]` package bundles the CUDA 12 and cuDNN 9 runtime libraries. The detector calls `onnxruntime.preload_dlls()` prior to initializing sessions. Full CUDA Toolkit and PyTorch installations are not required.

Official References:
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
