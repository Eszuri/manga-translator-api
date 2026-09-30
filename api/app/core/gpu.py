"""Strict GPU execution policy for ONNX Runtime model inference."""
import os
from typing import List

import onnxruntime as ort


GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")


def required_gpu_provider() -> str:
    """Return the configured GPU provider or the best available provider."""
    configured = os.getenv("GPU_PROVIDER", "").strip()
    available = ort.get_available_providers()

    if configured:
        if configured not in GPU_PROVIDERS:
            raise RuntimeError(
                f"GPU_PROVIDER must be one of {GPU_PROVIDERS}, got {configured!r}"
            )
        if configured not in available:
            raise RuntimeError(
                f"Required GPU provider {configured!r} is unavailable. "
                f"Available providers: {available}"
            )
        return configured

    for provider in GPU_PROVIDERS:
        if provider in available:
            return provider

    raise RuntimeError(
        "GPU-only backend requires CUDAExecutionProvider or DmlExecutionProvider. "
        f"Available providers: {available}"
    )


def configure_gpu_session(options: ort.SessionOptions) -> List[str]:
    """Configure an ONNX session with a required GPU execution provider.

    ONNX Runtime keeps a small number of shape/control nodes on its CPU provider
    by design. Runtime provider failure is still disabled after session creation,
    so CUDA/DirectML compute cannot silently fall back to CPU inference.
    """
    provider = required_gpu_provider()

    if provider == "CUDAExecutionProvider" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    elif provider == "DmlExecutionProvider":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    return [provider]


def verify_gpu_session(session: ort.InferenceSession, component: str) -> str:
    """Disable runtime fallback and verify that the requested GPU EP is active."""
    provider = required_gpu_provider()
    active = session.get_providers()
    if provider not in active:
        raise RuntimeError(
            f"{component} failed to activate {provider}. Active providers: {active}"
        )
    session.disable_fallback()
    return provider


def require_gpu_device(device: str) -> None:
    """Reject legacy CPU/auto request values before any processing begins."""
    if device.lower() != "gpu":
        raise ValueError("Backend ini GPU-only; parameter device wajib 'gpu'.")
