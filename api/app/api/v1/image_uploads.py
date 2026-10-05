"""Shared validation for image uploads accepted by API endpoints."""

from io import BytesIO
import warnings

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.core.config import settings


ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    # Some browser-extension clients omit a precise MIME type. Image decoding below
    # remains authoritative, so accepting this does not permit arbitrary files.
    "application/octet-stream",
}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


async def read_validated_image(file: UploadFile) -> Image.Image:
    """Read, verify, and decode a supported image within configured limits."""
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported upload content type. Use JPEG, PNG, or WebP.",
        )

    contents = await file.read(settings.MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty.",
        )
    if len(contents) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Uploaded image exceeds the configured size limit.",
        )

    # Decoding does not use ONNX; it must not wait behind a long GPU task before
    # a streaming response can start and send queue/processing heartbeats.
    return await run_in_threadpool(decode_validated_image, contents)


def decode_validated_image(contents: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            probe = Image.open(BytesIO(contents))
            width, height = probe.size
            if width <= 0 or height <= 0 or width * height > settings.MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="Uploaded image exceeds the configured pixel limit.",
                )
            if probe.format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Unsupported image format. Use JPEG, PNG, or WebP.",
                )
            probe.verify()

            image = Image.open(BytesIO(contents))
            image.load()
            return image
    except HTTPException:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image.",
        )
