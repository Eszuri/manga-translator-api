"""One end-to-end visual test of the manga pipeline (Google translation only).

The production API continues to use app.services.translation_service.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from PIL import Image, ImageDraw, ImageFont, ImageOps

from app.services.comic_text_detector import ComicTextDetector
from app.services.detector import (
    ContourBubbleDetector,
    annotate_and_save_bubbles,
    sort_manga_reading_order,
)
from app.services.hybrid_detector import HybridBubbleDetector
from app.services.inpainting_service import MangaInpaintingService
from app.services.ocr_service import MangaOcrService
from app.services.translation_filters import is_graphic_text, usable_translation
from app.services.typesetting_service import MangaTypesettingService


BASE_DIR = Path(__file__).resolve().parent
STAGES = (
    "01_box",
    "02_text_box",
    "03_ocr",
    "04_translate",
    "05_inpainting",
    "06_render",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class GoogleTestTranslationService:
    """Google Translate is confined to this visual test; production uses its API."""

    async def _request(self, texts: list[str], target_lang: str) -> list[str]:
        import httpx
        from googletrans import Translator

        for attempt in range(3):
            try:
                async with Translator(raise_exception=True, timeout=httpx.Timeout(20.0),
                                      list_operation_max_concurrency=2) as translator:
                    result = await translator.translate(texts, src="ja", dest=target_lang)
                if not isinstance(result, list):
                    result = [result]
                values = [str(item.text).strip() for item in result]
                if len(values) != len(texts) or not all(values):
                    raise ValueError("Google Translate returned incomplete results")
                return values
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"Google Translate failed: {exc}") from exc
                await asyncio.sleep(min(4, 2 ** attempt))
        raise AssertionError("unreachable")

    def translate_bubbles(self, bubbles, target_lang: str):
        pending = []
        for bubble in bubbles:
            original = (bubble.text or "").strip()
            compact = re.sub(r"\s+", "", original)
            if re.fullmatch(r"[.．…・·｡。⋯･]+", compact):
                bubble.translation = "..."
            elif re.fullmatch(r"[!！]+", compact):
                bubble.translation = "!"
            elif re.fullmatch(r"[?？]+", compact):
                bubble.translation = "?"
            elif original:
                pending.append(bubble)
        if pending:
            translated = asyncio.run(self._request(
                [bubble.text.strip() for bubble in pending], target_lang))
            for bubble, value in zip(pending, translated):
                value = unicodedata.normalize("NFKC", value).strip()
                value = re.sub(r"(?:\s*\.){2,}", "...", value)
                bubble.translation = re.sub(r"\s+([!?.,])", r"\1", value)
        return bubbles


def parse_args():
    parser = argparse.ArgumentParser(description="Single end-to-end manga visual test")
    parser.add_argument("--dir", type=Path, default=Path("image test"))
    parser.add_argument("--output-root", type=Path, default=Path("image test/output_pipeline"))
    parser.add_argument("--image", help="One input filename, e.g. 009.jpg")
    parser.add_argument("--limit", type=int, default=0, help="Pages to process; 0 means all")
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="gpu")
    parser.add_argument("--detector", choices=("hybrid", "comic_text_detector", "contour"),
                        default="hybrid")
    parser.add_argument("--target-lang", choices=("id", "en"), default="id")
    parser.add_argument("--font-scale", type=float, default=1.0)
    return parser.parse_args()


def resolve_from_api(path: Path) -> Path:
    return path if path.is_absolute() else BASE_DIR / path


def select_images(directory: Path, name: str | None, limit: int) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    images = sorted(p for p in directory.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                    and (name or (p.stem.isdecimal() and len(p.stem) == 3)))
    if name:
        images = [p for p in images if p.name.lower() == name.lower()]
        if not images:
            raise FileNotFoundError(f"Image not found: {name}")
    elif limit > 0:
        images = images[:limit]
    if not images:
        raise ValueError(f"No images found in {directory}")
    stems = [p.stem.lower() for p in images]
    if len(stems) != len(set(stems)):
        raise ValueError("Images with the same stem would overwrite stage outputs")
    return images


def make_detector(kind: str, device: str):
    if kind == "contour":
        return ContourBubbleDetector()
    comic_detector = ComicTextDetector(device=device, require_gpu=(device == "gpu"))
    if kind == "comic_text_detector":
        return comic_detector
    return HybridBubbleDetector(comic_detector=comic_detector)


def get_segmentation(detector, image: Image.Image):
    comic_detector = getattr(detector, "comic_detector", detector)
    if hasattr(comic_detector, "get_cached_segmentation"):
        return comic_detector.get_cached_segmentation(image)
    return None


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_font(size: int):
    for path in ("C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/arial.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def wrap_pixels(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines = []
    for paragraph in (text or "-").splitlines():
        line = ""
        for char in paragraph:
            proposed = line + char
            if line and draw.textbbox((0, 0), proposed, font=font)[2] > max_width:
                lines.append(line)
                line = char
            else:
                line = proposed
        lines.append(line or " ")
    return lines


def transcript_image(image: Image.Image, bubbles, title: str, translated: bool) -> Image.Image:
    """Show source page and per-box text together so OCR/translation is inspectable."""
    source = image.convert("RGB")
    panel_width = 530
    font = load_font(20)
    heading_font = load_font(25)
    scratch = ImageDraw.Draw(source)
    entries = []
    for bubble in bubbles:
        label = f"#{bubble.id}  ({bubble.bounding_box.x}, {bubble.bounding_box.y})"
        if translated:
            value = bubble.translation or "[tidak diterjemahkan]"
            if is_graphic_text(bubble.text or ""):
                value = "[teks gambar/tanggal dilewati]"
        else:
            value = bubble.text or "[OCR kosong]"
        entries.append((label, wrap_pixels(scratch, value, font, panel_width - 32)))
    line_height = 27
    required_height = 60 + sum(34 + len(lines) * line_height + 14 for _, lines in entries)
    canvas = Image.new("RGB", (source.width + panel_width, max(source.height, required_height)), "white")
    canvas.paste(source, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((source.width, 0, canvas.width, canvas.height), fill="white")
    x = source.width + 16
    y = 16
    draw.text((x, y), title, font=heading_font, fill="black")
    y += 45
    for label, lines in entries:
        draw.text((x, y), label, font=font, fill=(20, 80, 170))
        y += 34
        for line in lines:
            draw.text((x, y), line, font=font, fill="black")
            y += line_height
        y += 14
    return canvas


def contact_sheet(stage_images: list[Path], output: Path):
    tile_w, tile_h = 540, 740
    sheet = Image.new("RGB", (tile_w * 3, tile_h * 2), (230, 230, 230))
    draw = ImageDraw.Draw(sheet)
    font = load_font(22)
    for index, path in enumerate(stage_images):
        x, y = (index % 3) * tile_w, (index // 3) * tile_h
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=(110, 110, 110))
        draw.text((x + 12, y + 10), STAGES[index].replace("_", " "), font=font, fill="black")
        with Image.open(path) as stage:
            preview = ImageOps.contain(stage.convert("RGB"), (tile_w - 24, tile_h - 58))
            sheet.paste(preview, (x + (tile_w - preview.width) // 2, y + 48))
    sheet.save(output, format="JPEG", quality=90)


def process_page(path: Path, run_dir: Path, detector, ocr, translator,
                 inpainter, typesetter, args) -> dict:
    timings = {}
    stem = path.stem
    with Image.open(path) as original:
        image = original.convert("RGB")
    steps_dir = run_dir / "steps"
    outputs = {stage: steps_dir / f"{stem}_{stage}.png" for stage in STAGES}

    start = time.perf_counter()
    bubbles = sort_manga_reading_order(detector.detect(image), reading_direction="rtl")
    timings["box_ms"] = round((time.perf_counter() - start) * 1000)
    annotate_and_save_bubbles(image, bubbles, str(outputs["01_box"]), draw_text_boxes=False)
    write_json(steps_dir / f"{stem}_01_box.json", [b.model_dump() for b in bubbles])

    start = time.perf_counter()
    segmentation = get_segmentation(detector, image)
    annotate_and_save_bubbles(image, bubbles, str(outputs["02_text_box"]), draw_layout=True)
    timings["text_box_ms"] = round((time.perf_counter() - start) * 1000)
    write_json(steps_dir / f"{stem}_02_text_box.json", [b.model_dump() for b in bubbles])

    start = time.perf_counter()
    ocr.recognize_all_bubbles(image, bubbles, padding=6)
    timings["ocr_ms"] = round((time.perf_counter() - start) * 1000)
    transcript_image(image, bubbles, "OCR - teks Jepang", False).save(outputs["03_ocr"])
    write_json(steps_dir / f"{stem}_03_ocr.json", [b.model_dump() for b in bubbles])

    start = time.perf_counter()
    candidates = [b for b in bubbles if (b.text or "").strip() and not is_graphic_text(b.text)]
    if candidates:
        translator.translate_bubbles(candidates, target_lang=args.target_lang)
    timings["translate_ms"] = round((time.perf_counter() - start) * 1000)
    transcript_image(image, bubbles, f"Google Translate - {args.target_lang}", True).save(outputs["04_translate"])
    write_json(steps_dir / f"{stem}_04_translate.json", [b.model_dump() for b in bubbles])

    active = [b for b in candidates if usable_translation(b.translation or "")]
    start = time.perf_counter()
    cleaned = inpainter.inpaint(image, seg_mask=segmentation, bubbles=active)
    timings["inpainting_ms"] = round((time.perf_counter() - start) * 1000)
    cleaned.convert("RGB").save(outputs["05_inpainting"])

    start = time.perf_counter()
    rendered = typesetter.typeset(cleaned, active, font_scale=args.font_scale) if active else cleaned
    timings["render_ms"] = round((time.perf_counter() - start) * 1000)
    rendered.convert("RGB").save(outputs["06_render"])
    contact_sheet(list(outputs.values()), run_dir / "all_in_one" / f"{stem}.jpg")
    return {"image": path.name, "status": "ok", "detected": len(bubbles),
            "translated": len(active), "timings": timings}


def main() -> int:
    args = parse_args()
    if importlib.util.find_spec("googletrans") is None:
        print("Google Translate test dependency missing: pip install -r api/requirements-dev.txt",
              file=sys.stderr)
        return 2
    images = select_images(resolve_from_api(args.dir), args.image, args.limit)
    root = resolve_from_api(args.output_root)
    run_dir = root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    for folder in ("steps", "all_in_one"):
        (run_dir / folder).mkdir(parents=True, exist_ok=False)

    print(f"Pages: {len(images)} | device: {args.device} | translation: Google (test only)")
    print(f"Output: {run_dir}")
    detector = make_detector(args.detector, args.device)
    ocr = MangaOcrService(device=args.device, require_gpu=(args.device == "gpu"))
    translator = GoogleTestTranslationService()
    inpainter = MangaInpaintingService()
    typesetter = MangaTypesettingService()

    results = []
    for index, path in enumerate(images, 1):
        print(f"[{index}/{len(images)}] {path.name}", flush=True)
        try:
            result = process_page(path, run_dir, detector, ocr, translator,
                                  inpainter, typesetter, args)
            print(f"    box={result['detected']} rendered={result['translated']}", flush=True)
        except Exception as exc:
            result = {"image": path.name, "status": "failed", "error": str(exc)}
            print(f"    FAILED: {exc}", file=sys.stderr, flush=True)
        results.append(result)
        write_json(run_dir / "manifest.json", {"provider": "google", "pages": results})

    failures = sum(item["status"] != "ok" for item in results)
    print(f"Complete: {len(images) - failures}/{len(images)} pages; output: {run_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
