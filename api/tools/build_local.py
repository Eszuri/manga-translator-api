import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from PIL import Image, ImageDraw, ImageFont

from app.services.comic_text_detector import ComicTextDetector
from app.services.detector import (
    annotate_and_save_bubbles,
    sort_manga_reading_order,
)
from app.services.hybrid_detector import HybridBubbleDetector
from app.services.inpainting_service import MangaInpaintingService
from app.services.ocr_service import MangaOcrService
from app.services.translation_filters import is_graphic_text, usable_translation
from app.services.translation_service import get_translation_service
from app.services.typesetting_service import MangaTypesettingService


API_DIR = Path(__file__).resolve().parents[1]
STAGE_FOLDERS = {
    "box": "Box",
    "text_box": "Text Box",
    "ocr": "OCR",
    "translate": "translate",
    "inpainting": "inpainting",
    "render": "render",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build local manga output from original images")
    parser.add_argument("--input-dir", type=Path, default=Path("Images/original Images"))
    parser.add_argument("--output-dir", type=Path, default=Path("Images/build Images"))
    parser.add_argument("--image", help="One input filename, e.g. 009.jpg")
    parser.add_argument("--limit", type=int, default=0, help="Pages to process; 0 means all")
    parser.add_argument("--target-lang", choices=("id", "en"), default="id")
    parser.add_argument("--font-scale", type=float, default=1.0)
    return parser.parse_args()


def resolve_from_api(path: Path) -> Path:
    return path if path.is_absolute() else API_DIR / path


def select_images(directory: Path, name: str | None, limit: int) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    images = sorted(p for p in directory.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
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


def make_detector():
    comic_detector = ComicTextDetector()
    return HybridBubbleDetector(comic_detector=comic_detector)


def get_segmentation(detector, image: Image.Image):
    comic_detector = getattr(detector, "comic_detector", detector)
    if hasattr(comic_detector, "get_cached_segmentation"):
        return comic_detector.get_cached_segmentation(image)
    return None


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
    source = image.convert("RGB")
    panel_width = 530
    font = load_font(20)
    heading_font = load_font(25)
    scratch = ImageDraw.Draw(source)
    entries = []
    for bubble in bubbles:
        label = f"#{bubble.id}  ({bubble.bounding_box.x}, {bubble.bounding_box.y})"
        if translated:
            value = bubble.translation or "[not translated]"
            if is_graphic_text(bubble.text or ""):
                value = "[graphic text/date skipped]"
        else:
            value = bubble.text or "[empty OCR]"
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


def create_stage_outputs(output_dir: Path, stem: str) -> dict[str, Path]:
    return {
        stage: output_dir / folder / f"{stem}.png"
        for stage, folder in STAGE_FOLDERS.items()
    }


def process_page(path: Path, output_dir: Path, detector, ocr, translator,
                 inpainter, typesetter, args) -> dict:
    timings = {}
    stem = path.stem
    with Image.open(path) as original:
        image = original.convert("RGB")
    outputs = create_stage_outputs(output_dir, stem)

    start = time.perf_counter()
    bubbles = sort_manga_reading_order(detector.detect(image), reading_direction="rtl")
    timings["box_ms"] = round((time.perf_counter() - start) * 1000)
    annotate_and_save_bubbles(image, bubbles, str(outputs["box"]), draw_text_boxes=False)

    start = time.perf_counter()
    segmentation = get_segmentation(detector, image)
    annotate_and_save_bubbles(image, bubbles, str(outputs["text_box"]), draw_layout=True)
    timings["text_box_ms"] = round((time.perf_counter() - start) * 1000)

    start = time.perf_counter()
    ocr.recognize_all_bubbles(image, bubbles, padding=6)
    timings["ocr_ms"] = round((time.perf_counter() - start) * 1000)
    transcript_image(image, bubbles, "OCR - Japanese text", False).save(outputs["ocr"])

    start = time.perf_counter()
    candidates = [b for b in bubbles if (b.text or "").strip() and not is_graphic_text(b.text)]
    if candidates:
        asyncio.run(translator.translate_bubbles_async(
            candidates, target_lang=args.target_lang, translator="google"
        ))
    timings["translate_ms"] = round((time.perf_counter() - start) * 1000)
    transcript_image(image, bubbles, f"Google Translate - {args.target_lang}", True).save(outputs["translate"])

    active = [b for b in candidates if usable_translation(b.translation or "")]
    start = time.perf_counter()
    cleaned = inpainter.inpaint(image, seg_mask=segmentation, bubbles=active)
    timings["inpainting_ms"] = round((time.perf_counter() - start) * 1000)
    cleaned.convert("RGB").save(outputs["inpainting"])

    start = time.perf_counter()
    rendered = typesetter.typeset(cleaned, active, font_scale=args.font_scale) if active else cleaned
    timings["render_ms"] = round((time.perf_counter() - start) * 1000)
    rendered.convert("RGB").save(outputs["render"])
    return {"image": path.name, "status": "ok", "detected": len(bubbles),
            "translated": len(active), "timings": timings}


def main() -> int:
    args = parse_args()
    images = select_images(resolve_from_api(args.input_dir), args.image, args.limit)
    output_dir = resolve_from_api(args.output_dir)
    for folder in STAGE_FOLDERS.values():
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    print(f"Pages: {len(images)} | device: gpu | translation: Google")
    print(f"Output: {output_dir}")
    detector = make_detector()
    ocr = MangaOcrService()
    translator = get_translation_service()
    inpainter = MangaInpaintingService()
    typesetter = MangaTypesettingService()

    results = []
    for index, path in enumerate(images, 1):
        print(f"[{index}/{len(images)}] {path.name}", flush=True)
        try:
            result = process_page(path, output_dir, detector, ocr, translator,
                                  inpainter, typesetter, args)
            print(f"    box={result['detected']} rendered={result['translated']}", flush=True)
        except Exception as exc:
            result = {"image": path.name, "status": "failed", "error": str(exc)}
            print(f"    FAILED: {exc}", file=sys.stderr, flush=True)
        results.append(result)
    failures = sum(item["status"] != "ok" for item in results)
    print(f"Complete: {len(images) - failures}/{len(images)} pages; output: {output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
