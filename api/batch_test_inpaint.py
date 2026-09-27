import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

# Ensure app package is accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.hybrid_detector import HybridBubbleDetector
from app.services.comic_text_detector import ComicTextDetector
from app.services.detector import ContourBubbleDetector, sort_manga_reading_order
from app.services.ocr_service import MangaOcrService
from app.services.translation_service import MangaTranslationService
from app.services.inpainting_service import MangaInpaintingService
from app.services.typesetting_service import MangaTypesettingService


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch Runner for Manga Inpainting & Typesetting (Step 5)"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="image test",
        help="Directory containing manga test images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="image test/output_inpaint",
        help="Directory to save inpainted and typeset images"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of images to process (default: 5, set 0 for all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Device provider ('auto', 'gpu', or 'cpu')"
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="hybrid",
        choices=["hybrid", "comic_text_detector", "contour"],
        help="Bubble and text detector algorithm (default: 'hybrid')"
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="id",
        help="Target translation language code (default: 'id')"
    )
    parser.add_argument(
        "--no-typeset",
        action="store_true",
        help="Disable typesetting (clean inpaint raw image only)"
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Font scaling multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--no-all-caps",
        action="store_true",
        help="Disable ALL CAPS formatting in typesetting"
    )
    return parser.parse_args()


def get_detector(detector_type: str, device: str):
    if detector_type == "comic_text_detector":
        return ComicTextDetector(device=device)
    elif detector_type == "hybrid":
        comic_det = ComicTextDetector(device=device)
        return HybridBubbleDetector(comic_detector=comic_det)
    return ContourBubbleDetector()


def main():
    args = parse_args()
    input_dir = Path(args.dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[ERROR] Input directory '{input_dir}' not found.")
        sys.exit(1)

    image_files = sorted(
        [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    )
    if args.limit and args.limit > 0:
        image_files = image_files[:args.limit]

    total_images = len(image_files)
    if total_images == 0:
        print(f"[WARN] No valid images found in '{input_dir}'.")
        sys.exit(0)

    print("=" * 80)
    print(f"Manga Inpainting & Typesetting Batch Runner (Step 5)")
    print(f"Target Images: {total_images} | Device: {args.device} | Target Lang: {args.target_lang}")
    print(f"Detector: {args.detector} | Typeset: {not args.no_typeset} | Caps: {not args.no_all_caps}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    # 1. Initialize Pipeline Services
    print("\n[INIT] Initializing Detection & AI Services...")
    detector = get_detector(args.detector, args.device)
    ocr_service = MangaOcrService(device=args.device)
    trans_service = MangaTranslationService()
    inpaint_service = MangaInpaintingService()
    typeset_service = MangaTypesettingService(all_caps=not args.no_all_caps)
    print("[INIT] Services loaded successfully.\n")

    grand_total_time = 0.0

    for idx, img_path in enumerate(image_files, start=1):
        print(f"\n[{idx:02d}/{total_images:02d}] Processing: {img_path.name}")
        img = Image.open(img_path)
        img.load()

        step_t0 = time.perf_counter()

        # Step A: Detection & Segmentation
        t0 = time.perf_counter()
        detected_bubbles = detector.detect(img)
        ordered_bubbles = sort_manga_reading_order(detected_bubbles, reading_direction="rtl")
        det_ms = (time.perf_counter() - t0) * 1000

        # Step B: Neural character segmentation mask
        seg_mask = None
        if args.detector in ("hybrid", "comic_text_detector"):
            comic_det = getattr(detector, "comic_detector", detector)
            if hasattr(comic_det, "detect_raw") and hasattr(comic_det, "get_unletterboxed_seg"):
                try:
                    blk, seg, det, r, (dw, dh) = comic_det.detect_raw(img)
                    seg_mask = comic_det.get_unletterboxed_seg(seg, img.width, img.height, dw, dh)
                except Exception:
                    seg_mask = None

        # Step C: OCR Extraction
        ocr_ms = 0.0
        if not args.no_typeset and ordered_bubbles:
            t0 = time.perf_counter()
            ordered_bubbles = ocr_service.recognize_all_bubbles(img, ordered_bubbles)
            ocr_ms = (time.perf_counter() - t0) * 1000

        # Step D: Contextual Translation
        trans_ms = 0.0
        if not args.no_typeset and ordered_bubbles:
            t0 = time.perf_counter()
            ordered_bubbles = trans_service.translate_bubbles(ordered_bubbles, target_lang=args.target_lang)
            trans_ms = (time.perf_counter() - t0) * 1000

        # Step E: Inpainting / Text Erasure
        t0 = time.perf_counter()
        inpainted_img = inpaint_service.inpaint(img, seg_mask=seg_mask, bubbles=ordered_bubbles)
        inpaint_ms = (time.perf_counter() - t0) * 1000

        # Step F: Comic Typesetting
        typeset_ms = 0.0
        if not args.no_typeset and ordered_bubbles:
            t0 = time.perf_counter()
            final_img = typeset_service.typeset(inpainted_img, ordered_bubbles, font_scale=args.font_scale)
            typeset_ms = (time.perf_counter() - t0) * 1000
        else:
            final_img = inpainted_img

        total_ms = (time.perf_counter() - step_t0) * 1000
        grand_total_time += total_ms

        # Save output image
        out_file = output_dir / f"translated_{img_path.stem}.jpg"
        if final_img.mode != "RGB":
            final_img = final_img.convert("RGB")
        final_img.save(out_file, format="JPEG", quality=95)

        print(
            f"       Timing: Det {det_ms:.0f}ms | OCR {ocr_ms:.0f}ms | "
            f"Trans {trans_ms:.0f}ms | Inpaint {inpaint_ms:.0f}ms | "
            f"Typeset {typeset_ms:.0f}ms | Total: {total_ms/1000:.2f}s"
        )
        print(f"       Saved to: {out_file.name} (Detected: {len(ordered_bubbles)} bubbles)")

        for b in ordered_bubbles:
            jp = (b.text or "").strip()
            tr = (b.translation or "").strip()
            try:
                print(f"         #{b.id} [{b.bounding_box.width}x{b.bounding_box.height}]: '{jp}' -> '{tr}'")
            except Exception:
                safe_jp = jp.encode("ascii", "replace").decode("ascii")
                safe_tr = tr.encode("ascii", "replace").decode("ascii")
                print(f"         #{b.id} [{b.bounding_box.width}x{b.bounding_box.height}]: '{safe_jp}' -> '{safe_tr}'")

    print("\n" + "=" * 80)
    print(f"Batch Inpainting & Typesetting Complete!")
    print(f"Total Images: {total_images} | Total Duration: {grand_total_time/1000:.2f}s | Avg/Page: {(grand_total_time/total_images)/1000:.2f}s")
    print(f"All translated pages saved in: {output_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
