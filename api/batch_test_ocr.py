import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import sys
import glob
import time
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from app.services.hybrid_detector import HybridBubbleDetector
from app.services.comic_text_detector import ComicTextDetector
from app.services.detector import sort_manga_reading_order, annotate_and_save_bubbles
from app.services.ocr_service import get_ocr_service

IMAGE_DIR = os.path.join(os.path.dirname(__file__), "image test")
OUTPUT_DIR = os.path.join(IMAGE_DIR, "output")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch test Manga OCR & Text Extraction on manga images.")
    parser.add_argument(
        "--device",
        choices=["auto", "gpu", "cpu"],
        default="gpu",
        help="Execution device: 'gpu' (DirectML) or 'cpu'."
    )
    parser.add_argument(
        "--image",
        default="",
        help="Specific image filename to test (e.g., '018.jpg'). If empty, runs batch."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit number of images for batch test (default: 5, 0 = all)."
    )
    parser.add_argument(
        "--detector",
        choices=["hybrid", "comic_text_detector", "contour"],
        default="hybrid",
        help="Detector engine: 'hybrid' (AI+Balloon), 'comic_text_detector' (AI text), or 'contour' (fast OpenCV)."
    )
    parser.add_argument(
        "--direction",
        choices=["rtl", "ltr"],
        default="rtl",
        help="Reading direction: 'rtl' (Manga) or 'ltr' (Manhwa)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.image:
        img_path = os.path.join(IMAGE_DIR, args.image)
        if not os.path.exists(img_path):
            print(f"[!] File not found: {img_path}")
            return
        images = [img_path]
    else:
        pattern = os.path.join(IMAGE_DIR, "*.jpg")
        images = sorted(glob.glob(pattern))
        if args.limit > 0:
            images = images[:args.limit]

    print("=" * 65)
    print("MANGA DETECTION & OCR BATCH TEST (STEP 3)")
    print(f"Compute Device     : {args.device.upper()}")
    print(f"Detector Engine    : {args.detector.upper()}")
    print(f"Total Pages        : {len(images)}")
    print("=" * 65)

    print(f"\n[1] Initializing Detector ({args.detector})...")
    if args.detector == "comic_text_detector":
        detector = ComicTextDetector(device=args.device, require_gpu=(args.device == "gpu"))
        print(f"    -> Detector Active on: {detector.device_name}")
    elif args.detector == "hybrid":
        comic_det = ComicTextDetector(device=args.device, require_gpu=(args.device == "gpu"))
        detector = HybridBubbleDetector(comic_detector=comic_det)
        print(f"    -> Detector Active on: {comic_det.device_name}")
    else:
        from app.services.detector import ContourBubbleDetector
        detector = ContourBubbleDetector()
        print("    -> Detector Active on: CPU (OpenCV Pipeline)")

    print("\n[2] Initializing Manga-OCR Service (ViT + RoBERTa Transformer)...")
    ocr_service = get_ocr_service(device=args.device, require_gpu=(args.device == "gpu"))
    print(f"    -> OCR Engine Active on: {ocr_service.device_name}")

    print("\n" + "=" * 65)
    print("STARTING TEXT EXTRACTION PIPELINE...")
    print("=" * 65)

    total_bubbles = 0
    t_start = time.time()

    for idx, img_path in enumerate(images, 1):
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]

        t0 = time.perf_counter()
        with Image.open(img_path) as img:
            # 1. Detection
            t_det0 = time.perf_counter()
            bubbles = detector.detect(img)
            ordered = sort_manga_reading_order(bubbles, reading_direction=args.direction)
            t_det = (time.perf_counter() - t_det0) * 1000

            # 2. OCR
            t_ocr0 = time.perf_counter()
            ordered = ocr_service.recognize_all_bubbles(img, ordered, padding=6)
            t_ocr = (time.perf_counter() - t_ocr0) * 1000

            t_total = (time.perf_counter() - t0) * 1000
            total_bubbles += len(ordered)

            print(f"\n[{idx:02d}/{len(images)}] {fname} | Detection: {t_det:.0f}ms | OCR: {t_ocr:.0f}ms | Total: {t_total:.0f}ms")
            print("-" * 65)

            if not ordered:
                print("   (No speech bubbles detected)")
            else:
                for b in ordered:
                    box = b.bounding_box
                    tbox = b.text_box
                    print(f"   Bubble #{b.id} [Position: x={box.x}, y={box.y}, {box.width}x{box.height}]:")
                    print(f"      -> Japanese Text: \"{b.text}\"")

            # Save OCR summary to JSON file
            json_out = os.path.join(OUTPUT_DIR, f"ocr_{stem}.json")
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "image": fname,
                        "total_bubbles": len(ordered),
                        "detection_ms": round(t_det, 1),
                        "ocr_ms": round(t_ocr, 1),
                        "bubbles": [b.model_dump() for b in ordered]
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            # Save visualization image
            out_img = os.path.join(OUTPUT_DIR, f"out_{stem}.png")
            annotate_and_save_bubbles(img, ordered, out_img)

    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print("BATCH TEST COMPLETED")
    print(f"Total Pages Processed: {len(images)}")
    print(f"Total Bubbles Extracted: {total_bubbles}")
    print(f"Total Elapsed Time    : {elapsed:.1f}s (Average: {elapsed / max(1, len(images)):.2f}s/page)")
    print(f"JSON and image outputs saved in: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
