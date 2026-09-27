import os
import glob
import time
import argparse
from PIL import Image, ImageDraw

from app.services.detector import (
    ContourBubbleDetector,
    sort_manga_reading_order,
    annotate_and_save_bubbles
)

IMAGE_DIR = r"D:\Codingan\manga translator\api\image test"
OUTPUT_DIR = os.path.join(IMAGE_DIR, "output")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch test manga bubble/text detection on folder.")
    parser.add_argument(
        "--detector",
        choices=["contour", "comic_text_detector", "hybrid"],
        default="contour",
        help="Detector engine: 'contour' (fast OpenCV), 'comic_text_detector' (Deep Learning AI), or 'hybrid' (AI + Balloon)."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="Hardware execution device: 'gpu' (force GPU DirectML/CUDA), 'cpu' (force CPU), or 'auto'."
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Stop if GPU is unavailable (equivalent to --device gpu)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images to process (0 = all images)."
    )
    parser.add_argument(
        "--direction",
        choices=["rtl", "ltr"],
        default="rtl",
        help="Reading direction: 'rtl' (Manga) or 'ltr' (Manhwa)."
    )
    parser.add_argument(
        "--input-dir",
        default=IMAGE_DIR,
        help="Path to folder containing manga images."
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Path to folder where annotated outputs will be saved."
    )
    args = parser.parse_args()
    if args.require_gpu:
        args.device = "gpu"
    if args.device == "gpu" and args.detector not in ("comic_text_detector", "hybrid"):
        parser.error("--device gpu requires --detector comic_text_detector or hybrid")
    return args


def get_detector(detector_type: str, device: str = "auto", require_gpu: bool = False):
    if detector_type == "comic_text_detector":
        from app.services.comic_text_detector import ComicTextDetector
        print("[INFO] Initializing ComicTextDetector (AI Deep Learning)...")
        detector = ComicTextDetector(
            conf_threshold=0.35, nms_threshold=0.35, num_threads=4,
            require_gpu=require_gpu or (device == "gpu"),
            device=device
        )
        print(f"[INFO] Target Device: {detector.device_name}")
        if detector.target_device == "gpu":
            print("[INFO] CPU Fallback : Disabled (Inference 100% terkunci di GPU)")
        return detector
    elif detector_type == "hybrid":
        from app.services.hybrid_detector import HybridBubbleDetector
        from app.services.comic_text_detector import ComicTextDetector
        print("[INFO] Initializing HybridBubbleDetector (AI Text + OpenCV Balloon)...")
        comic_det = ComicTextDetector(
            num_threads=4,
            require_gpu=require_gpu or (device == "gpu"),
            device=device
        )
        detector = HybridBubbleDetector(comic_detector=comic_det, num_threads=4)
        print(f"[INFO] Target Device: {comic_det.device_name}")
        if comic_det.target_device == "gpu":
            print("[INFO] CPU Fallback : Disabled (Inference 100% terkunci di GPU)")
        return detector
    else:
        print("[INFO] Initializing ContourBubbleDetector (Fast OpenCV Heuristic)...")
        print("[INFO] Target Device: CPU (OpenCV C++ Pipeline)")
        return ContourBubbleDetector()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Collect all images (.jpg, .jpeg, .png, .webp)
    pattern_jpg = os.path.join(args.input_dir, "*.jpg")
    pattern_jpeg = os.path.join(args.input_dir, "*.jpeg")
    pattern_png = os.path.join(args.input_dir, "*.png")
    pattern_webp = os.path.join(args.input_dir, "*.webp")
    images = sorted(
        glob.glob(pattern_jpg) + glob.glob(pattern_jpeg) + 
        glob.glob(pattern_png) + glob.glob(pattern_webp)
    )

    if not images:
        print(f"[!] No images found in: {args.input_dir}")
        return

    if args.limit > 0:
        images = images[:args.limit]

    print(f"Total images to process: {len(images)} (Detector: {args.detector})")
    print(f"Output directory: {args.output_dir}\n")

    detector = get_detector(args.detector, device=args.device, require_gpu=args.require_gpu)

    zero_detected = []
    total_bubbles_detected = 0
    start_time = time.time()

    for idx, img_path in enumerate(images, 1):
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]
        out_file = os.path.join(args.output_dir, f"out_{stem}.png")

        try:
            t0 = time.perf_counter()
            with Image.open(img_path) as img:
                bubbles = detector.detect(img)
                ordered = sort_manga_reading_order(bubbles, reading_direction=args.direction)
                duration_img = (time.perf_counter() - t0) * 1000

                count = len(ordered)
                total_bubbles_detected += count

                if count == 0:
                    zero_detected.append(fname)
                    print(f"[{idx:02d}/{len(images)}] {fname} -> [!] 0 bubbles detected ({duration_img:.0f}ms)")
                else:
                    print(f"[{idx:02d}/{len(images)}] {fname} -> {count} bubbles ({duration_img:.0f}ms)")

                # Save strictly ONE annotated image per input image
                annotate_and_save_bubbles(img, ordered, out_file)

        except Exception as e:
            print(f"[{idx:02d}/{len(images)}] {fname} -> ERROR: {e}")

    total_duration = time.time() - start_time
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETED")
    print("=" * 60)
    print(f"Total images processed: {len(images)}")
    print(f"Total detection items: {total_bubbles_detected}")
    print(f"Pages with 0 detection: {len(zero_detected)} -> {zero_detected}")
    print(f"Total elapsed time: {total_duration:.1f}s (avg: {total_duration / max(1, len(images)):.2f}s/page)")
    print(f"Annotated outputs saved in: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
