import os
import json
import argparse
import time
from typing import Dict, List, Tuple
from PIL import Image

from app.schemas import BoundingBox, DetectedBubble
from app.services.detector import calculate_iou, ContourBubbleDetector
from app.services.comic_text_detector import ComicTextDetector
from app.services.hybrid_detector import HybridBubbleDetector

BENCHMARK_PATH = os.path.join(
    os.path.dirname(__file__), "ground_truth", "benchmark_annotations.json"
)
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "image test")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate manga text/bubble detectors against Ground Truth.")
    parser.add_argument(
        "--detector",
        choices=["contour", "comic_text_detector", "hybrid"],
        default="hybrid",
        help="Detector engine to benchmark."
    )
    parser.add_argument(
        "--split",
        choices=["all", "dev", "holdout"],
        default="dev",
        help="Data split to evaluate: 'dev' (10 dev/validation pages), 'holdout' (5 holdout pages), or 'all' (15 pages)."
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.45,
        help="Minimum IoU threshold to consider a detection as a True Positive (default: 0.45)."
    )
    return parser.parse_args()


def load_ground_truth(benchmark_file: str) -> dict:
    with open(benchmark_file, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_page(
    detected_bubbles: List[DetectedBubble],
    gt_bubbles: List[dict],
    iou_thresh: float = 0.45,
    eval_text_box: bool = False
) -> Tuple[int, int, int, float]:
    """
    Evaluates detections against ground truth for a single page.
    Returns: (TP, FP, FN, mean_matched_iou)
    """
    gt_matched = [False] * len(gt_bubbles)
    det_matched = [False] * len(detected_bubbles)
    matched_ious = []

    for d_idx, d_b in enumerate(detected_bubbles):
        # Choose box to evaluate: text_box if requested, else bounding_box
        target_box = (d_b.text_box if (eval_text_box and d_b.text_box) else d_b.bounding_box)

        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, g_b in enumerate(gt_bubbles):
            if gt_matched[g_idx]:
                continue
            gt_coords = g_b["text_box"] if eval_text_box else g_b["bubble_box"]
            gt_bbox = BoundingBox(
                x=gt_coords["x"],
                y=gt_coords["y"],
                width=gt_coords["width"],
                height=gt_coords["height"]
            )

            iou = calculate_iou(target_box, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            gt_matched[best_gt_idx] = True
            det_matched[d_idx] = True
            matched_ious.append(best_iou)

    tp = sum(det_matched)
    fp = len(detected_bubbles) - tp
    fn = len(gt_bubbles) - sum(gt_matched)
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    return tp, fp, fn, mean_iou


def main():
    args = parse_args()

    if not os.path.exists(BENCHMARK_PATH):
        print(f"Error: Benchmark file {BENCHMARK_PATH} not found.")
        return

    data = load_ground_truth(BENCHMARK_PATH)
    splits = data["metadata"]["split"]

    if args.split == "dev":
        target_images = splits["development_validation"]
    elif args.split == "holdout":
        target_images = splits["holdout_test"]
    else:
        target_images = splits["development_validation"] + splits["holdout_test"]

    print("=" * 75)
    print(f"EVALUASI BENCHMARK: Detektor '{args.detector}' pada Split '{args.split}' ({len(target_images)} Halaman)")
    print("=" * 75)

    # Instantiate detector
    t0_init = time.time()
    if args.detector == "comic_text_detector":
        detector = ComicTextDetector(num_threads=4)
        eval_text = True
    elif args.detector == "hybrid":
        detector = HybridBubbleDetector(num_threads=4)
        eval_text = False
    else:
        detector = ContourBubbleDetector()
        eval_text = False

    print(f"Inisialisasi detektor selesai dalam {time.time() - t0_init:.2f}s.\n")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    page_ious = []
    total_time = 0.0

    print(f"{'Halaman':<12} | {'GT':<4} | {'Det':<4} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'IoU':<6} | {'Waktu'}")
    print("-" * 75)

    for img_name in target_images:
        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"{img_name:<12} | [File gambar tidak ditemukan]")
            continue

        gt_page = data["annotations"].get(img_name)
        if not gt_page:
            continue
        gt_bubbles = gt_page["bubbles"]

        t0 = time.perf_counter()
        with Image.open(img_path) as img:
            detected = detector.detect(img)
        dt_ms = (time.perf_counter() - t0) * 1000
        total_time += dt_ms

        tp, fp, fn, mean_iou = evaluate_page(
            detected, gt_bubbles, iou_thresh=args.iou_thresh, eval_text_box=eval_text
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        if mean_iou > 0:
            page_ious.append(mean_iou)

        print(f"{img_name:<12} | {len(gt_bubbles):<4} | {len(detected):<4} | {tp:<4} | {fp:<4} | {fn:<4} | {mean_iou:.2f}   | {dt_ms:.0f}ms")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    overall_mean_iou = sum(page_ious) / len(page_ious) if page_ious else 0.0

    print("-" * 75)
    print("RINGKASAN HASIL EVALUASI:")
    print(f"  * Total True Positives  (TP) : {total_tp}")
    print(f"  * Total False Positives (FP) : {total_fp} (salah deteksi)")
    print(f"  * Total False Negatives (FN) : {total_fn} (balon terlewat)")
    print(f"  * Precision                  : {precision * 100:.1f}%")
    print(f"  * Recall                     : {recall * 100:.1f}%")
    print(f"  * F1-Score                   : {f1 * 100:.1f}%")
    print(f"  * Mean Matched IoU           : {overall_mean_iou:.3f}")
    print(f"  * Rata-rata waktu per halaman: {total_time / max(1, len(target_images)):.1f} ms")
    print("=" * 75)


if __name__ == "__main__":
    main()
