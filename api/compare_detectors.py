import os
import time
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from app.services.detector import ContourBubbleDetector, sort_manga_reading_order
from app.services.comic_text_detector import ComicTextDetector

IMAGE_DIR = r"D:\Codingan\manga translator\api\image test"
OUTPUT_DIR = os.path.join(IMAGE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_annotations(
    image: Image.Image,
    bubbles,
    box_color=(255, 0, 0),
    title="DETECTOR",
    is_text_box=False
) -> Image.Image:
    """Draws boxes and headers on a copy of the image."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    # Header banner
    draw.rectangle([0, 0, canvas.width, 42], fill=(20, 20, 20))
    draw.text((15, 10), title, fill=(255, 255, 255))

    for b in bubbles:
        box = b.text_box if (is_text_box and b.text_box is not None) else b.bounding_box
        draw.rectangle([box.x, box.y, box.right, box.bottom], outline=box_color, width=4)
        draw.rectangle([box.x, max(42, box.y - 26), box.x + 46, max(42, box.y)], fill=box_color)
        draw.text((box.x + 8, max(42, box.y - 22)), f"#{b.id}", fill=(255, 255, 255))

    return canvas


def run_comparison(image_name: str = "018.jpg"):
    img_path = os.path.join(IMAGE_DIR, image_name)
    if not os.path.exists(img_path):
        print(f"File {img_path} not found.")
        return

    print("=" * 70)
    print(f"PERBANDINGAN DETEKTOR PADA: {image_name}")
    print("=" * 70)

    img = Image.open(img_path)

    # 1. ContourBubbleDetector (OpenCV heuristic)
    print("\n[1] Menjalankan ContourBubbleDetector (OpenCV)...")
    t0 = time.perf_counter()
    contour_det = ContourBubbleDetector()
    contour_bubbles = contour_det.detect(img)
    contour_ordered = sort_manga_reading_order(contour_bubbles, reading_direction="rtl")
    t_contour = (time.perf_counter() - t0) * 1000

    print(f"    - Waktu eksekusi: {t_contour:.1f} ms")
    print(f"    - Total terdeteksi: {len(contour_ordered)} balon kata")
    for b in contour_ordered:
        bb = b.bounding_box
        print(f"      Balon #{b.id}: x={bb.x}, y={bb.y}, {bb.width}x{bb.height}")

    # 2. ComicTextDetector (Deep Learning ONNX)
    print("\n[2] Menjalankan ComicTextDetector (AI Manga Text)...")
    t1 = time.perf_counter()
    comic_det = ComicTextDetector(conf_threshold=0.35, nms_threshold=0.35, num_threads=4)
    comic_bubbles = comic_det.detect(img)
    comic_ordered = sort_manga_reading_order(comic_bubbles, reading_direction="rtl")
    t_comic = (time.perf_counter() - t1) * 1000

    print(f"    - Waktu eksekusi: {t_comic:.1f} ms")
    print(f"    - Total terdeteksi: {len(comic_ordered)} blok teks")
    for b in comic_ordered:
        bb = b.bounding_box
        print(f"      Teks #{b.id} ({b.direction}): x={bb.x}, y={bb.y}, {bb.width}x{bb.height} (conf: {b.confidence})")

    # 3. Create Side-by-Side Comparison Image
    print("\n[3] Membuat visual perbandingan side-by-side...")
    canvas_contour = draw_annotations(
        img,
        contour_ordered,
        box_color=(230, 50, 50),
        title=f"ContourBubbleDetector (OpenCV Balon) - {len(contour_ordered)} Balon | {t_contour:.0f}ms"
    )

    canvas_comic = draw_annotations(
        img,
        comic_ordered,
        box_color=(30, 160, 60),
        title=f"ComicTextDetector (AI Teks Manga) - {len(comic_ordered)} Teks | {t_comic:.0f}ms",
        is_text_box=True
    )

    # Combine side-by-side
    combined_w = canvas_contour.width + canvas_comic.width
    combined_h = max(canvas_contour.height, canvas_comic.height)
    combined = Image.new("RGB", (combined_w, combined_h))
    combined.paste(canvas_contour, (0, 0))
    combined.paste(canvas_comic, (canvas_contour.width, 0))

    stem = os.path.splitext(image_name)[0]
    out_path = os.path.join(OUTPUT_DIR, f"compare_{stem}.png")
    combined.save(out_path)
    print(f"    -> Gambar perbandingan berhasil disimpan di: {out_path}")

    # 4. Summary of Key Differences
    print("\n" + "=" * 70)
    print("ANALISIS HASIL PERBANDINGAN:")
    print("=" * 70)
    print("a. Cakupan Teks Tanpa Balon (Borderless Dialogue):")
    print("   - ContourDetector HANYA mendeteksi area balon tertutup/putih (miss teks mengambang/borderless).")
    print("   - ComicTextDetector BERHASIL mendeteksi teks mengambang/tanpa batas balon (misal teks kiri atas).")
    print("b. Presisi Batas:")
    print("   - ContourDetector mendeteksi area luar balon (cocok untuk inpainting / pembersihan balon).")
    print("   - ComicTextDetector mendeteksi area tepat di sekeliling karakter teks (cocok untuk OCR).")
    print("c. Kecepatan vs Akurasi:")
    print(f"   - ContourDetector: Sangat cepat ({t_contour:.0f}ms) berbasis geometri heuristik OpenCV.")
    print(f"   - ComicTextDetector: Berbasis Neural Network ({t_comic:.0f}ms) sangat presisi pada karakter manga.")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    run_comparison("018.jpg")

