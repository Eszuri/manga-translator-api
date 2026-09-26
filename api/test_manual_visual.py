"""
Script uji coba visual deteksi balon kata.
Dapat dijalankan langsung:
    python test_manual_visual.py
Atau menentukan gambar spesifik:
    python test_manual_visual.py 020.jpg
"""

import sys
import os
import glob
from PIL import Image, ImageDraw
from app.services.detector import ContourBubbleDetector, sort_manga_reading_order


def run_visual_test_on_file(image_path: str):
    print("=" * 65)
    print(f"[*] Memproses gambar: {os.path.basename(image_path)}")

    image = Image.open(image_path)
    detector = ContourBubbleDetector()
    detected = detector.detect(image)
    ordered = sort_manga_reading_order(detected, reading_direction="rtl")

    print(f"    -> Ditemukan {len(ordered)} balon kata (Urutan Baca RTL):")

    output_img = image.copy()
    draw = ImageDraw.Draw(output_img)

    for bubble in ordered:
        b = bubble.bounding_box
        print(f"       #{bubble.id}: x={b.x}, y={b.y}, w={b.width}, h={b.height}, arah={bubble.direction}")
        draw.rectangle([b.x, b.y, b.right, b.bottom], outline=(255, 0, 0), width=4)
        draw.rectangle([b.x, max(0, b.y - 28), b.x + 44, b.y], fill=(255, 0, 0))
        draw.text((b.x + 8, max(0, b.y - 24)), f"#{bubble.id}", fill=(255, 255, 255))

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_filename = f"output_{base_name}.png"
    output_img.save(out_filename)
    print(f"    -> Hasil visual disimpan ke: {out_filename}")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None

    if target:
        if os.path.exists(target):
            run_visual_test_on_file(target)
        else:
            print(f"File '{target}' tidak ditemukan!")
    else:
        # Cari semua file komik jpg/png di folder saat ini
        images = [f for f in glob.glob("*.jpg") + glob.glob("*.png") if not f.startswith("output_") and not f.startswith("sample_")]
        if not images:
            print("Tidak ditemukan file gambar komik (*.jpg/*.png).")
            return

        print("=" * 65)
        print(f"  MEMERIKSA {len(images)} GAMBAR KOMIK DI DALAM FOLDER")
        print("=" * 65)
        for img_path in sorted(images):
            run_visual_test_on_file(img_path)
        print("\n[V] Seluruh gambar telah selesai diproses! Periksa file output_*.png di folder.")


if __name__ == "__main__":
    main()
