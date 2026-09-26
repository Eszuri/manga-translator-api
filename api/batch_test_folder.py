import os
import glob
import time
from PIL import Image, ImageDraw
from app.services.detector import ContourBubbleDetector, sort_manga_reading_order

IMAGE_DIR = r"D:\Codingan\manga translator\api\image test"
OUTPUT_DIR = os.path.join(IMAGE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = ContourBubbleDetector()

images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGE_DIR, "*.png")))
print(f"Total images found: {len(images)}")

zero_detected = []
total_bubbles_detected = 0
start_time = time.time()

for idx, img_path in enumerate(images, 1):
    fname = os.path.basename(img_path)
    try:
        img = Image.open(img_path)
        bubbles = detector.detect(img)
        ordered = sort_manga_reading_order(bubbles, reading_direction="rtl")
        
        count = len(ordered)
        total_bubbles_detected += count
        
        if count == 0:
            zero_detected.append(fname)
            print(f"[{idx:02d}/{len(images)}] {fname} -> [!] 0 bubbles detected")
        else:
            print(f"[{idx:02d}/{len(images)}] {fname} -> {count} bubbles")

        # Save visual output
        out = img.copy()
        draw = ImageDraw.Draw(out)
        for b in ordered:
            box = b.bounding_box
            draw.rectangle([box.x, box.y, box.right, box.bottom], outline=(255, 0, 0), width=4)
            draw.rectangle([box.x, max(0, box.y - 28), box.x + 44, box.y], fill=(255, 0, 0))
            draw.text((box.x + 8, max(0, box.y - 24)), f"#{b.id}", fill=(255, 255, 255))
            
        stem = os.path.splitext(fname)[0]
        out.save(os.path.join(OUTPUT_DIR, f"out_{stem}.png"))
        out.save(os.path.join(OUTPUT_DIR, f"out_{fname}.png"))

    except Exception as e:
        print(f"[{idx:02d}/{len(images)}] {fname} -> ERROR: {e}")

duration = time.time() - start_time
print("=" * 60)
print(f"Completed in {duration:.1f}s")
print(f"Total images: {len(images)}")
print(f"Total bubbles detected across chapter: {total_bubbles_detected}")
print(f"Pages with 0 bubbles detected: {len(zero_detected)} -> {zero_detected}")
print(f"Annotated outputs saved in: {OUTPUT_DIR}")
