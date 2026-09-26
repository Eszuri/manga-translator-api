import os
import glob
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw

def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[0]+box_a[2], box_b[0]+box_b[2])
    y2 = min(box_a[1]+box_a[3], box_b[1]+box_b[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = box_a[2]*box_a[3] + box_b[2]*box_b[3] - inter
    return inter / union if union > 0 else 0.0

def containment_ratio(inner, outer):
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[0]+inner[2], outer[0]+outer[2])
    y2 = min(inner[1]+inner[3], outer[1]+outer[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    return ((x2 - x1) * (y2 - y1)) / float(inner[2] * inner[3])

def detect_page(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total_area = h * w
    scale = max(w, h) / 1600.0

    boxes = []

    # 1. Narration Boxes (rectangles with distinct black border on white background)
    _, bin_rect = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
    cnts_rect, _ = cv2.findContours(bin_rect, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_rect:
        area = cv2.contourArea(c)
        if area < (total_area * 0.002) or area > (total_area * 0.15):
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw > w * 0.70 or bh > h * 0.35 or bh < 30 or bw < 40:
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        extent = area / (bw * bh) if bw * bh > 0 else 0

        # Narration boxes are strictly rectangular
        if extent > 0.85 and solidity > 0.92:
            crop = gray[by:by+bh, bx:bx+bw]
            bg_white = np.mean(crop > 205)
            ink_pct = np.mean(crop < 110)

            # Check for actual text characters inside
            pad_b = max(3, int(4 * scale))
            crop_clean = crop.copy()
            crop_clean[:pad_b, :] = 255; crop_clean[-pad_b:, :] = 255
            crop_clean[:, :pad_b] = 255; crop_clean[:, -pad_b:] = 255

            _, ink_crop = cv2.threshold(crop_clean, 110, 255, cv2.THRESH_BINARY_INV)
            num_c, _, stats_c, _ = cv2.connectedComponentsWithStats(ink_crop)
            num_valid = sum(1 for i in range(1, num_c) if stats_c[i][4] >= 25 and stats_c[i][2] >= 6 and stats_c[i][3] >= 6)

            if bg_white >= 0.70 and 0.015 <= ink_pct <= 0.30 and num_valid >= 3:
                boxes.append({
                    'box': (bx, by, bw, bh),
                    'dir': 'horizontal' if bw >= bh * 1.2 else 'vertical',
                    'score': 0.98,
                    'area': bw * bh,
                    'type': 'narration'
                })

    # 2. Speech Bubble Detection via Multi-Threshold Contours
    for thresh in [235, 220, 205]:
        _, bin_w = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(bin_w, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            if area < (total_area * 0.0015) or area > (total_area * 0.15):
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw > w * 0.70 or bh > h * 0.40 or bw < 35 or bh < 30:
                continue

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.70:
                continue

            crop = gray[by:by+bh, bx:bx+bw]
            bg_white = np.mean(crop > 205)
            ink_pct = np.mean(crop < 115)
            if bg_white < 0.65 or not (0.015 <= ink_pct <= 0.32):
                continue

            crop_clean = crop.copy()
            pad_b = max(3, int(4 * scale))
            crop_clean[:pad_b, :] = 255; crop_clean[-pad_b:, :] = 255
            crop_clean[:, :pad_b] = 255; crop_clean[:, -pad_b:] = 255

            _, ink_crop = cv2.threshold(crop_clean, 110, 255, cv2.THRESH_BINARY_INV)
            k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            ink_dilated = cv2.dilate(ink_crop, k2)
            num_c, _, stats_c, _ = cv2.connectedComponentsWithStats(ink_dilated)

            chars = []
            for i in range(1, num_c):
                cx, cy, cw, ch, ca = stats_c[i]
                if 8 <= cw <= bw * 0.80 and 8 <= ch <= bh * 0.80 and ca >= 25:
                    chars.append((cx, cy, cw, ch, ca))

            if len(chars) < 2:
                continue

            # Extract vertical text lines
            chars_y = sorted(chars, key=lambda ch: ch[1])
            v_lines = []
            used_v = set()
            for i1, c1 in enumerate(chars_y):
                if i1 in used_v: continue
                line = [c1]
                used_v.add(i1)
                curr = c1
                while True:
                    best_j = None
                    best_dist = float('inf')
                    for j, c2 in enumerate(chars_y):
                        if j in used_v: continue
                        dx = abs((c2[0] + c2[2]/2.0) - (curr[0] + curr[2]/2.0))
                        dy = c2[1] - (curr[1] + curr[3])
                        font_s = max(curr[2], curr[3], c2[2], c2[3])
                        ratio_h = min(curr[3], c2[3]) / max(curr[3], c2[3])
                        ratio_w = min(curr[2], c2[2]) / max(curr[2], c2[2])
                        if dx <= max(font_s * 0.45, 15) and -6 <= dy <= max(font_s * 0.85, 25) and ratio_h >= 0.40 and ratio_w >= 0.40:
                            dist = dy + dx * 2
                            if dist < best_dist:
                                best_dist = dist
                                best_j = j
                    if best_j is not None:
                        line.append(chars_y[best_j])
                        used_v.add(best_j)
                        curr = chars_y[best_j]
                    else:
                        break
                if len(line) >= 3 or (len(line) == 2 and line[0][4] >= 90):
                    span_y = (line[-1][1] + line[-1][3]) - line[0][1]
                    if span_y >= max(bh * 0.22, 35):
                        v_lines.append(line)

            # Extract horizontal text lines
            chars_x = sorted(chars, key=lambda ch: ch[0])
            h_lines = []
            used_h = set()
            for i1, c1 in enumerate(chars_x):
                if i1 in used_h: continue
                line = [c1]
                used_h.add(i1)
                curr = c1
                while True:
                    best_j = None
                    best_dist = float('inf')
                    for j, c2 in enumerate(chars_x):
                        if j in used_h: continue
                        dy = abs((c2[1] + c2[3]/2.0) - (curr[1] + curr[3]/2.0))
                        dx = c2[0] - (curr[0] + curr[2])
                        font_s = max(curr[2], curr[3], c2[2], c2[3])
                        ratio_h = min(curr[3], c2[3]) / max(curr[3], c2[3])
                        ratio_w = min(curr[2], c2[2]) / max(curr[2], c2[2])
                        if dy <= max(font_s * 0.45, 15) and -6 <= dx <= max(font_s * 0.85, 25) and ratio_h >= 0.40 and ratio_w >= 0.40:
                            dist = dx + dy * 2
                            if dist < best_dist:
                                best_dist = dist
                                best_j = j
                    if best_j is not None:
                        line.append(chars_x[best_j])
                        used_h.add(best_j)
                        curr = chars_x[best_j]
                    else:
                        break
                if (len(line) >= 3 or (len(line) == 2 and bw >= bh * 1.3)):
                    span_x = (line[-1][0] + line[-1][2]) - line[0][0]
                    max_h = max(c[3] for c in line)
                    if span_x >= max(bw * 0.22, 45) and max_h >= 14:
                        h_lines.append(line)

            # Add tight bounding box around the text lines
            for vl in v_lines:
                min_x = min(c[0] for c in vl)
                max_x = max(c[0] + c[2] for c in vl)
                min_y = min(c[1] for c in vl)
                max_y = max(c[1] + c[3] for c in vl)

                mx = min(min_x / bw, (bw - max_x) / bw)
                my = min(min_y / bh, (bh - max_y) / bh)
                if mx < 0.04 or my < 0.04:
                    continue

                pad_x = int(18 * scale)
                pad_y = int(18 * scale)
                final_x = max(0, bx + min_x - pad_x)
                final_y = max(0, by + min_y - pad_y)
                final_w = min(w, bx + max_x + pad_x) - final_x
                final_h = min(h, by + max_y + pad_y) - final_y

                boxes.append({
                    'box': (final_x, final_y, final_w, final_h),
                    'dir': 'vertical',
                    'score': 0.96,
                    'area': final_w * final_h,
                    'type': 'bubble'
                })

            for hl in h_lines:
                min_x = min(c[0] for c in hl)
                max_x = max(c[0] + c[2] for c in hl)
                min_y = min(c[1] for c in hl)
                max_y = max(c[1] + c[3] for c in hl)

                mx = min(min_x / bw, (bw - max_x) / bw)
                my = min(min_y / bh, (bh - max_y) / bh)
                if mx < 0.04 or my < 0.04:
                    continue

                pad_x = int(18 * scale)
                pad_y = int(18 * scale)
                final_x = max(0, bx + min_x - pad_x)
                final_y = max(0, by + min_y - pad_y)
                final_w = min(w, bx + max_x + pad_x) - final_x
                final_h = min(h, by + max_y + pad_y) - final_y

                boxes.append({
                    'box': (final_x, final_y, final_w, final_h),
                    'dir': 'horizontal',
                    'score': 0.96,
                    'area': final_w * final_h,
                    'type': 'bubble'
                })

    # Deduplicate via NMS & Containment
    boxes.sort(key=lambda c: c['area'], reverse=True)
    final = []
    for cand in boxes:
        cb = cand['box']
        keep = True
        for f in final:
            fb = f['box']
            if calculate_iou(cb, fb) > 0.35 or containment_ratio(cb, fb) > 0.70:
                keep = False
                break
        if keep:
            final.append(cand)

    return final

if __name__ == '__main__':
    images = sorted(glob.glob('image test/*.jpg'))
    out_dir = 'image test/output'
    os.makedirs(out_dir, exist_ok=True)

    print(f'Starting batch test on {len(images)} images...')
    t0 = time.time()
    total_detected = 0
    zero_pages = []

    for idx, img_path in enumerate(images, 1):
        fname = os.path.basename(img_path)
        try:
            boxes = detect_page(img_path)
            count = len(boxes)
            total_detected += count
            if count == 0:
                zero_pages.append(fname)

            # Draw and save visual output
            img_pil = Image.open(img_path)
            draw = ImageDraw.Draw(img_pil)
            for b_idx, b in enumerate(boxes, 1):
                bx, by, bw, bh = b['box']
                draw.rectangle([bx, by, bx+bw, by+bh], outline=(255, 0, 0), width=4)
                draw.rectangle([bx, max(0, by - 26), bx + 44, by], fill=(255, 0, 0))
                draw.text((bx + 6, max(0, by - 22)), f'#{b_idx}', fill=(255, 255, 255))
            img_pil.save(os.path.join(out_dir, f'out_{fname}.png'))

            print(f'[{idx:02d}/{len(images)}] {fname} -> {count} bubbles')
        except Exception as e:
            print(f'[{idx:02d}/{len(images)}] {fname} -> ERROR: {e}')

    duration = time.time() - t0
    print('=' * 60)
    print(f'Completed in {duration:.1f}s (avg {duration/len(images):.2f}s per page)')
    print(f'Total bubbles detected: {total_detected}')
    print(f'Pages with 0 bubbles: {len(zero_pages)} -> {zero_pages}')
