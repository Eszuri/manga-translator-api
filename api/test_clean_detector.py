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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total_area = h * w
    scale = max(w, h) / 1600.0

    candidates = []

    # -------------------------------------------------------------
    # 1. Narration Boxes (Rectangles with straight borders)
    # -------------------------------------------------------------
    _, bin_rect = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
    cnts_rect, _ = cv2.findContours(bin_rect, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_rect:
        area = cv2.contourArea(c)
        if area < (total_area * 0.002) or area > (total_area * 0.20):
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw > w * 0.70 or bh > h * 0.35 or bh < 30 or bw < 40:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        extent = area / (bw * bh) if bw * bh > 0 else 0

        # Highly rectangular
        if extent > 0.85 and solidity > 0.92:
            crop = gray[by:by+bh, bx:bx+bw]
            bg_white = np.mean(crop > 205)
            ink_pct = np.mean(crop < 110)
            
            # Reject empty margins/gutters
            if bx <= 2 or by <= 2 or (bx + bw) >= (w - 2):
                if ink_pct < 0.03:
                    continue

            # Must contain actual text characters inside
            pad_b = max(3, int(4 * scale))
            crop_clean = crop.copy()
            crop_clean[:pad_b, :] = 255; crop_clean[-pad_b:, :] = 255
            crop_clean[:, :pad_b] = 255; crop_clean[:, -pad_b:] = 255

            _, ink_crop = cv2.threshold(crop_clean, 110, 255, cv2.THRESH_BINARY_INV)
            num_c, _, stats_c, _ = cv2.connectedComponentsWithStats(ink_crop)
            num_valid = sum(1 for i in range(1, num_c) if stats_c[i][4] >= 25 and stats_c[i][2] >= 6 and stats_c[i][3] >= 6)

            if bg_white >= 0.70 and 0.015 <= ink_pct <= 0.30 and num_valid >= 3:
                candidates.append({
                    'box': (bx, by, bw, bh),
                    'dir': 'horizontal' if bw >= bh * 1.2 else 'vertical',
                    'score': 0.98,
                    'area': bw * bh,
                    'type': 'narration'
                })

    # -------------------------------------------------------------
    # 2. Text line detection in speech bubbles & sound effects
    # -------------------------------------------------------------
    _, ink = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    ink_dilated = cv2.dilate(ink, k2)
    num, _, stats, _ = cv2.connectedComponentsWithStats(ink_dilated)

    min_dim = max(6, int(8 * scale))
    max_dim = int(160 * scale)
    chars = []

    for i in range(1, num):
        cx, cy, cw, ch, ca = stats[i]
        if min_dim <= cw <= max_dim and min_dim <= ch <= max_dim and ca >= 20:
            aspect = ch / cw if cw > 0 else 1.0
            fill = ca / (cw * ch)
            if 0.35 <= aspect <= 3.2 and 0.10 <= fill <= 0.85:
                pad = max(6, int(10 * scale))
                patch = gray[max(0, cy - pad):min(h, cy + ch + pad), max(0, cx - pad):min(w, cx + cw + pad)]
                bg_w = np.mean(patch > 205)
                bg_d = np.mean(patch < 65)
                if bg_w >= 0.65 or bg_d >= 0.55:
                    chars.append({
                        'x': cx, 'y': cy, 'w': cw, 'h': ch,
                        'cx': cx + cw/2.0, 'cy': cy + ch/2.0,
                        'is_dark': bg_d >= 0.55
                    })

    # Group into vertical lines (Tategaki)
    chars_by_y = sorted(chars, key=lambda c: c['y'])
    v_lines = []
    used_v = set()

    for i, c1 in enumerate(chars_by_y):
        if i in used_v: continue
        line = [c1]
        used_v.add(i)
        curr = c1
        while True:
            best_j = None
            best_dist = float('inf')
            for j, c2 in enumerate(chars_by_y):
                if j in used_v: continue
                if c2['is_dark'] != c1['is_dark']: continue
                dx = abs(c2['cx'] - curr['cx'])
                dy = c2['y'] - (curr['y'] + curr['h'])
                font_s = max(curr['w'], curr['h'], c2['w'], c2['h'])
                max_dx = max(font_s * 0.65, 18)
                max_dy = max(font_s * 1.8, 28)
                if dx <= max_dx and -8 <= dy <= max_dy:
                    dist = dy + dx * 2
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
            if best_j is not None:
                line.append(chars_by_y[best_j])
                used_v.add(best_j)
                curr = chars_by_y[best_j]
            else:
                break

        # A valid vertical line: >= 2 chars, or 1 if large shout/SFX
        if len(line) >= 2 or (len(line) == 1 and line[0]['h'] >= int(38 * scale)):
            v_lines.append(line)

    # Group into horizontal lines (Yokogaki)
    unused_for_h = [c for idx, c in enumerate(chars) if idx not in used_v]
    chars_by_x = sorted(unused_for_h, key=lambda c: c['x'])
    h_lines = []
    used_h = set()

    for i, c1 in enumerate(chars_by_x):
        if i in used_h: continue
        line = [c1]
        used_h.add(i)
        curr = c1
        while True:
            best_j = None
            best_dist = float('inf')
            for j, c2 in enumerate(chars_by_x):
                if j in used_h: continue
                if c2['is_dark'] != c1['is_dark']: continue
                dy = abs(c2['cy'] - curr['cy'])
                dx = c2['x'] - (curr['x'] + curr['w'])
                font_s = max(curr['w'], curr['h'], c2['w'], c2['h'])
                max_dy = max(font_s * 0.65, 18)
                max_dx = max(font_s * 1.8, 28)
                if dy <= max_dy and -8 <= dx <= max_dx:
                    dist = dx + dy * 2
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
            if best_j is not None:
                line.append(chars_by_x[best_j])
                used_h.add(best_j)
                curr = chars_by_x[best_j]
            else:
                break

        if len(line) >= 3:
            h_lines.append(line)

    # Merge vertical lines in same bubble
    used_vl = set()
    for i, l1 in enumerate(v_lines):
        if i in used_vl: continue
        group = list(l1)
        used_vl.add(i)

        l1_x1 = min(c['x'] for c in l1)
        l1_y1 = min(c['y'] for c in l1)
        l1_x2 = max(c['x'] + c['w'] for c in l1)
        l1_y2 = max(c['y'] + c['h'] for c in l1)

        changed = True
        while changed:
            changed = False
            for j, l2 in enumerate(v_lines):
                if j in used_vl: continue
                l2_x1 = min(c['x'] for c in l2)
                l2_y1 = min(c['y'] for c in l2)
                l2_x2 = max(c['x'] + c['w'] for c in l2)
                l2_y2 = max(c['y'] + c['h'] for c in l2)

                dx = max(0, max(l1_x1 - l2_x2, l2_x1 - l1_x2))
                dy = max(0, max(l1_y1 - l2_y2, l2_y1 - l1_y2))
                overlap_y = min(l1_y2, l2_y2) - max(l1_y1, l2_y1)

                if dx <= int(60 * scale) and (overlap_y > 0 or dy <= int(30 * scale)):
                    group.extend(l2)
                    used_vl.add(j)
                    changed = True

        gx1 = min(c['x'] for c in group)
        gy1 = min(c['y'] for c in group)
        gx2 = max(c['x'] + c['w'] for c in group)
        gy2 = max(c['y'] + c['h'] for c in group)

        pad_x = int(18 * scale)
        pad_y = int(20 * scale)
        bx = max(0, gx1 - pad_x)
        by = max(0, gy1 - pad_y)
        bw = min(w, gx2 + pad_x) - bx
        bh = min(h, gy2 + pad_y) - by

        crop = gray[by:by+bh, bx:bx+bw]
        bg_w = np.mean(crop > 205)
        bg_d = np.mean(crop < 65)

        # Bubble background validation
        if bg_w >= 0.62 or bg_d >= 0.38:
            candidates.append({
                'box': (bx, by, bw, bh),
                'dir': 'vertical',
                'score': 0.96,
                'area': bw * bh,
                'type': 'bubble'
            })

    # Merge horizontal lines
    for line in h_lines:
        gx1 = min(c['x'] for c in line)
        gy1 = min(c['y'] for c in line)
        gx2 = max(c['x'] + c['w'] for c in line)
        gy2 = max(c['y'] + c['h'] for c in line)

        pad_x = int(18 * scale)
        pad_y = int(16 * scale)
        bx = max(0, gx1 - pad_x)
        by = max(0, gy1 - pad_y)
        bw = min(w, gx2 + pad_x) - bx
        bh = min(h, gy2 + pad_y) - by

        crop = gray[by:by+bh, bx:bx+bw]
        bg_w = np.mean(crop > 205)
        bg_d = np.mean(crop < 65)

        if bg_w >= 0.62 or bg_d >= 0.38:
            candidates.append({
                'box': (bx, by, bw, bh),
                'dir': 'horizontal',
                'score': 0.96,
                'area': bw * bh,
                'type': 'bubble'
            })

    # Deduplicate via NMS
    candidates.sort(key=lambda c: c['area'], reverse=True)
    final = []
    for cand in candidates:
        keep = True
        cb = cand['box']
        for f in final:
            fb = f['box']
            if calculate_iou(cb, fb) > 0.35 or containment_ratio(cb, fb) > 0.70:
                keep = False
                break
        if keep:
            final.append(cand)

    print(f'Total detected on {img_path}: {len(final)}')
    for f in final:
        print(' ', f['box'], f['dir'], f['type'])
    return final

if __name__ == '__main__':
    detect_page('image test/015.jpg')
    detect_page('image test/019.jpg')
