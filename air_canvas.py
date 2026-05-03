"""
Air Canvas - Draw with your finger using OpenCV + MediaPipe
============================================================
Controls:
  - Hold index finger up, curl others to DRAW
  - Hold index + middle finger up to HOVER/SELECT (no drawing)
  - Move finger to the TOP BAR to select color / eraser / brush size
  - Press 'c' to clear canvas
  - Press 'q' to quit

Toolbar layout (left to right):
  [Project Logo] | [12 Color Swatches] | [Eraser] | [S M L brush]

Requirements:
  pip install opencv-python mediapipe numpy cairosvg

Place these files in the same folder as this script:
  project_logo.svg   (SVG logo)
  eraiser_logo.svg   (SVG eraser icon)
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os

try:
    import cairosvg
    CAIROSVG_OK = True
except ImportError:
    CAIROSVG_OK = False
    print("WARNING: cairosvg not found. Install: pip install cairosvg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Mediapipe ────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
)

# ── 12 colours (BGR) ─────────────────────────────────────────────────────────
COLORS = [
    (255,  50,  50),   # Blue
    (30,  30, 220),   # Red
    (200,  30, 220),   # Pink
    (30, 200,  30),   # Green
    (255, 237, 41),   # Yellow
    (220, 160,  30),   # Cyan
    (180,  30,  30),   # Dark Blue
    (220,  30, 220),   # Magenta
    (180, 130,  30),   # Steel Blue
    (0, 140, 255),   # Orange
    (255, 255, 197),   # Bright Yellow
    (42,  42, 165),   # Brown
]
COLOR_NAMES = [
    "Blue", "Red", "Pink", "Green", "Yellow", "Cyan",
    "D.Blue", "Magenta", "Steel", "Orange", "Br.Yel", "Brown",
]

# ── Brush / eraser sizes ─────────────────────────────────────────────────────
BRUSH_SIZES = [4, 10, 20]
ERASER_SIZES = [20, 40, 70]
BRUSH_LABELS = ["S", "M", "L"]

# ── Toolbar layout constants ─────────────────────────────────────────────────
BAR_H = 80
LOGO_W = 130
COL_W = 44
COL_PAD = 3
ERASER_W = 70
BRUSH_W = 42
GAP = 8

# ── Smoothing ────────────────────────────────────────────────────────────────
SMOOTH_ALPHA = 0.55
_px, _py = None, None

# ── Stroke storage: each entry is (x, y, color_idx, brush_size) or None ──────
all_strokes = deque(maxlen=8192)


# ════════════════════════════════════════════════════════════════════════════
# Asset loaders
# ════════════════════════════════════════════════════════════════════════════

def load_image_bgra(path, w, h):
    """
    Load any image (PNG/JPG or SVG) as a BGRA array resized to (w, h).
    Returns None if missing or load fails.
    """
    if not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".svg":
            if not CAIROSVG_OK:
                return None
            raw = open(path, "rb").read()
            png = cairosvg.svg2png(
                bytestring=raw, output_width=w, output_height=h)
            arr = np.frombuffer(png, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return cv2.resize(img, (w, h))
    except Exception as e:
        print(f"Asset load error ({path}): {e}")
        return None


def alpha_composite(dst_bgr, src_bgra, x=0, y=0):
    """Alpha-composite src_bgra onto dst_bgr (BGR) in-place."""
    sh, sw = src_bgra.shape[:2]
    dh, dw = dst_bgr.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, dw), min(y + sh, dh)
    sx0, sy0 = x0 - x, y0 - y
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
    if x1 <= x0 or y1 <= y0:
        return
    src = src_bgra[sy0:sy1, sx0:sx1]
    a = src[:, :, 3:4].astype(np.float32) / 255.0
    dst_bgr[y0:y1, x0:x1] = (
        src[:, :, :3].astype(np.float32) * a +
        dst_bgr[y0:y1, x0:x1].astype(np.float32) * (1.0 - a)
    ).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════════
# Toolbar
# ════════════════════════════════════════════════════════════════════════════

def compute_layout(W):
    lay = {}
    lay["logo"] = (0, LOGO_W)
    x = LOGO_W + GAP

    lay["colors"] = []
    for _ in range(len(COLORS)):
        lay["colors"].append((x, x + COL_W))
        x += COL_W + COL_PAD
    x += GAP

    lay["eraser"] = (x, x + ERASER_W)
    x += ERASER_W + GAP

    lay["brushes"] = []
    for _ in range(len(BRUSH_SIZES)):
        lay["brushes"].append((x, x + BRUSH_W))
        x += BRUSH_W + COL_PAD

    return lay


def build_toolbar(W, lay, sel_color, sel_brush, is_eraser, logo_img, eraser_img):
    # Pure black background
    bar = np.zeros((BAR_H, W, 3), dtype=np.uint8)
    Y1, Y2 = 7, BAR_H - 7

    # Logo
    if logo_img is not None:
        lh = BAR_H - 10
        lw = int(logo_img.shape[1] * lh / logo_img.shape[0])
        alpha_composite(bar, cv2.resize(logo_img, (lw, lh)), 5, 5)

    # 12 colour swatches
    for i, (x1, x2) in enumerate(lay["colors"]):
        cv2.rectangle(bar, (x1, Y1), (x2, Y2), COLORS[i], -1)
        active = (not is_eraser) and (i == sel_color)
        cv2.rectangle(bar, (x1, Y1), (x2, Y2),
                      (0, 240, 240) if active else (50, 50, 50),
                      3 if active else 1)

    # Eraser — icon only, no filled background box, cyan ring when active
    ex1, ex2 = lay["eraser"]
    if is_eraser:
        cv2.rectangle(bar, (ex1, Y1), (ex2, Y2), (0, 240, 240), 3)
    if eraser_img is not None:
        pad = 5
        ew = ex2 - ex1 - pad * 2
        eh = Y2 - Y1 - pad * 2
        if ew > 0 and eh > 0:
            resized = cv2.resize(eraser_img, (ew, eh))
            alpha_composite(bar, resized, ex1 + pad, Y1 + pad)
    else:
        # Fallback drawn eraser shape
        cx = (ex1 + ex2) // 2
        cy = (Y1 + Y2) // 2
        cv2.rectangle(bar, (cx - 15, cy - 8),
                      (cx + 15, cy + 8), (210, 210, 210), -1)
        cv2.rectangle(bar, (cx - 15, cy - 8),
                      (cx - 1,  cy + 8), (150, 110, 90),  -1)

    # Brush size buttons
    for i, (bx1, bx2) in enumerate(lay["brushes"]):
        active = (i == sel_brush)
        cv2.rectangle(bar, (bx1, Y1), (bx2, Y2),
                      (0, 240, 240) if active else (40, 40, 40),
                      3 if active else 1)
        cx = (bx1 + bx2) // 2
        cy = (Y1 + Y2) // 2
        cv2.circle(bar, (cx, cy - 4), min(BRUSH_SIZES[i] + 2, 17),
                   (210, 210, 210), -1)
        cv2.putText(bar, BRUSH_LABELS[i], (bx1 + 13, Y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)

    return bar


# ════════════════════════════════════════════════════════════════════════════
# Hit-test
# ════════════════════════════════════════════════════════════════════════════

def hit_test(x, lay):
    for i, (x1, x2) in enumerate(lay["colors"]):
        if x1 <= x <= x2:
            return ("color", i)
    ex1, ex2 = lay["eraser"]
    if ex1 <= x <= ex2:
        return ("eraser", None)
    for i, (bx1, bx2) in enumerate(lay["brushes"]):
        if bx1 <= x <= bx2:
            return ("brush", i)
    return None


# ════════════════════════════════════════════════════════════════════════════
# Hand utils
# ════════════════════════════════════════════════════════════════════════════

def fingers_up(lm, handedness="Right"):
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    up = []
    if handedness == "Right":
        up.append(lm[tips[0]].x < lm[pips[0]].x)
    else:
        up.append(lm[tips[0]].x > lm[pips[0]].x)
    for t, p in zip(tips[1:], pips[1:]):
        up.append(lm[t].y < lm[p].y)
    return up


def smooth(cx, cy):
    global _px, _py
    if _px is None:
        _px, _py = cx, cy
    else:
        _px = int(SMOOTH_ALPHA * cx + (1 - SMOOTH_ALPHA) * _px)
        _py = int(SMOOTH_ALPHA * cy + (1 - SMOOTH_ALPHA) * _py)
    return _px, _py


# ════════════════════════════════════════════════════════════════════════════
# Canvas helpers
# ════════════════════════════════════════════════════════════════════════════

def replay_strokes(canvas):
    """Redraw all strokes onto a fresh white canvas using per-point brush size."""
    canvas[:] = 255
    pts = list(all_strokes)
    for j in range(1, len(pts)):
        p0, p1 = pts[j - 1], pts[j]
        if p0 is None or p1 is None:
            continue
        x0, y0, cidx, bsize = p0
        x1, y1, _,    _ = p1
        cv2.line(canvas, (x0, y0), (x1, y1), COLORS[cidx], bsize * 2)


def erase_strokes(cx, cy, radius):
    """Remove stored stroke points within radius pixels of (cx, cy)."""
    new_q = deque(maxlen=all_strokes.maxlen)
    prev_was_none = True
    for pt in all_strokes:
        if pt is None:
            if not prev_was_none:
                new_q.append(None)
            prev_was_none = True
            continue
        px, py, cidx, bsize = pt
        if ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 <= radius:
            if not prev_was_none:
                new_q.append(None)
            prev_was_none = True
        else:
            new_q.append(pt)
            prev_was_none = False
    all_strokes.clear()
    all_strokes.extend(new_q)


# ════════════════════════════════════════════════════════════════════════════
# Main loop
# ════════════════════════════════════════════════════════════════════════════

def main():
    global _px, _py

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    ret, frame = cap.read()
    if not ret:
        print("Cannot open webcam.")
        return

    H, W = frame.shape[:2]
    canvas_h = H - BAR_H
    canvas = np.ones((canvas_h, W, 3), dtype=np.uint8) * 255

    # ── Load assets ───────────────────────────────────────────────────────
    print("Loading assets…")

    logo_img = load_image_bgra(
        os.path.join(SCRIPT_DIR, "project_logo.svg"),
        LOGO_W - 10, BAR_H - 10)

    eraser_img = load_image_bgra(
        os.path.join(SCRIPT_DIR, "eraiser_logo.svg"),
        ERASER_W - 10, BAR_H - 16)

    print("Assets loaded.")

    layout = compute_layout(W)
    sel_color = 0
    sel_brush = 1
    is_eraser = False

    def rebuild():
        return build_toolbar(W, layout, sel_color, sel_brush,
                             is_eraser, logo_img, eraser_img)

    toolbar = rebuild()
    print("Air Canvas ready!  'c' = clear  |  'q' = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        frame[:BAR_H] = toolbar.copy()
        draw_zone = frame[BAR_H:]

        # ── Hand tracking ─────────────────────────────────────────────────
        if result.multi_hand_landmarks:
            for hlm, hinfo in zip(result.multi_hand_landmarks,
                                  result.multi_handedness):
                lm = hlm.landmark
                hand = hinfo.classification[0].label
                up = fingers_up(lm, hand)

                ix = int(lm[8].x * W)
                iy = int(lm[8].y * H)
                sx, sy = smooth(ix, iy)

                mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

                in_bar = sy < BAR_H

                # HOVER / SELECT — index + middle up
                if up[1] and up[2]:
                    _px = _py = None
                    all_strokes.append(None)
                    if in_bar:
                        hit = hit_test(sx, layout)
                        if hit:
                            kind, idx = hit
                            if kind == "color":
                                sel_color = idx
                                is_eraser = False
                                toolbar = rebuild()
                            elif kind == "eraser":
                                is_eraser = True
                                toolbar = rebuild()
                            elif kind == "brush":
                                sel_brush = idx
                                toolbar = rebuild()
                    cv2.circle(frame, (sx, sy), 12, (0, 240, 240), 2)

                # DRAW — only index up
                elif up[1] and not up[2]:
                    if not in_bar:
                        dy = sy - BAR_H
                        if is_eraser:
                            r = ERASER_SIZES[sel_brush]
                            erase_strokes(sx, dy, r)
                            replay_strokes(canvas)
                            cv2.circle(draw_zone, (sx, dy),
                                       r, (180, 180, 180), 2)
                            cv2.circle(draw_zone, (sx, dy),
                                       3, (100, 100, 100), -1)
                        else:
                            all_strokes.append(
                                (sx, dy, sel_color, BRUSH_SIZES[sel_brush]))
                            cv2.circle(draw_zone, (sx, dy),
                                       BRUSH_SIZES[sel_brush],
                                       COLORS[sel_color], -1)
                else:
                    _px = _py = None
                    all_strokes.append(None)
        else:
            _px = _py = None

        # ── Replay all strokes ────────────────────────────────────────────
        replay_strokes(canvas)

        # ── Blend canvas onto camera feed ─────────────────────────────────
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        mask_i = cv2.bitwise_not(mask)
        fg = cv2.bitwise_and(canvas,    canvas,    mask=mask)
        bg = cv2.bitwise_and(draw_zone, draw_zone, mask=mask_i)
        draw_zone[:] = cv2.add(fg, bg)

        # ── HUD ───────────────────────────────────────────────────────────
        tool_s = "Eraser" if is_eraser else COLOR_NAMES[sel_color]
        hud = f"Tool:{tool_s}  Size:{BRUSH_LABELS[sel_brush]}"
        cv2.putText(frame, hud, (W - 320, BAR_H + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 3)
        cv2.putText(frame, hud, (W - 320, BAR_H + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1)
        cv2.putText(frame, "2-finger=select  1-finger=draw  c=clear  q=quit",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (80, 80, 80), 1)

        cv2.imshow("Air Canvas", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            all_strokes.clear()
            canvas[:] = 255
            print("Canvas cleared.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
