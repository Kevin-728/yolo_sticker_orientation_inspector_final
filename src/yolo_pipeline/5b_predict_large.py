"""
5b_predict_large.py
--------------------
Full-image inference for the LARGE-OBJECT model (fuse_cover, green_label).

Unlike the small-object pipeline (5_predict.py), this does NOT crop/tile the
image. The whole raw image is handed to the model in one call; Ultralytics
internally letterbox-resizes it to --imgsz and automatically rescales the
returned box coordinates back to the ORIGINAL input image's pixel space
before returning them (this is standard `results[0].boxes` behaviour — no
manual scale/offset math is needed here, unlike the tiled pipeline where we
have to invert our own manual crop+upscale).

Because there's no tiling, there's no cross-tile duplication problem for
this pass — each object is seen exactly once. A light standard NMS is still
applied per class purely as a safety net (e.g. in case two products' fuse
covers are close enough for the model to emit an overlapping double-fire on
one of them), not to solve a structural duplication issue.

Can be run standalone for debugging, or imported by 5_predict.py to merge
its output with the small-object tiled pass.

Usage (standalone)
-------------------
    python 5b_predict_large.py --model large_best.pt --img image.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Classes this model is responsible for (must match 3b_prepare_large_dataset.py)
LARGE_CLASSES = {2, 3}   # green_label, fuse_cover


def predict_image_large(model: YOLO, img_path: Path,
                        imgsz: int = 1536,
                        conf: float = 0.25, iou: float = 0.45) -> list:
    """
    Returns list of (cls_id, cx_n, cy_n, w_n, h_n, score), normalised to the
    original raw image, restricted to LARGE_CLASSES.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    ih, iw = img.shape[:2]

    # Single full-image call — Ultralytics handles the letterbox resize to
    # imgsz internally and returns box coordinates already rescaled back to
    # this original (iw, ih) image, so no manual coordinate mapping needed.
    results = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, verbose=False)

    out = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in LARGE_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = float(box.conf[0])
            out.append((
                cls_id,
                float((x1 + x2) / 2 / iw),
                float((y1 + y2) / 2 / ih),
                float((x2 - x1) / iw),
                float((y2 - y1) / ih),
                score,
            ))

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Path to large-object weights')
    p.add_argument('--img',   required=True, help='Input image path')
    p.add_argument('--out',   default=None, help='Output .txt path')
    p.add_argument('--imgsz', type=int, default=1536)
    p.add_argument('--conf',  type=float, default=0.25)
    p.add_argument('--iou',   type=float, default=0.45)
    a = p.parse_args()

    img_path = Path(a.img)
    out_path = Path(a.out) if a.out else img_path.with_suffix('.large.txt')

    model      = YOLO(a.model)
    detections = predict_image_large(model, img_path, a.imgsz, a.conf, a.iou)

    with open(out_path, 'w') as f:
        for cls_id, cx, cy, w, h, _score in detections:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    print(f"Wrote {len(detections)} detections -> {out_path}")
    cls_names = {2: 'green_label', 3: 'fuse_cover'}
    for cls_id, cx, cy, w, h, score in detections:
        name = cls_names.get(cls_id, f'cls{cls_id}')
        print(f"  {name:<20s} cx={cx:.4f} cy={cy:.4f} "
              f"w={w:.4f} h={h:.4f}  conf={score:.3f}")


if __name__ == '__main__':
    main()
