"""
4b_train_large.py
------------------
Train a YOLOv11s detector for the LARGE objects (fuse_cover, green_label)
on the full-image dataset produced by 3b_prepare_large_dataset.py.

This is a separate model from the small-object tiled detector trained by
4_train.py. It never sees tiled/upsampled crops — just whole images resized
by Ultralytics' normal imgsz handling, since fuse_cover/green_label are
large enough to survive that downscale comfortably (unlike qrcode/
yellow_triangle, which is why THEY need the tiling pipeline and this model
does not).

Usage
-----
    python 4b_train_large.py --data /path/to/large_object_dataset/dataset.yaml

Resolution choice
------------------
imgsz=1536 (default here) keeps a ~1800-2000px fuse_cover comfortably large
after downscale, and is well above what a ~50px green_label needs. Raise to
1920 or drop to 1280 if your GPU can't fit 1536 at a workable batch size —
see the VRAM notes below (same 8GB RTX 4070 assumption as 4_train.py).

Hardware note (laptop RTX 4070, 8 GB VRAM)
--------------------------------------------
Larger imgsz costs much more VRAM per image than the small-object model at
640. Recommended starting point:
  imgsz=1536, batch=4   → try first
  imgsz=1536, batch=2   → fall back if OOM
  imgsz=1280, batch=8   → fall back if 1536 is unworkable at any batch size

Why augmentation differs from 4_train.py
------------------------------------------
mosaic=0.0    KEPT disabled for consistency, though for large objects mosaic
              is less actively harmful than for tiny stickers — left off
              simply to keep training behaviour predictable and comparable.
scale=0.3     Default Ultralytics scale jitter is fine here (NOT reduced to
              0.15 like the small-object model) — fuse_cover/green_label are
              nowhere near the detection floor, so aggressive scale jitter
              is safe and improves robustness instead of risking recall.
degrees=180   KEPT — products still arrive at any rotation.
fliplr/flipud=0.5  KEPT — detection only needs localisation.
"""

import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',   default='yolo11s.pt',
                   help='Pretrained YOLO weights to fine-tune '
                        '(default: yolo11s.pt, downloaded automatically)')
    p.add_argument('--data',    required=True,
                   help='Path to dataset.yaml for the large-object dataset '
                        '(from 3b_prepare_large_dataset.py)')
    p.add_argument('--epochs',  type=int,   default=100,
                   help='Ceiling on training epochs (default 100). '
                        'patience below will usually stop training earlier.')
    p.add_argument('--imgsz',   type=int,   default=1536,
                   help='Training image size (default 1536). Must match '
                        '--imgsz used in 5b_predict_large.py.')
    p.add_argument('--batch',   type=int,   default=4,
                   help='Batch size (default 4; large imgsz costs much more '
                        'VRAM per image than the small-object 640 model)')
    p.add_argument('--name',    default='sticker_large_v1',
                   help='Run name written under runs/detect/ '
                        '(default sticker_large_v1)')
    p.add_argument('--device',  default='0',
                   help='CUDA device index, or "cpu" (default 0)')
    a = p.parse_args()

    model = YOLO(a.model)
    model.train(
        data          = a.data,
        imgsz         = a.imgsz,
        epochs        = a.epochs,
        batch         = a.batch,
        device        = a.device,
        name          = a.name,

        # ── Augmentation ──────────────────────────────────────────────────
        degrees       = 180,   # full rotation range
        fliplr        = 0.5,
        flipud        = 0.5,
        scale         = 0.3,   # default jitter — large objects tolerate it fine
        mosaic        = 0.0,   # disabled for consistency with small-object model

        # ── Training control ───────────────────────────────────────────────
        patience      = 20,    # early stopping: stop if no improvement for 20 epochs
        save_period   = 10,    # save checkpoint every 10 epochs
        workers       = 4,
    )


if __name__ == '__main__':
    main()
