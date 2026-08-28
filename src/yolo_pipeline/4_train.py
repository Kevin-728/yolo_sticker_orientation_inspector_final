"""
4_train.py
----------
Train a YOLOv11s detector on the tiled, upsampled sticker dataset (Option A).

Usage
-----
    python 4_train.py
    python 4_train.py --model yolo11m.pt --epochs 150 --batch 8

    # Resume an interrupted run (true resume=True continuation — same
    # optimizer state / LR schedule position / epoch counter as when it
    # stopped, NOT a fresh run seeded with those weights):
    python 4_train.py --resume runs/detect/sticker_v1-3/weights/last.pt

The script downloads the pretrained backbone automatically on first run
(requires internet access).

--resume vs --model <last.pt> (no --resume)
--------------------------------------------
These look similar but do different things:

  --resume PATH        True resume=True. Locks ALL hyperparameters to what
                        that run was already using (imgsz, batch, augment,
                        data path, epoch ceiling). Writes back into PATH's
                        existing run directory — no new --name is created.
                        Use this when you killed a run early and want to
                        keep going as if it had never stopped.

  --model PATH          Loads PATH as initial weights for a brand-new run:
  (no --resume)          fresh optimizer, fresh LR warmup, new --name run
                        directory, epoch counter restarts at 0. Use this
                        only when you're deliberately changing something
                        (batch size, augmentation, etc.) — which is exactly
                        why the earlier sticker_v1 -> sticker_v1-2 ->
                        sticker_v1-3 runs above used this form.

Trained weights are saved to:
    runs/detect/<name>/weights/best.pt   ← use this in 5_predict.py
    runs/detect/<name>/weights/last.pt

epochs=100 is a CEILING, not a target
--------------------------------------
patience=20 (below) stops training automatically once validation mAP hasn't
improved for 20 consecutive epochs. For a dataset this size, convergence
typically happens well before 100 — commonly in the 40-70 range. Leave
epochs=100 as a safety ceiling; patience does the real stopping.

Hardware note (laptop RTX 4070, 8 GB VRAM)
--------------------------------------------
  imgsz=640, batch=16  →  ~6 GB VRAM  (default, recommended sweet spot)
  imgsz=640, batch=8   →  ~3.5 GB VRAM (use if batch=16 is unstable)
  imgsz=640, batch=32  →  OOM risk — avoid

Laptop GPUs can thermal-throttle under sustained load. If per-epoch time
noticeably increases after the first 10-15 minutes, check cooling / power
mode rather than assuming a code problem.

Why mosaic is disabled and scale is reduced (Option A specific)
-------------------------------------------------------------------
mosaic=0.5   REMOVED (set to 0.0). Mosaic augmentation combines four source
             tiles into one sample, each shrunk to fit a quadrant — this
             would shrink already-tiny upsampled stickers (~20-22 px) right
             back down toward the detection floor, undoing the benefit of
             Option A's upsampling.
scale=0.3    REDUCED to 0.15. Scale jitter of ±30% can shrink a ~20 px
             sticker to ~14 px, below the reliable floor. ±15% keeps worst
             case around ~17 px.
degrees=180  KEPT. Products arrive at any rotation → train on full range.
fliplr/flipud=0.5  KEPT. Detection only needs localisation, not orientation,
             so flips are safe.
"""

import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', default=None, metavar='LAST_PT',
                   help='Path to last.pt of an INTERRUPTED run to resume '
                        '(e.g. runs/detect/sticker_v1-3/weights/last.pt). '
                        'This is a true resume=True continuation: same '
                        'optimizer state, LR schedule position, and epoch '
                        'counter as when training stopped — NOT a fresh run '
                        'seeded with those weights. Only valid if no '
                        'training args (imgsz/batch/augmentation/etc.) have '
                        'changed since that run was launched, since '
                        'resume=True reuses the original run\'s saved '
                        'args.yaml and writes back into the SAME run '
                        'directory rather than creating a new --name. '
                        'When --resume is set, all other CLI args below '
                        '(--model/--data/--epochs/--imgsz/--batch/--name/'
                        '--device) are ignored.')
    p.add_argument('--model',   default='yolo11s.pt',
                   help='Pretrained YOLO weights to fine-tune '
                        '(default: yolo11s.pt, downloaded automatically)')
    p.add_argument('--data',    default='dataset.yaml',
                   help='Path to dataset.yaml (default: dataset.yaml)')
    p.add_argument('--epochs',  type=int,   default=100,
                   help='Ceiling on training epochs (default 100). '
                        'patience below will usually stop training earlier.')
    p.add_argument('--imgsz',   type=int,   default=640,
                   help='Training image size — must match NET_SIZE in '
                        '3_tile_dataset.py / 5_predict.py (default 640)')
    p.add_argument('--batch',   type=int,   default=16,
                   help='Batch size (default 16; sized for laptop RTX 4070 '
                        '8GB VRAM at imgsz=640)')
    p.add_argument('--name',    default='sticker_v1',
                   help='Run name written under runs/detect/ (default sticker_v1)')
    p.add_argument('--device',  default='0',
                   help='CUDA device index, or "cpu" (default 0)')
    a = p.parse_args()

    if a.resume:
        print(f"[INFO] Resuming interrupted run from: {a.resume}")
        print("[INFO] resume=True — all other CLI args ignored; reusing "
              "the original run's saved config (args.yaml) and writing "
              "back into its existing run directory.")
        model = YOLO(a.resume)
        model.train(resume=True)
        return

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
        scale         = 0.15,  # reduced from 0.3 — protects tiny upsampled stickers
        mosaic        = 0.0,   # disabled — protects tiny upsampled stickers

        # ── Training control ───────────────────────────────────────────────
        patience      = 20,    # early stopping: stop if no improvement for 20 epochs
        save_period   = 10,    # save checkpoint every 10 epochs
        workers       = 4,
    )


if __name__ == '__main__':
    main()
