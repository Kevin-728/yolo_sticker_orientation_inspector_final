"""
3b_prepare_large_dataset.py
----------------------------
Build the training set for the LARGE-OBJECT model (fuse_cover, green_label).

Why this exists
----------------
fuse_cover (~1800-2000px raw) and green_label (50px+ raw) are far bigger than
CROP_SIZE=320 used for the small-object tiled pipeline. Tiling them produces
fragmented, contradictory labels (see 3_tile_dataset.py docstring). These two
classes don't have a small-object-detection-floor problem in the first place
— they're easily visible even after a large downscale — so they get their
own, much simpler dataset: full, un-tiled images, resized as a whole to the
large-model's training resolution (e.g. 1536px) by Ultralytics internally.

What this script does
----------------------
Takes the OUTPUT of 1_split_dataset.py (images/labels split into train/val,
full-resolution, un-tiled) and copies it to a new dataset root, but with
every label file filtered down to ONLY fuse_cover / green_label lines.
qrcode / yellow_triangle lines are dropped entirely — at large-image training
resolution those objects are only 1-2px, invisible and useless as training
signal for this model, and this model is never asked to detect them.

Images are copied unchanged (full native resolution) — no cropping, no
resizing. Resizing to the training resolution is left to Ultralytics'
standard `imgsz` handling in 4b_train_large.py, exactly like the model would
do for any conventional non-tiled dataset.

Usage
-----
    python 3b_prepare_large_dataset.py \
        --src /path/to/split_dataset \
        --dst /path/to/large_object_dataset
"""

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

# 2 = green_label, 3 = fuse_cover  (per CLASS_MAP in yolo_converter.py)
LARGE_CLASSES = {2, 3}


def filter_label_file(src_txt: Path, dst_txt: Path, large_classes=LARGE_CLASSES):
    """Copy only large-class lines from src_txt to dst_txt. Returns True if
    at least one line was kept (i.e. this image is useful for training)."""
    kept = []
    if src_txt.exists():
        for line in src_txt.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            if int(parts[0]) in large_classes:
                kept.append(line)
    dst_txt.write_text('\n'.join(kept) + ('\n' if kept else ''))
    return len(kept) > 0


def process_split(src: Path, dst: Path, split: str):
    src_img = src / 'images' / split
    src_lbl = src / 'labels' / split
    if not src_img.exists():
        print(f"[INFO] No {split} split in source — skipping.")
        return

    dst_img = dst / 'images' / split
    dst_lbl = dst / 'labels' / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_kept  = 0
    for img_path in sorted(src_img.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        n_total += 1
        src_txt = src_lbl / (img_path.stem + '.txt')
        dst_txt = dst_lbl / (img_path.stem + '.txt')

        has_large = filter_label_file(src_txt, dst_txt)
        if not has_large:
            # No fuse_cover/green_label in this image at all — drop it,
            # an empty-label image adds no signal for this model.
            dst_txt.unlink(missing_ok=True)
            continue

        shutil.copy2(img_path, dst_img / img_path.name)
        n_kept += 1

    print(f"  [{split}] {n_kept}/{n_total} images kept "
          f"(had at least one fuse_cover/green_label instance)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True,
                   help='Root of the (un-tiled) split dataset from 1_split_dataset.py')
    p.add_argument('--dst', required=True,
                   help='Output root for the large-object full-image dataset')
    a = p.parse_args()

    src = Path(a.src)
    dst = Path(a.dst)

    print(f"Building large-object dataset (classes {sorted(LARGE_CLASSES)}: "
          f"green_label, fuse_cover)\n")

    for split in ('train', 'val'):
        print(f"=== {split} ===")
        process_split(src, dst, split)

    print("\nDone. Point 4b_train_large.py at this dataset's dataset.yaml "
          "(copy/adjust dataset.yaml from the original split dataset — "
          "class names/order must stay identical, this script only removes "
          "instances, not class definitions).")


if __name__ == '__main__':
    main()
