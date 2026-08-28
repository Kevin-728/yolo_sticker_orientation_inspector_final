"""
1_split_dataset.py
------------------
Split a flat folder of YOLO-labeled images into train / val sets.

Usage
-----
    python 1_split_dataset.py --src /path/to/raw_dataset --dst /path/to/split_dataset

Arguments
---------
--src   folder containing images (.jpg/.png/.bmp) and matching .txt YOLO labels
--dst   output folder  (creates images/train, images/val, labels/train, labels/val)
--ratio train fraction  (default 0.8)
--seed  random seed     (default 42)

Expected input
--------------
raw_dataset/
    image001.jpg
    image001.txt
    image002.jpg
    image002.txt
    ...

Output
------
split_dataset/
    images/
        train/
        val/
    labels/
        train/
        val/
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def split_dataset(src: Path, dst: Path, ratio: float = 0.8, seed: int = 42):
    pairs = []
    for f in sorted(src.iterdir()):
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        txt = src / (f.stem + '.txt')
        if not txt.exists():
            print(f"[WARN] No label for {f.name} — skipping.")
            continue
        pairs.append((f, txt))

    if not pairs:
        raise ValueError(f"No valid image+label pairs found in: {src}")

    print(f"Found {len(pairs)} labeled images.")

    random.seed(seed)
    random.shuffle(pairs)

    n_train = int(len(pairs) * ratio)
    splits  = {'train': pairs[:n_train], 'val': pairs[n_train:]}

    for split, split_pairs in splits.items():
        img_dir = dst / 'images' / split
        lbl_dir = dst / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path, txt_path in split_pairs:
            shutil.copy2(img_path, img_dir / img_path.name)
            shutil.copy2(txt_path, lbl_dir / txt_path.name)
        print(f"  {split:5s}: {len(split_pairs)} images")

    print(f"\nDone. Output written to: {dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src',   required=True, help='Raw dataset folder')
    p.add_argument('--dst',   required=True, help='Output folder')
    p.add_argument('--ratio', type=float, default=0.8,
                   help='Train fraction (default 0.8)')
    p.add_argument('--seed',  type=int,   default=42,
                   help='Random seed (default 42)')
    a = p.parse_args()
    split_dataset(Path(a.src), Path(a.dst), a.ratio, a.seed)


if __name__ == '__main__':
    main()
