"""
2_validate_labels.py
--------------------
Sanity-check a YOLO dataset (split or tiled) before training.

Usage
-----
    python 2_validate_labels.py --dataset /path/to/split_dataset

Checks
------
• Every image has a matching .txt label.
• All normalised coordinates are in (0, 1].
• Reports instance count PER CLASS, each with its own min/max bounding-box
  size in raw pixels (previously this was a single min/max blended across
  ALL classes -- meaningless for a dataset spanning ~10px triangles to
  ~2000px fuse_covers, since the printed range was really just "smallest
  class's min, biggest class's max").
  → Use the per-class ranges to decide imgsz for that class's model, and
     to sanity-check assumed size floors before configuring pipeline
     constants (CROP_SIZE/NET_SIZE/imgsz) around them.
     Rule of thumb: smallest object should be ≥ 20 px at training resolution.
"""

import argparse
from pathlib import Path

import cv2

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

CLASS_NAMES = {
    0: 'qrcode',
    1: 'yellow_triangle',
    2: 'green_label',
    3: 'fuse_cover',
}


def validate_split(img_dir: Path, lbl_dir: Path) -> dict:
    counts    = {k: 0 for k in CLASS_NAMES}
    # Per-class size tracking (was a single global min/max before -- that
    # blended e.g. fuse_cover's max with qrcode's min into one meaningless
    # pair). Each class gets its own [min_w, max_w, min_h, max_h].
    size_stats = {k: {'min_w': float('inf'), 'max_w': 0.0,
                      'min_h': float('inf'), 'max_h': 0.0}
                 for k in CLASS_NAMES}
    errors    = []
    n_images  = 0

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        n_images += 1

        txt_path = lbl_dir / (img_path.stem + '.txt')
        if not txt_path.exists():
            errors.append(f"Missing label: {img_path.name}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            errors.append(f"Cannot read image: {img_path.name}")
            continue
        ih, iw = img.shape[:2]

        for lineno, line in enumerate(txt_path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                errors.append(f"{txt_path.name}:{lineno} — too few fields")
                continue

            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])

            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and
                    0.0 <  w  <= 1.0 and 0.0 <  h  <= 1.0):
                errors.append(
                    f"{txt_path.name}:{lineno} — out of range  "
                    f"cls={cid} cx={cx:.4f} cy={cy:.4f} "
                    f"w={w:.4f} h={h:.4f}"
                )
                continue

            counts[cid] = counts.get(cid, 0) + 1
            st = size_stats.setdefault(
                cid, {'min_w': float('inf'), 'max_w': 0.0,
                      'min_h': float('inf'), 'max_h': 0.0})
            w_px, h_px = w * iw, h * ih
            st['min_w'] = min(st['min_w'], w_px);  st['max_w'] = max(st['max_w'], w_px)
            st['min_h'] = min(st['min_h'], h_px);  st['max_h'] = max(st['max_h'], h_px)

    return dict(
        n_images=n_images,
        counts=counts,
        size_stats=size_stats,
        errors=errors,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                   help='Dataset root (contains images/ and labels/)')
    a = p.parse_args()
    root = Path(a.dataset)

    for split in ('train', 'val'):
        img_dir = root / 'images' / split
        lbl_dir = root / 'labels' / split
        if not img_dir.exists():
            print(f"[INFO] No {split} split found — skipping.\n")
            continue

        r = validate_split(img_dir, lbl_dir)
        print(f"=== {split.upper()} ===")
        print(f"  Images          : {r['n_images']}")
        print(f"  Instances per class, with per-class raw-pixel size range:")
        for cid, name in CLASS_NAMES.items():
            n = r['counts'].get(cid, 0)
            st = r['size_stats'].get(cid, {})
            if n > 0 and st.get('min_w', float('inf')) < float('inf'):
                size_str = (f"w=[{st['min_w']:.1f}, {st['max_w']:.1f}]  "
                            f"h=[{st['min_h']:.1f}, {st['max_h']:.1f}]")
            else:
                size_str = "(no instances)"
            print(f"    {cid}  {name:<20s}: n={n:<6d} {size_str}")
        if r['errors']:
            print(f"  Errors ({len(r['errors'])}):")
            for e in r['errors'][:20]:
                print(f"    {e}")
            if len(r['errors']) > 20:
                print(f"    ... and {len(r['errors']) - 20} more.")
        else:
            print("  Errors          : none")
        print()


if __name__ == '__main__':
    main()
