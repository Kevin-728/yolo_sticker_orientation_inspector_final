#!/usr/bin/env python3
# main.py
"""
Sticker Orientation Validation – command-line entry point.

Usage
-----
Option A – interleaved image/label pairs:
    python main.py img1.jpg img1.txt img2.jpg img2.txt ...

Option B – folder (auto-pairs .jpg/.png with matching .txt stem):
    python main.py --folder /path/to/folder
    e.g. python main.py --folder C:/Users/KUA4SZH/Desktop/inspector_test

Flags
-----
  --folder PATH   Scan a folder for image+label pairs.
  --json          Output raw JSON instead of the default ASCII table.

Output (default)
----------------
Plain-ASCII table, one table per image, one row per product:
  image: <path>
  id | up_vector | qr_1 | qr_2 | tri_1 | tri_2 | tri_3 | overall
  ----------------------------------------------------------------
  1  | ...       | ...  | ...  | ...   | ...   | ...   | ...
"""

import argparse
import json
import os
import sys

from yolo_converter        import load_yolo_labels
from product_grouper       import group_by_fuse_cover
from orientation_inspector import run_inspection
from table_formatter       import format_results


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ── Pair discovery ─────────────────────────────────────────────────────────────

def pairs_from_interleaved(args):
    """
    Parse Option A: img1.jpg img1.txt img2.jpg img2.txt ...
    Returns list of (img_path, txt_path) tuples.
    """
    if len(args) == 0:
        raise ValueError("No files provided.")
    if len(args) % 2 != 0:
        raise ValueError(
            "Option A requires an even number of arguments "
            "(image/label pairs): img1.jpg img1.txt img2.jpg img2.txt ..."
        )
    pairs = []
    for i in range(0, len(args), 2):
        img_path = args[i]
        txt_path = args[i + 1]
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Expected an image file at position {i}, got: {img_path!r}"
            )
        if not os.path.splitext(txt_path)[1].lower() == ".txt":
            raise ValueError(
                f"Expected a .txt label file at position {i+1}, got: {txt_path!r}"
            )
        pairs.append((img_path, txt_path))
    return pairs


def pairs_from_folder(folder):
    """
    Scan *folder* for image files and match each with a .txt of the same stem.
    Returns list of (img_path, txt_path) tuples sorted by stem.
    """
    pairs = []
    for fname in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in IMAGE_EXTENSIONS:
            continue
        txt_path = os.path.join(folder, stem + ".txt")
        if not os.path.isfile(txt_path):
            print(f"[WARNING] No label file for {fname!r} – skipping.",
                  file=sys.stderr)
            continue
        pairs.append((os.path.join(folder, fname), txt_path))
    if not pairs:
        raise ValueError(f"No valid image+label pairs found in folder: {folder!r}")
    return pairs


# ── Processing ─────────────────────────────────────────────────────────────────

def process_pair(img_path, txt_path):
    """Full pipeline for one image+label pair."""
    detections = load_yolo_labels(txt_path)
    products   = group_by_fuse_cover(detections)
    if not products:
        return {
            "image":    img_path,
            "products": [],
            "error":    "No fuse_cover detected – cannot inspect.",
        }
    results = run_inspection(img_path, products)
    return {"image": img_path, "products": results}


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="Sticker Orientation Validation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help="Interleaved image/label pairs: img1.jpg img1.txt img2.jpg img2.txt ...",
    )
    parser.add_argument(
        "--folder",
        metavar="PATH",
        help="Folder containing image+label pairs (auto-paired by filename stem).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of the default ASCII table.",
    )
    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Resolve pairs ──────────────────────────────────────────────────────────
    if args.folder and args.files:
        parser.error("Use either positional FILE pairs or --folder, not both.")

    if args.folder:
        try:
            pairs = pairs_from_folder(args.folder)
        except ValueError as e:
            parser.error(str(e))
    elif args.files:
        try:
            pairs = pairs_from_interleaved(args.files)
        except ValueError as e:
            parser.error(str(e))
    else:
        parser.print_help()
        sys.exit(0)

    # ── Process each pair ──────────────────────────────────────────────────────
    all_results = []
    for img_path, txt_path in pairs:
        print(f"[INFO] Processing: {img_path}", file=sys.stderr)
        try:
            result = process_pair(img_path, txt_path)
        except Exception as e:
            result = {"image": img_path, "error": str(e), "products": []}
        all_results.append(result)

    # ── Output ─────────────────────────────────────────────────────────────────
    if args.json:
        print(json.dumps(all_results, indent=2))
    else:
        print(format_results(all_results))


if __name__ == "__main__":
    main()
