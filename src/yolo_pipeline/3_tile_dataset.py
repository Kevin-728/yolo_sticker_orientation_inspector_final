"""
3_tile_dataset.py
-----------------
Tile a YOLO split dataset into upsampled sub-tiles for training (Option A).

*** UPDATED ***
Large classes (fuse_cover, green_label) are now EXCLUDED from tiling.
Reason: these objects (~1800-2000px raw for fuse_cover, 50px+ for green_label)
are far bigger than CROP_SIZE=320. Tiling them produced two compounding
problems:
  1. Fragmentation — the same object appears as many different partial
     boxes across many tiles, which NMS at inference cannot consolidate
     (low mutual IoU between fragments).
  2. Contradictory supervision — a tile only receives a label for an
     object if the object's CENTER falls inside that tile (see
     _remap_box's caller below). For an object much larger than one tile,
     many neighboring tiles still show large solid chunks of the same
     object with NO label, teaching the model that the same visual
     texture is both "fuse_cover" (in the center-owning tile) and
     "background" (everywhere else it's visible). This is a direct
     contradiction in the training signal, not just noisy labels.

Large classes are now trained/inferred via a separate full-image pipeline
(see 3b_prepare_large_dataset.py, 4b_train_large.py, 5b_predict_large.py).
This script continues to handle only the small, tiling-appropriate classes
(qrcode, yellow_triangle) exactly as before.

Why upsampling (unchanged)
---------------------------
Standard tiling crops a NET_SIZE x NET_SIZE raw-pixel region and feeds it to
the network unchanged — a sticker's pixel footprint in network-input space
equals its raw pixel size, regardless of tile size.  For this project, the
smallest stickers (yellow_triangle, qrcode) were originally assumed to be as
small as 10-11 px raw; per-class measurement via 2_validate_labels.py later
showed the real minimum is ~21px for qrcode and ~10-14px for yellow_triangle
(the two classes have independent, mostly-continuous size distributions, not
a single shared floor) — still below or near YOLO's reliable detection floor
(~20 px), so the upsampling strategy below remains necessary.

Option A fixes this by decoupling the *captured* native region from the
*network input* size:

    1. Crop a smaller native region:  CROP_SIZE x CROP_SIZE raw pixels
    2. Upscale that crop to:          NET_SIZE  x NET_SIZE  (fed to YOLO)

A CROP_SIZE=320 -> NET_SIZE=640 config is a 2x upscale, taking a 21 px
qrcode to ~42 px in network-input space — comfortably clear of the floor.

Label normalisation is scale-invariant (cx/cy/w/h are fractions of the tile
extent), so the remap-and-normalise logic is identical to a non-upsampled
pipeline; only the final image is resized before saving.

Usage
-----
    python 3_tile_dataset.py \
        --src  /path/to/split_dataset \
        --dst  /path/to/tiled_dataset \
        --crop_size 320 \
        --net_size  640 \
        --overlap 0.5 \
        --neg_ratio 0.3

IMPORTANT: crop_size, net_size, and overlap here must exactly match the
           constants at the top of 5_predict.py (CROP_SIZE, NET_SIZE,
           TILE_OVERLAP).

Tiles with no (small-class) labels are mostly discarded, but a SAMPLED
FRACTION (governed by --neg_ratio, default 0.3) is now kept as negative
(background-only) training examples — see NEGATIVE_TILE_FIX below. This
is a change from the original behavior, which discarded ALL empty tiles.

Bug fixes (post-training-run diagnosis)
-----------------------------------------
1. PARTIAL_BOX_FIX (_remap_box) — previously, when an object's center fell
   inside a tile but its edges extended past that tile's own crop
   boundary, the box was CLIPPED to the crop and the clipped fragment was
   written as the ground-truth label — teaching the model, some fraction
   of the time, that a small fragment of an object IS the complete
   object. This directly explained "detection only gets part of a
   qrcode" and plausibly degraded confidence more broadly through
   conflicting supervision (same texture labeled "complete" in one
   overlapping tile, "tiny fragment" in another). Fixed: a box is only
   labeled in a tile if FULLY contained within that tile's crop;
   otherwise it's dropped for that tile (a neighboring overlapping tile
   almost always captures it fully instead, at 50% overlap).

2. NEGATIVE_TILE_FIX, v2 (tile_single) — previously EVERY tile with zero
   small-class labels was discarded, so the small-object model never saw
   a single background-only example during training, leaving it with no
   calibrated notion of "not qrcode/triangle" — a strong contributor to
   scattered false-positive detections in empty/background regions at
   inference time. Fixed: keep a sampled fraction of empty tiles
   (--neg_ratio, default 0.3 relative to that image's positive-tile
   count). v1 prioritized tiles ADJACENT to a positive tile, but this
   turned out to mostly select MORE PRODUCT SURFACE (the housing is
   large and fills most of the frame, so "near a sticker" rarely means
   "off the product") rather than the true cardboard/floor background
   that was actually causing false positives -- confirmed by inspecting
   the generated dataset directly. v2 instead uses each image's large-
   class boxes (fuse_cover/green_label) as a PRODUCT-FOOTPRINT MASK and
   prioritizes true-background tiles (zero overlap with any product box)
   first, falling back to on-product empty tiles only if not enough true
   background exists. neg_ratio is NOT set arbitrarily high, since over-
   weighting negatives risks suppressing recall, which is the opposite
   problem this project has also been fighting.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

# ── Classes excluded from tiling (see module docstring) ────────────────────────
# 2 = green_label, 3 = fuse_cover  (per CLASS_MAP in yolo_converter.py)
LARGE_CLASSES = {2, 3}


# ── Core helpers ───────────────────────────────────────────────────────────────

def _tile_positions(iw: int, ih: int, crop_size: int, overlap: float):
    """Return (xs, ys) lists of top-left corners for the crop grid."""
    stride = max(1, int(crop_size * (1.0 - overlap)))
    xs = list(range(0, iw - crop_size + 1, stride))
    ys = list(range(0, ih - crop_size + 1, stride))
    # Ensure the right/bottom edge is always covered
    if not xs or xs[-1] + crop_size < iw:
        xs.append(max(0, iw - crop_size))
    if not ys or ys[-1] + crop_size < ih:
        ys.append(max(0, ih - crop_size))
    return sorted(set(xs)), sorted(set(ys))


def _remap_box(cid, cx_px, cy_px, w_px, h_px,
               tx1, ty1, tx2, ty2, crop_size):
    """
    Re-normalise a pixel-space box against this tile's crop region.

    *** BUG FIX (PARTIAL_BOX_FIX) ***
    Previously this function CLIPPED the box to the crop region and wrote
    the clipped (partial) box as the ground-truth label whenever an
    object's center fell inside a tile but its edges extended past that
    tile's own crop boundary. That taught the model, some fraction of the
    time, that a small fragment of an object (e.g. one corner of a QR
    code) IS the complete object -- directly explaining "detection only
    gets a part of a qrcode" behaviour, and plausibly degrading confidence
    more broadly through conflicting supervision (the same finder-pattern/
    module texture labeled as "a full box" in one overlapping tile and "a
    tiny partial box" in another).

    Fix: only accept a label for this tile if the box is FULLY contained
    within the tile's crop. If it isn't, return None -- the object is
    simply not labeled in THIS tile. With 50% tile overlap, an object
    small enough relative to CROP_SIZE is virtually always fully captured
    by at least one neighboring overlapping tile instead, so this doesn't
    lose training signal for normal-sized instances; it only removes bad
    (partial-fragment) labels. The rare instance too large to be fully
    captured by any tile at these settings is simply dropped from the
    tiled dataset -- an acceptable trade given how rare that is here, and
    far better than training on a corrupted label.
    """
    bx1 = cx_px - w_px / 2;  bx2 = cx_px + w_px / 2
    by1 = cy_px - h_px / 2;  by2 = cy_px + h_px / 2

    if bx1 < tx1 or by1 < ty1 or bx2 > tx2 or by2 > ty2:
        return None   # not fully contained -- drop for this tile

    new_cx = (cx_px - tx1) / crop_size
    new_cy = (cy_px - ty1) / crop_size
    new_w  = w_px / crop_size
    new_h  = h_px / crop_size

    # Clamp defensively (shouldn't trigger given the containment check
    # above, but guards against float edge cases at the exact boundary)
    new_cx = max(0.0, min(1.0, new_cx))
    new_cy = max(0.0, min(1.0, new_cy))
    new_w  = max(1e-4, min(1.0, new_w))
    new_h  = max(1e-4, min(1.0, new_h))

    return cid, new_cx, new_cy, new_w, new_h


# ── Per-image tiling ───────────────────────────────────────────────────────────

def tile_single(img_path: Path, txt_path: Path,
                out_img_dir: Path, out_lbl_dir: Path,
                crop_size: int, net_size: int, overlap: float,
                large_classes=LARGE_CLASSES,
                neg_ratio: float = 0.3,
                rng: random.Random = None) -> int:
    """
    *** BUG FIX (NEGATIVE_TILE_FIX, v2) ***
    Previously EVERY tile with zero small-class labels was discarded
    (`if not tile_labels: continue`), meaning the small-object model's
    entire training set consisted exclusively of crops containing a
    qrcode/yellow_triangle -- it never once saw a pure-background patch.
    For a detection model, background-only examples are the model's only
    source of information about what NOT to fire on; without them, novel
    background textures at inference time (cardboard, floor tiles, box
    edges, etc.) have nothing anchoring the network away from
    false-firing on them.

    Fix: keep a SAMPLED FRACTION of empty tiles as negatives (governed by
    `neg_ratio`, default 0.3 -- i.e. ~1 negative tile per ~3 positive
    tiles kept for this image), rather than discarding all of them or
    keeping all of them (all-empty-tiles-kept would drown out the already
    -sparse positive signal and risks HURTING recall, which is a real
    trade-off, not a free lever).

    v1 of this fix prioritized negatives from tiles ADJACENT to a
    positive tile, reasoning that nearby clutter is what's most likely to
    confuse the model. That assumption turned out to be wrong for this
    dataset: the product itself (the metal housing) is large and fills
    most of the frame, with stickers sitting somewhere in the middle of
    it -- so "adjacent to a sticker" mostly just means "a different patch
    of the same metal surface," not the cardboard/floor background that
    was actually producing false positives at inference. Confirmed
    directly: reviewing the generated dataset showed kept negative tiles
    were still overwhelmingly on-product.

    v2 fix: use each image's LARGE-class boxes (fuse_cover, green_label)
    -- already present in the label file, even though this model doesn't
    detect them -- purely as a PRODUCT-FOOTPRINT MASK. A tile is
    classified "on-product" if its crop overlaps any large-class box at
    all, otherwise "true background." Negative sampling now prioritizes
    true-background tiles FIRST (since that's the confirmed failure
    mode), falling back to on-product empty tiles only if there aren't
    enough background tiles to reach the target ratio. If an image has
    no large-class boxes at all (can't build a footprint mask), all empty
    tiles are treated as background candidates, same as before.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [WARN] Cannot read {img_path.name} — skipping.")
        return 0
    ih, iw = img.shape[:2]

    if rng is None:
        rng = random.Random()

    # Parse labels into pixel-space (cx, cy, w, h). Small-class boxes go
    # into `boxes` (these get tile labels). Large-class boxes go into
    # `product_boxes` purely as a PRODUCT-FOOTPRINT MASK for negative-tile
    # classification below -- this model never trains to detect them.
    boxes = []
    product_boxes = []   # (bx1, by1, bx2, by2) in raw pixel space
    n_skipped_large = 0
    if txt_path.exists():
        for line in txt_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            cx_px, cy_px, w_px, h_px = cx * iw, cy * ih, w * iw, h * ih
            if cid in large_classes:
                n_skipped_large += 1
                product_boxes.append((
                    cx_px - w_px / 2, cy_px - h_px / 2,
                    cx_px + w_px / 2, cy_px + h_px / 2,
                ))
                continue
            boxes.append((cid, cx_px, cy_px, w_px, h_px))

    xs, ys = _tile_positions(iw, ih, crop_size, overlap)
    stem   = img_path.stem

    # ── Pass 1: compute every tile's labels (fully-contained boxes only) ───
    tile_grid       = {}   # (row_i, col_i) -> (tx, ty, tx2, ty2, tile_labels)
    positive_cells  = []
    empty_cells     = []
    n_dropped_partial = 0

    for row_i, ty in enumerate(ys):
        for col_i, tx in enumerate(xs):
            tx2 = min(tx + crop_size, iw)
            ty2 = min(ty + crop_size, ih)

            # ── Collect boxes whose centre falls inside this crop ───────────
            tile_labels = []
            for cid, cx_px, cy_px, w_px, h_px in boxes:
                if not (tx <= cx_px < tx2 and ty <= cy_px < ty2):
                    continue
                entry = _remap_box(cid, cx_px, cy_px, w_px, h_px,
                                   tx, ty, tx2, ty2, crop_size)
                if entry is not None:
                    tile_labels.append(entry)
                else:
                    n_dropped_partial += 1

            tile_grid[(row_i, col_i)] = (tx, ty, tx2, ty2, tile_labels)
            (positive_cells if tile_labels else empty_cells).append((row_i, col_i))

    # ── Pass 2: choose which empty tiles to keep as negatives ──────────────
    def _overlaps_product(tx, ty, tx2, ty2):
        """True if this tile's crop overlaps ANY large-class (product)
        box at all -- i.e. this tile shows at least some of the product's
        own housing, not pure surrounding background."""
        for bx1, by1, bx2, by2 in product_boxes:
            if bx1 < tx2 and bx2 > tx and by1 < ty2 and by2 > ty:
                return True
        return False

    kept_negatives = set()
    n_bg_chosen = n_onprod_chosen = 0
    if positive_cells and empty_cells and neg_ratio > 0:
        target_neg = round(neg_ratio * len(positive_cells))
        if target_neg > 0:
            if product_boxes:
                background_empty = []
                onproduct_empty  = []
                for cell in empty_cells:
                    tx, ty, tx2, ty2, _ = tile_grid[cell]
                    if _overlaps_product(tx, ty, tx2, ty2):
                        onproduct_empty.append(cell)
                    else:
                        background_empty.append(cell)
            else:
                # No large-class boxes in this image at all -- can't build
                # a footprint mask, so treat every empty tile as a
                # background candidate (same as pre-v2 behavior).
                background_empty = list(empty_cells)
                onproduct_empty  = []

            rng.shuffle(background_empty)
            rng.shuffle(onproduct_empty)

            # True background (confirmed source of false positives) is
            # prioritized; on-product empty tiles are only used to fill
            # the remainder if there aren't enough background tiles.
            chosen = background_empty[:target_neg]
            n_bg_chosen = len(chosen)
            if len(chosen) < target_neg:
                fill = onproduct_empty[:target_neg - len(chosen)]
                n_onprod_chosen = len(fill)
                chosen += fill

            kept_negatives = set(chosen)

    # ── Pass 3: write positive tiles + selected negative tiles ─────────────
    n_out = 0
    for (row_i, col_i), (tx, ty, tx2, ty2, tile_labels) in tile_grid.items():
        if not tile_labels and (row_i, col_i) not in kept_negatives:
            continue  # discarded empty tile

        # ── Crop native region, zero-pad if edge ────────────────────────
        patch = img[ty:ty2, tx:tx2]
        if patch.shape[:2] != (crop_size, crop_size):
            canvas = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            canvas[:ty2 - ty, :tx2 - tx] = patch
            patch = canvas

        # ── Upscale crop to network input size ──────────────────────────
        if net_size != crop_size:
            patch = cv2.resize(patch, (net_size, net_size),
                               interpolation=cv2.INTER_CUBIC)

        # ── Write tile image and label ───────────────────────────────────
        # Labels stay normalised against crop_size — resize does not
        # change relative (normalised) box position, so no remap needed.
        # Negative tiles get an empty label file (valid YOLO convention
        # for "no objects present").
        name = f"{stem}_r{row_i:02d}_c{col_i:02d}"
        cv2.imwrite(str(out_img_dir / (name + img_path.suffix)), patch)
        with open(out_lbl_dir / (name + '.txt'), 'w') as f:
            for cid, cx, cy, w, h in tile_labels:
                f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        n_out += 1

    if n_skipped_large:
        print(f"    ({img_path.name}: skipped {n_skipped_large} large-class "
              f"instance(s) — handled by the full-image pipeline instead)")
    if n_dropped_partial:
        print(f"    ({img_path.name}: dropped {n_dropped_partial} tile-"
              f"boundary-clipped label(s) — object fully captured by a "
              f"different overlapping tile instead)")
    if kept_negatives:
        print(f"    ({img_path.name}: kept {len(kept_negatives)} negative "
              f"tile(s) out of {len(empty_cells)} empty candidates -- "
              f"{n_bg_chosen} true-background, {n_onprod_chosen} on-product)")

    return n_out


# ── Per-split tiling ───────────────────────────────────────────────────────────

def tile_split(src: Path, dst: Path, split: str,
               crop_size: int, net_size: int, overlap: float,
               neg_ratio: float = 0.3, seed: int = 42):
    src_img = src / 'images' / split
    src_lbl = src / 'labels' / split
    if not src_img.exists():
        print(f"[INFO] No {split} split in source — skipping.")
        return

    dst_img = dst / 'images' / split
    dst_lbl = dst / 'labels' / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    images      = sorted(f for f in src_img.iterdir()
                         if f.suffix.lower() in IMAGE_EXTS)
    total_tiles = 0

    # Single shared RNG across all images in this split, seeded once, so
    # negative-tile sampling is reproducible run-to-run for a given seed.
    rng = random.Random(seed)

    for img_path in images:
        txt_path = src_lbl / (img_path.stem + '.txt')
        n = tile_single(img_path, txt_path,
                        dst_img, dst_lbl, crop_size, net_size, overlap,
                        neg_ratio=neg_ratio, rng=rng)
        print(f"  {img_path.name}: {n} tiles written")
        total_tiles += n

    print(f"  [{split}] total tiles written: {total_tiles}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src',        required=True,
                   help='Root of split dataset (has images/ and labels/ subdirs)')
    p.add_argument('--dst',        required=True,
                   help='Output root for tiled dataset')
    p.add_argument('--crop_size',  type=int,   default=320,
                   help='Native raw-pixel crop size before upscaling (default 320). '
                        'Must match CROP_SIZE in 5_predict.py.')
    p.add_argument('--net_size',   type=int,   default=640,
                   help='Final tile size fed to the network after upscaling '
                        '(default 640). Must match NET_SIZE in 5_predict.py '
                        'and --imgsz in 4_train.py.')
    p.add_argument('--overlap',    type=float, default=0.5,
                   help='Fractional overlap between adjacent crops (default 0.5). '
                        'Must match TILE_OVERLAP in 5_predict.py.')
    p.add_argument('--neg_ratio',  type=float, default=0.3,
                   help='Fraction of empty (background-only) tiles to KEEP '
                        'per image, relative to that image\'s positive-tile '
                        'count (default 0.3, i.e. ~1 negative per ~3 '
                        'positives). Previously ALL empty tiles were '
                        'discarded, so the model never saw background at '
                        'all -- see NEGATIVE_TILE_FIX in tile_single\'s '
                        'docstring. Set to 0 to restore the old '
                        'all-discarded behavior.')
    p.add_argument('--seed',       type=int,   default=42,
                   help='Random seed for negative-tile sampling '
                        '(default 42, for reproducibility)')
    a = p.parse_args()

    src = Path(a.src)
    dst = Path(a.dst)

    upscale = a.net_size / a.crop_size
    print(f"Crop size (native) : {a.crop_size} px")
    print(f"Net size (network) : {a.net_size} px")
    print(f"Upscale factor      : {upscale:.2f}x")
    print(f"Overlap             : {a.overlap:.0%}")
    print(f"Stride              : {int(a.crop_size * (1 - a.overlap))} px")
    print(f"Negative tile ratio : {a.neg_ratio:.0%} of each image's positive "
          f"tile count (0 = old behavior, all empty tiles discarded)")
    print(f"Excluded classes    : {sorted(LARGE_CLASSES)} "
          f"(green_label, fuse_cover — see module docstring)\n")

    for split in ('train', 'val'):
        print(f"=== Tiling {split} ===")
        tile_split(src, dst, split, a.crop_size, a.net_size, a.overlap,
                  neg_ratio=a.neg_ratio, seed=a.seed)

    print("\nDone.")
    print("NOTE: fuse_cover / green_label were excluded from this tiled dataset.")
    print("      Run 3b_prepare_large_dataset.py separately to build their")
    print("      full-image training set.")


if __name__ == '__main__':
    main()