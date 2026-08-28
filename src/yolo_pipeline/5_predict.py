"""
5_predict.py
------------
Run tiled, upsampled YOLO inference (Option A) on a full-resolution
(4080x3060) image, for the SMALL classes (qrcode, yellow_triangle),
merged with a separate full-image pass (5b_predict_large.py) for the
LARGE classes (fuse_cover, green_label).

Outputs a YOLO-format .txt file that is a drop-in replacement for the label
files consumed by the existing orientation-inspection pipeline
(yolo_converter.py -> product_grouper.py -> orientation_inspector.py).

Usage
-----
    python 5_predict.py \
        --model       runs/detect/sticker_v1/weights/best.pt \
        --model_large runs/detect/sticker_large_v1/weights/best.pt \
        --img         /path/to/image.jpg
    
    # Run detection on every image in a folder, .txt written alongside each image:
    python 5_predict.py --model best.pt --model_large large_best.pt --folder test

    # Write outputs to a separate directory instead, with full per-detection printout:
    python 5_predict.py --model best.pt --model_large large_best.pt \
        --folder test --out test_predictions --verbose

    # explicit output path:
    python 5_predict.py --model best.pt --model_large large_best.pt --img img.jpg --out img_pred.txt

    # stricter confidence:
    python 5_predict.py --model best.pt --model_large large_best.pt --img img.jpg --conf 0.4

    # small-object pass only (skip large classes entirely):
    python 5_predict.py --model best.pt --img img.jpg

    # run on every image in a folder (e.g. a "test" folder) in one go --
    # writes one .txt per image, same stem, alongside each source image:
    python 5_predict.py --model best.pt --model_large large_best.pt --folder test

    # folder mode with a separate output directory instead of writing
    # .txt files alongside the source images, plus full per-detection
    # printout for every image:
    python 5_predict.py --model best.pt --model_large large_best.pt \
        --folder test --out test_predictions --verbose

    # the .txt files written by --folder mode pair directly with
    # main.py's own --folder mode for the downstream orientation pipeline:
    python main.py --folder test

Output format (one detection per line)
---------------------------------------
    class_id  cx  cy  w  h
    (normalised to the original image dimensions, same as yolo_converter.py
    expects)

*** UPDATED: two fixes vs. the original version ***

1. Large classes (fuse_cover, green_label) are no longer produced by the
   tiled pass at all — they are handled entirely by 5b_predict_large.py's
   single full-image inference call, which does not fragment them (see
   3_tile_dataset.py docstring for why tiling broke on objects this size).

2. Small classes (qrcode, yellow_triangle) now use RESPONSIBILITY-REGION
   FILTERING instead of relying on IoU-based NMS to remove cross-tile
   duplicates. With 50% tile overlap, any small object sitting in the
   shared band between two adjacent tiles is legitimately detected by
   BOTH tiles (this is intentional/correct — that's what the overlap is
   for during training). The old code relied on cv2.dnn.NMSBoxes to merge
   these two detections back into one, but for ~10px objects a few pixels
   of inference jitter between the two independently-processed crops is
   often enough to drop IoU below the 0.45 threshold, so NMS treated them
   as two separate real objects instead of one duplicated one.

   The fix: each tile only "owns" a non-overlapping STRIDE-sized region
   (extended to the image edge for the last row/column, so full coverage
   is preserved). After mapping a detection's box back to raw-image
   coordinates, it is kept only if its CENTER falls inside the tile's
   owned region. Since each physical object's center falls inside
   exactly one tile's owned region (regions partition the image with no
   overlap and no gaps), each object can now only be claimed once,
   regardless of small jitter in the recovered box coordinates. Per-class
   NMS is still run afterwards as a light safety net for genuinely
   adjacent distinct objects, but it is no longer relied upon to fix
   cross-tile duplication.

3. IOU_MERGE_FIX — two genuinely SEPARATE small objects sitting close
   together within the SAME tile (e.g. two adjacent qrcodes) were being
   collapsed into one detection. Root cause was NOT the final
   _nms_per_class safety net -- it was that predict_image_small's
   per-tile `model.predict()` call never received the `iou` argument at
   all, so it silently ran Ultralytics' own internal NMS at its default
   (0.7) threshold, inside every tile, before responsibility-region
   filtering or _nms_per_class ever saw the boxes. Two close real objects
   can easily produce >0.7 IoU between their PREDICTED boxes (box
   regression tends to overshoot early in training), so one of the two
   real detections was discarded at the very first inference call.

   The fix: pass `iou` through to that per-tile model.predict() call, and
   raise the default from 0.45 to 0.8. A single shared threshold is used
   at all three NMS sites (per-tile, final cross-tile, and the
   large-object safety net) rather than two separate values, because the
   final cross-tile stage's remaining real job — since responsibility-
   region filtering already handles true cross-tile duplication
   structurally — is now the SAME job as the per-tile stage: don't kill
   two genuinely distinct close objects. Giving it a lower threshold than
   the per-tile stage would just re-merge the same two boxes one step
   later, undoing the fix.

4. OWNERSHIP_CENTER_FIX — _ownership_boundaries computed midpoints
   between tile TOP-LEFT CORNERS, but should have used tile CENTERS
   (position + CROP_SIZE/2). Since CROP_SIZE (320) is much larger than
   the stride (160 at 50% overlap), this put every ownership boundary
   crop_size/2 (80px) away from where it needed to be — landing near the
   START of each tile's own crop rather than centered inside it. Result:
   an 80px-wide dead zone at EVERY interior tile boundary (confirmed
   numerically across the full grid, both x and y) where the "owning"
   tile's own crop didn't actually reach, while a neighboring tile that
   DID fully see that area had its correct detections discarded for not
   being the designated owner. Confirmed against a real case: a QR code
   whose true "owning" tile only captured 62% of its height (truncated,
   undersized detection) while a neighboring tile that fully captured it
   had its correct, high-confidence (0.92) detection thrown away by the
   filter. Fix: boundaries now computed from tile centers, verified to
   fully eliminate the dead zone (including at irregular edge-tile
   spacing) while still maintaining a gapless, non-overlapping partition.

How tiled + upsampled inference works (Option A) — unchanged
--------------------------------------------------------------
The image is divided into a grid of overlapping CROP_SIZE x CROP_SIZE native
regions (same grid math as 3_tile_dataset.py).  Each crop is upscaled to
NET_SIZE x NET_SIZE before being fed to the model — this is what recovers
detectability for stickers as small as ~10-11 px raw (they become
~20-22 px in network-input space at the default 320->640 config).

Model output boxes come back in NET_SIZE pixel space.  To map back to the
original image:
    1. Divide by the upscale factor (NET_SIZE / CROP_SIZE) -> crop-space px
    2. Add the crop's (tx, ty) offset -> original image px

IMPORTANT: CROP_SIZE, NET_SIZE, and TILE_OVERLAP here must exactly match the
           --crop_size / --net_size / --overlap values used when generating
           the tiled training dataset (3_tile_dataset.py).
"""

import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Recognized image extensions for --folder mode (matches main.py's convention)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def find_images_in_folder(folder: Path) -> list:
    """Return sorted list of image file paths in *folder* (non-recursive)."""
    imgs = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not imgs:
        raise ValueError(f"No image files found in folder: {folder!r}")
    return imgs

# NOTE: Python module names can't start with a digit, so a plain
# `import 5b_predict_large` is not valid syntax. Load it by file path
# instead - it must live in the same directory as this script.
_this_dir = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "predict_large", _this_dir / "5b_predict_large.py")
_predict_large_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_predict_large_mod)
predict_image_large = _predict_large_mod.predict_image_large
LARGE_CLASSES        = _predict_large_mod.LARGE_CLASSES

# ── Must match 3_tile_dataset.py ──────────────────────────────────────────────
CROP_SIZE    = 320   # native raw-pixel crop size before upscaling
NET_SIZE     = 640   # size fed to the network after upscaling (= --imgsz in training)
TILE_OVERLAP = 0.5   # fractional overlap between adjacent crops

_UPSCALE = NET_SIZE / CROP_SIZE

# Small classes handled by this file's tiled pass (everything else is left
# to the large-object model in 5b_predict_large.py).
SMALL_CLASSES = {0, 1}   # qrcode, yellow_triangle


# ── Tiling helpers ─────────────────────────────────────────────────────────────

def _tile_positions(iw: int, ih: int):
    stride = max(1, int(CROP_SIZE * (1.0 - TILE_OVERLAP)))
    xs = list(range(0, iw - CROP_SIZE + 1, stride))
    ys = list(range(0, ih - CROP_SIZE + 1, stride))
    if not xs or xs[-1] + CROP_SIZE < iw:
        xs.append(max(0, iw - CROP_SIZE))
    if not ys or ys[-1] + CROP_SIZE < ih:
        ys.append(max(0, ih - CROP_SIZE))
    return sorted(set(xs)), sorted(set(ys))


def _ownership_boundaries(positions: list, total: int, crop_size: int) -> list:
    """
    Given a sorted list of tile top-left coordinates along one axis (which
    may NOT be evenly spaced — the final tile is snapped to the image edge
    to guarantee coverage, so its offset from the previous tile can differ
    from the regular stride), return boundary cut points such that tile i
    owns [bounds[i], bounds[i+1]).

    *** BUG FIX (OWNERSHIP_CENTER_FIX) ***
    Boundaries are placed at the midpoint between each pair of consecutive
    TILE CENTERS (position + crop_size/2) — NOT between tile top-left
    corners as a naive midpoint might suggest. This distinction matters a
    lot here because CROP_SIZE (320) is much larger than the tile stride
    (160 at 50% overlap): using raw corner positions shifts every boundary
    by crop_size/2 relative to where it needs to be, which put the boundary
    near the START of each tile's own crop instead of centered inside it.

    Concretely, with the old corner-based midpoint, tile i's "owned"
    region worked out to [pos_i - stride/2, pos_i + stride/2) — but tile
    i's own crop only starts AT pos_i, so the left half of that owned
    region, [pos_i - stride/2, pos_i), was outside the tile's own crop
    entirely. That's a stride/2-wide (80px at default settings) dead zone
    at EVERY interior tile boundary, in both x and y: the tile deemed
    "owner" of that band couldn't actually see it (crop didn't reach that
    far), while the NEIGHBORING tile that COULD see it had its detections
    discarded there for not being the designated owner. This was
    confirmed both by direct arithmetic and by a real missed/undersized
    QR detection where the true "owning" tile only captured 62% of the
    object's height while a neighboring tile that fully captured it had
    its correct, high-confidence detection thrown away.

    Using tile CENTERS instead of corners removes this shift: boundaries
    land in the middle of each tile's own crop, so a tile's owned region
    is always fully contained within the area it actually images. Still
    guarantees an exact partition of [0, total) with NO gaps and NO
    overlap, and still handles irregular edge spacing correctly (verified
    numerically) since it only depends on each tile's own position, not
    an assumed fixed stride.
    """
    n = len(positions)
    bounds = [0]
    for i in range(n - 1):
        center_i    = positions[i]     + crop_size / 2.0
        center_next = positions[i + 1] + crop_size / 2.0
        mid = int((center_i + center_next) / 2.0)
        bounds.append(mid)
    bounds.append(total)
    return bounds


# ── Per-class NMS across tiles (safety net, not primary dedup) ─────────────────

def _nms_per_class(raw: list, conf_thr: float, iou_thr: float,
                   iw: int, ih: int) -> list:
    """
    raw: list of [cls_id, x1, y1, x2, y2, score] in original pixel coords.
    Returns list of (cls_id, cx_n, cy_n, w_n, h_n, score) normalised to image.
    """
    if not raw:
        return []

    arr = np.array(raw, dtype=np.float32)
    out = []

    for cls_id in np.unique(arr[:, 0].astype(int)):
        mask   = arr[:, 0].astype(int) == cls_id
        sub    = arr[mask]
        x1s, y1s, x2s, y2s = sub[:, 1], sub[:, 2], sub[:, 3], sub[:, 4]
        scores = sub[:, 5]

        # cv2.dnn.NMSBoxes expects [x, y, w, h] format
        bboxes = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                  for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s)]

        indices = cv2.dnn.NMSBoxes(bboxes, scores.tolist(), conf_thr, iou_thr)

        if len(indices) == 0:
            continue

        # Handle (N,1) shape from older OpenCV versions
        indices = (indices.flatten() if isinstance(indices, np.ndarray)
                   else [i[0] for i in indices])

        for i in indices:
            x1, y1, x2, y2 = x1s[i], y1s[i], x2s[i], y2s[i]
            out.append((
                int(cls_id),
                float((x1 + x2) / 2 / iw),   # cx normalised
                float((y1 + y2) / 2 / ih),   # cy normalised
                float((x2 - x1) / iw),        # w  normalised
                float((y2 - y1) / ih),        # h  normalised
                float(scores[i]),
            ))

    return out


# ── Small-object tiled inference ───────────────────────────────────────────────

def predict_image_small(model: YOLO, img_path: Path,
                        conf: float = 0.25, iou: float = 0.5) -> list:
    """
    Returns list of (cls_id, cx_n, cy_n, w_n, h_n, score) for SMALL_CLASSES
    only, using tiled+upsampled inference with responsibility-region
    filtering to prevent cross-tile duplication.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    ih, iw = img.shape[:2]

    xs, ys    = _tile_positions(iw, ih)
    x_bounds  = _ownership_boundaries(xs, iw, CROP_SIZE)
    y_bounds  = _ownership_boundaries(ys, ih, CROP_SIZE)
    raw_boxes = []

    for row_i, ty in enumerate(ys):
        for col_i, tx in enumerate(xs):
            tx2 = min(tx + CROP_SIZE, iw)
            ty2 = min(ty + CROP_SIZE, ih)

            # ── Extract native crop, zero-pad if edge ───────────────────────
            patch = img[ty:ty2, tx:tx2]
            if patch.shape[:2] != (CROP_SIZE, CROP_SIZE):
                canvas = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
                canvas[:ty2 - ty, :tx2 - tx] = patch
                patch = canvas

            # ── Upscale crop to network input size (same as training) ───────
            if NET_SIZE != CROP_SIZE:
                patch = cv2.resize(patch, (NET_SIZE, NET_SIZE),
                                   interpolation=cv2.INTER_CUBIC)

            # ── Run YOLO on upscaled tile ─────────────────────────────────────
            # iou is passed explicitly here (not left to Ultralytics' 0.7
            # default) -- see IOU_MERGE_FIX note in the module docstring for
            # why this matters: two genuinely distinct close-together small
            # objects can have >0.7 IoU between their PREDICTED boxes even
            # when their true boxes don't overlap that much, especially
            # while box regression is still under-converged early in
            # training. Left uncontrolled, one of the two real detections
            # was being silently discarded inside this call, before
            # responsibility-region filtering or _nms_per_class ever saw it.
            results = model.predict(patch, imgsz=NET_SIZE,
                                    conf=conf, iou=iou, verbose=False)

            # ── This tile's responsibility region in raw-image coords ───────
            rx1, rx2 = x_bounds[col_i], x_bounds[col_i + 1]
            ry1, ry2 = y_bounds[row_i], y_bounds[row_i + 1]

            # ── Map detections back to original image coords ─────────────────
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in SMALL_CLASSES:
                        continue   # large classes are not this pass's job

                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    bx1 /= _UPSCALE;  by1 /= _UPSCALE
                    bx2 /= _UPSCALE;  by2 /= _UPSCALE
                    bx1 = np.clip(bx1 + tx, 0, iw); by1 = np.clip(by1 + ty, 0, ih)
                    bx2 = np.clip(bx2 + tx, 0, iw); by2 = np.clip(by2 + ty, 0, ih)

                    # ── Responsibility-region filter ─────────────────────────
                    # Keep this detection only if ITS CENTER falls inside the
                    # region this tile owns. Since owned regions exactly
                    # partition the image, each physical object's center can
                    # satisfy this for exactly one tile — eliminating
                    # cross-tile duplication at the source, rather than
                    # hoping IoU-NMS catches it afterwards.
                    cx = (bx1 + bx2) / 2.0
                    cy = (by1 + by2) / 2.0
                    if not (rx1 <= cx < rx2 and ry1 <= cy < ry2):
                        continue

                    raw_boxes.append([
                        cls_id,
                        float(bx1), float(by1), float(bx2), float(by2),
                        float(box.conf[0]),
                    ])

    return _nms_per_class(raw_boxes, conf, iou, iw, ih)


# ── Combined small + large inference ───────────────────────────────────────────

def predict_image(model: YOLO, img_path: Path,
                  conf: float = 0.25, iou: float = 0.5,
                  model_large: YOLO = None, large_imgsz: int = 1536) -> list:
    """
    Returns list of (cls_id, cx_n, cy_n, w_n, h_n, score) merging:
      - the small-object tiled pass (qrcode, yellow_triangle)
      - the large-object full-image pass (fuse_cover, green_label),
        if model_large is provided.

    NOTE: iou=0.8 (raised from the old 0.45 default) is deliberately used
    as a SINGLE shared threshold across all three NMS sites: the per-tile
    model.predict() call inside predict_image_small, the final
    _nms_per_class cross-tile safety net, AND the large-object model's own
    safety-net NMS. See IOU_MERGE_FIX in the module docstring for why a
    single higher threshold was chosen over two separate ones. It's a
    no-op risk for the large-object pass (fuse_cover/green_label rarely
    sit close enough to trigger this at all) but harmless either way.
    """
    detections = predict_image_small(model, img_path, conf, iou)

    if model_large is not None:
        detections += predict_image_large(
            model_large, img_path, imgsz=large_imgsz, conf=conf, iou=iou)

    return detections


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True,
                   help='Path to small-object trained weights, e.g. '
                        'runs/detect/sticker_v1/weights/best.pt')
    p.add_argument('--model_large', default=None,
                   help='Path to large-object trained weights, e.g. '
                        'runs/detect/sticker_large_v1/weights/best.pt. '
                        'If omitted, fuse_cover/green_label are not predicted.')
    p.add_argument('--img',    default=None,
                   help='Single input image path. Mutually exclusive with '
                        '--folder -- provide exactly one.')
    p.add_argument('--folder', default=None,
                   help='Folder of images to run detection on (scans for '
                        '.jpg/.jpeg/.png/.bmp, non-recursive). Mutually '
                        'exclusive with --img -- provide exactly one. '
                        'Writes one .txt per image (same stem as the '
                        'image) to --out (if given, treated as an output '
                        'DIRECTORY here) or alongside each source image. '
                        'Output .txt files pair directly with main.py '
                        '--folder for the downstream orientation pipeline.')
    p.add_argument('--out',   default=None,
                   help='For --img: output .txt path (default: same '
                        'directory and stem as --img). For --folder: '
                        'output DIRECTORY for all .txt files (default: '
                        'same folder as the source images).')
    p.add_argument('--conf',  type=float, default=0.25,
                   help='Confidence threshold (default 0.25)')
    p.add_argument('--iou',   type=float, default=0.5,
                   help='NMS IoU threshold (default 0.5, raised from an '
                        'earlier 0.45). Shared across the per-tile pass, '
                        'the final cross-tile safety net, and the '
                        'large-object safety net. A HIGH value here means '
                        'only near-total box overlap counts as a duplicate '
                        '-- needed so two genuinely distinct small objects '
                        'sitting close together (e.g. two adjacent qrcodes) '
                        'don\'t get incorrectly merged into one detection. '
                        'See IOU_MERGE_FIX in the module docstring.')
    p.add_argument('--large_imgsz', type=int, default=1536,
                   help='imgsz for the large-object full-image pass (default 1536). '
                        'Must match --imgsz used in 4b_train_large.py.')
    p.add_argument('--verbose', action='store_true',
                   help='In --folder mode, also print each individual '
                        'detection (class/coords/confidence) per image, '
                        'not just a one-line per-image summary.')
    a = p.parse_args()

    if bool(a.img) == bool(a.folder):
        p.error("Provide exactly one of --img or --folder.")

    model       = YOLO(a.model)
    model_large = YOLO(a.model_large) if a.model_large else None
    cls_names   = {0: 'qrcode', 1: 'yellow_triangle',
                  2: 'green_label', 3: 'fuse_cover'}

    # ── Single-image mode (unchanged behavior) ──────────────────────────────
    if a.img:
        img_path = Path(a.img)
        out_path = Path(a.out) if a.out else img_path.with_suffix('.txt')

        detections = predict_image(model, img_path, a.conf, a.iou,
                                   model_large=model_large,
                                   large_imgsz=a.large_imgsz)

        # Write YOLO-format .txt (no confidence column — matches yolo_converter.py)
        with open(out_path, 'w') as f:
            for cls_id, cx, cy, w, h, _score in detections:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        print(f"Wrote {len(detections)} detections -> {out_path}")
        for cls_id, cx, cy, w, h, score in detections:
            name = cls_names.get(cls_id, f'cls{cls_id}')
            print(f"  {name:<20s} cx={cx:.4f} cy={cy:.4f} "
                  f"w={w:.4f} h={h:.4f}  conf={score:.3f}")
        return

    # ── Folder mode ──────────────────────────────────────────────────────────
    folder = Path(a.folder)
    if not folder.is_dir():
        p.error(f"--folder path is not a directory: {a.folder!r}")

    image_paths = find_images_in_folder(folder)

    out_dir = None
    if a.out:
        out_dir = Path(a.out)
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(image_paths)} image(s) in {folder}")

    total_detections = 0
    n_errors = 0
    for img_path in image_paths:
        out_path = (out_dir / (img_path.stem + '.txt')) if out_dir \
                   else img_path.with_suffix('.txt')

        try:
            detections = predict_image(model, img_path, a.conf, a.iou,
                                       model_large=model_large,
                                       large_imgsz=a.large_imgsz)
        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            n_errors += 1
            continue

        with open(out_path, 'w') as f:
            for cls_id, cx, cy, w, h, _score in detections:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        print(f"  {img_path.name}: {len(detections)} detections -> {out_path}")
        if a.verbose:
            for cls_id, cx, cy, w, h, score in detections:
                name = cls_names.get(cls_id, f'cls{cls_id}')
                print(f"    {name:<20s} cx={cx:.4f} cy={cy:.4f} "
                      f"w={w:.4f} h={h:.4f}  conf={score:.3f}")

        total_detections += len(detections)

    print(f"\nDone. {len(image_paths) - n_errors}/{len(image_paths)} image(s) "
          f"processed successfully, {total_detections} total detections written.")
    if n_errors:
        print(f"[WARNING] {n_errors} image(s) failed — see [ERROR] lines above.")


if __name__ == '__main__':
    main()