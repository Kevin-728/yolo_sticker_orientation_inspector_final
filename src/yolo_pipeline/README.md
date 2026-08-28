# Object Detection Pipeline — README

This folder contains the full YOLO-based detection pipeline for the sticker
orientation validation project: raw images in, per-image YOLO-format `.txt`
label files out (which feed into the separate orientation-inspection part:
`yolo_converter.py` → `product_grouper.py` → `orientation_inspector.py`,
not included here).

Classes (fixed across the whole pipeline):

    0 = qrcode            (small,  ~10-11 px raw)
    1 = yellow_triangle    (small,  ~10-11 px raw)
    2 = green_label        (large,  50+ px raw)
    3 = fuse_cover          (large,  ~1800-2000 px raw)

## Why there are two models, not one

`qrcode` / `yellow_triangle` are so small in the raw 4080x3060 image that a
normal single full-image YOLO pass would shrink them below the reliable
detection floor (~20 px net input). They need **tiled + upsampled**
inference (crop a small native window, upscale it, run YOLO on the
upscaled crop) to stay detectable.

`fuse_cover` / `green_label` are the opposite problem: they are far BIGGER
than the tiling window. Feeding them through the same tiling scheme
fragments them into many partial boxes across tiles, and — worse — creates
contradictory training labels (the same visual surface is "fuse_cover" in
the one tile that contains its center, and unlabeled "background" in every
other tile that still shows a large chunk of it). This was the root cause
of the ~70-80x fuse_cover overdetection seen during development.

So the pipeline trains and runs **two separate models**:

  - **Small-object model**: tiled + upsampled pipeline, detects
    `qrcode` / `yellow_triangle` only.
  - **Large-object model**: plain full-image pipeline (no tiling), detects
    `fuse_cover` / `green_label` only.

`5_predict.py` runs both and merges their output into one final `.txt` per
image, in the same format the orientation-inspection part already expects.

## Files, in pipeline order

| Step | File | Status | Purpose |
|---|---|---|---|
| 1 | `1_split_dataset.py` | unchanged | Splits raw labeled images into train/val. |
| 2 | `2_validate_labels.py` | unchanged | Sanity-checks a YOLO dataset (any split — run again after tiling). |
| 3a | `3_tile_dataset.py` | **patched** | Builds the tiled+upsampled dataset for the small-object model. **Now excludes `fuse_cover`/`green_label`** (see "What changed" below). |
| 3b | `3b_prepare_large_dataset.py` | **new** | Builds the full-image dataset for the large-object model — copies images unchanged, filters labels down to only `fuse_cover`/`green_label`. |
| 4a | `4_train.py` | unchanged | Trains the small-object model (YOLOv11s, imgsz=640, on tiled data). |
| 4b | `4b_train_large.py` | **new** | Trains the large-object model (YOLOv11s, imgsz=1536, on full images). |
| 5 | `5_predict.py` | **patched** | Runs both models on a raw image and merges detections into one `.txt`. |
| 5b | `5b_predict_large.py` | **new** | Large-object full-image inference. Imported by `5_predict.py`; can also be run standalone for debugging. |

## What changed and why (summary)

1. **`3_tile_dataset.py` — large classes excluded from tiling.**
   `fuse_cover`/`green_label` no longer get tile labels at all. This
   removes the fragmented/contradictory training signal that was the
   dominant cause of fuse_cover overdetection.

2. **`3b_prepare_large_dataset.py` (new) + `4b_train_large.py` (new) —
   dedicated large-object training path.** Full, un-tiled images; only
   `fuse_cover`/`green_label` labels kept; trained at `imgsz=1536` since
   these objects don't have a small-object detection-floor problem and
   don't need upsampling.

3. **`5_predict.py` — responsibility-region filtering for the small-object
   pass.** With 50% tile overlap, a small object sitting in the shared
   band between two tiles is legitimately detected by both — that's
   expected. The old code relied on IoU-based NMS to merge these two
   detections back into one, but for ~10 px objects, a few pixels of
   inference jitter between the two independently-processed tiles is
   often enough to drop IoU below threshold, so NMS kept both as separate
   "real" detections (the ~2x qrcode/yellow_triangle overdetection).
   The fix: each tile now only "owns" a non-overlapping region of the
   image (computed via exact midpoint boundaries between neighboring
   tile positions — verified programmatically to partition the full
   image with zero gaps and zero overlap, including at the irregular
   edge tiles). A detection is kept only if its center falls inside the
   tile that produced it. Each physical object can now only be claimed
   once, regardless of small coordinate jitter.

4. **`5_predict.py` — merged two-model inference.** `predict_image()` now
   calls the small-object tiled pass and the large-object full-image pass
   (via `5b_predict_large.py`) and concatenates their results.

## How to run end-to-end

```bash
# 1. Split raw labeled images
python 1_split_dataset.py --src raw_dataset --dst split_dataset

# 2. Sanity-check the split
python 2_validate_labels.py --dataset split_dataset

# 3a. Build the tiled dataset for the small-object model
python 3_tile_dataset.py --src split_dataset --dst tiled_dataset \
    --crop_size 320 --net_size 640 --overlap 0.5

# 3b. Build the full-image dataset for the large-object model
python 3b_prepare_large_dataset.py --src split_dataset --dst large_object_dataset
# Copy/adjust dataset.yaml into large_object_dataset/ — class names/order
# must stay IDENTICAL to the original (this script only removes instances,
# not class definitions).

# 3-check. Confirm the tiling fix worked (no GPU needed)
python 2_validate_labels.py --dataset tiled_dataset
# fuse_cover/green_label counts should now be ZERO in the tiled dataset.

# 4a. Train the small-object model
python 4_train.py --data tiled_dataset/dataset.yaml --name sticker_v1

# pause training: ctrl+C
# resume training
# Note: this starts a new fine-tuning run from those weights, not a true resume=True continuation (optimizer state, epoch counter, and LR schedule restart). Given your memory notes on preferring "load last.pt as initial weights... when hyperparameters need to change," that's likely what you want anyway. If you instead want a byte-for-byte continuation (same epoch count, same LR schedule position), you'd need model.train(resume=True) pointed at the same run directory instead — but that only works if you haven't changed any training args since the original launch.
python 4_train.py --data tiled_dataset/dataset.yaml --name sticker_v1 \
       --model runs/detect/sticker_v1/weights/last.pt

python 4_train.py --resume runs/detect/sticker_v1/weights/last.pt

# 4b. Train the large-object model
python 4b_train_large.py --data large_object_dataset/dataset.yaml --name sticker_large_v1

# 5. Run combined inference on a raw image
python 5_predict.py \
    --model       runs/detect/sticker_v1/weights/best.pt \
    --model_large runs/detect/sticker_large_v1/weights/best.pt \
    --img         image.jpg
# Writes image.txt — same format the orientation-inspection part expects.
```

## Important: keep constants in sync

`CROP_SIZE`, `NET_SIZE`, and `TILE_OVERLAP` at the top of `5_predict.py`
must exactly match the `--crop_size` / `--net_size` / `--overlap` CLI args
used with `3_tile_dataset.py`. Likewise, `--large_imgsz` in `5_predict.py`
must match `--imgsz` used in `4b_train_large.py`. A mismatch here silently
misaligns coordinates or shrinks/grows objects relative to what the model
was trained on — no error is raised, so it's worth double-checking before
a full training run.

## Pre-flight checklist before a full (multi-hour) training run

- [ ] `dataset.yaml` `names:` order matches the class IDs above (`0 qrcode,
      1 yellow_triangle, 2 green_label, 3 fuse_cover`) in BOTH the tiled
      dataset and the large-object dataset.
- [ ] `2_validate_labels.py` on the tiled dataset shows zero instances of
      `green_label`/`fuse_cover`.
- [ ] `2_validate_labels.py` on the large-object dataset shows zero (or
      near-zero, if some slipped through) instances of
      `qrcode`/`yellow_triangle`.
- [ ] `3_tile_dataset.py` and `5_predict.py` constants agree (see above).
- [ ] `4b_train_large.py` `--imgsz` and `5_predict.py` `--large_imgsz`
      agree.

Since `save_period=10` is set in both training scripts, you don't need to
wait for a full run to sanity-check behavior — grab `weights/last.pt` after
10-20 epochs and run it through `5_predict.py` / `5b_predict_large.py` to
confirm detection counts and coordinate mapping look right before
committing GPU time to full convergence.
