# Models

This directory is a placeholder for trained model weights and configs.

## Files

| File | Produced by | Used by |
|---|---|---|
| `sticker_v1_best.pt` | `src/yolo_pipeline/4_train.py` (small-object model: qrcode, yellow_triangle) | `src/yolo_pipeline/5_predict.py --model` |
| `sticker_large_v1_best.pt` | `src/yolo_pipeline/4b_train_large.py` (large-object model: green_label, fuse_cover) | `src/yolo_pipeline/5_predict.py --model_large` |

Both are the `best.pt` checkpoint from their respective
`runs/detect/<name>/weights/` output directory after training (see
`src/yolo_pipeline/README.md`).

Weight files

`dataset.yaml` (class names/order — must stay identical between the
tiled and large-object datasets) is also expected to live alongside the
data in `data/`, not here — see `data/README.md`.


