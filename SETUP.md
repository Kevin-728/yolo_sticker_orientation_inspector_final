# Setup & Installation

This project runs entirely locally — no external API keys or third-party
services are required.

## Prerequisites

- **Anaconda / Miniconda** (Python environment manager)
- **Windows users:** run all commands from the **Anaconda Prompt**, not
  Git Bash — some package installs behave inconsistently under Git Bash on
  this project.
- **GPU (recommended, not required):** the pipeline was developed and
  tested on an NVIDIA GeForce RTX 4070 laptop GPU (8 GB VRAM) with
  `pytorch-cuda=12.1`. Training and inference will also run on CPU, just
  much more slowly. Inference-only use on a modern CPU is reasonably
  practical; full training runs are not.

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate sticker-yolo
```

`environment.yml` pins the packages known to be required (PyTorch +
CUDA 12.1, Ultralytics YOLO, OpenCV, pyzbar, zxingcpp, numpy). If your
machine has a different CUDA version or no GPU at all, edit the
`pytorch-cuda` line (or remove it for CPU-only) before creating the
environment.

**Note (pyzbar on Windows):** `pyzbar` depends on the ZBar shared library.
If `import pyzbar` fails after installation, install the ZBar DLL
separately (e.g. via the `pyzbar` package's own Windows wheel, which
usually bundles it) — see the
[pyzbar README](https://github.com/NaturalHistoryMuseum/pyzbar) for
platform-specific notes.

## 2. Verify the environment

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import cv2, ultralytics, pyzbar, zxingcpp; print('OK')"
```

## 3. Prepare a dataset (only needed if you're retraining)

Raw input is a flat folder of images with matching YOLO-format `.txt`
labels (classes: `0 qrcode`, `1 yellow_triangle`, `2 green_label`,
`3 fuse_cover`). Full step-by-step commands (split → validate → tile →
train, for both the small- and large-object models) are documented in
[`src/yolo_pipeline/README.md`](src/yolo_pipeline/README.md) — follow the
"How to run end-to-end" section there. Put your raw dataset under
`data/` (see [`data/README.md`](data/README.md)).

## 4. Get trained weights

Trained weights (`best.pt` for both the small- and large-object models)
are not committed to this repository — factory training images and
resulting weights may be proprietary, and weight files are also too
large for a plain git repo. Two options for graders:

- **Use provided weights:** if weights have been shared separately (e.g.
  a download link), place them under `models/` — see
  [`models/README.md`](models/README.md) for expected filenames.
- **Train your own (short) run:** `save_period=10` is set in both
  training scripts, so you don't need to wait for full convergence to
  sanity-check the pipeline — train for 10–20 epochs and use the
  resulting `weights/last.pt`.

```bash
python src/yolo_pipeline/4_train.py       --data data/tiled_dataset/dataset.yaml       --name sticker_v1
python src/yolo_pipeline/4b_train_large.py --data data/large_object_dataset/dataset.yaml --name sticker_large_v1
```

## 5. Run inference + inspection on an image

```bash
# Detect stickers (writes a YOLO-format .txt next to the image)
python src/yolo_pipeline/5_predict.py \
    --model       models/sticker_v1_best.pt \
    --model_large models/sticker_large_v1_best.pt \
    --img         path/to/image.jpg

# Inspect the resulting image + label pair, print a PASS/FAIL/UNKNOWN table
python src/inspection_pipeline/main.py path/to/image.jpg path/to/image.txt

# Or process a whole folder of image+label pairs at once
python src/inspection_pipeline/main.py --folder path/to/folder
```

Add `--json` to `main.py` for machine-readable output instead of the
default ASCII table.

## Testing without training anything

If you just want to confirm the *inspection* half of the pipeline works
(orientation/presence logic, independent of the detector), you only need
one image and one matching YOLO-format `.txt` label file — you can hand-
label a couple of stickers yourself, or use any `.txt` produced by
`5_predict.py`. Then run step 5 above directly, skipping steps 3–4
entirely.
