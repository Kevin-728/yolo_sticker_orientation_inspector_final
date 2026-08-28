# Attribution

## Libraries & Frameworks

| Library | Used for | License (as of writing — verify against the package's own repo) |
|---|---|---|
| [Ultralytics YOLO (YOLOv11)](https://github.com/ultralytics/ultralytics) | Object detection backbone/training/inference for both the small- and large-object models | AGPL-3.0 |
| [PyTorch](https://pytorch.org/) | Deep learning framework underlying Ultralytics | BSD-3-Clause |
| [OpenCV (opencv-python)](https://github.com/opencv/opencv-python) | Image I/O, resizing/cropping, contour/finder-pattern geometry, NMS, thresholding | Apache 2.0 |
| [NumPy](https://numpy.org/) | Array/vector math throughout the detection and inspection pipelines | BSD-3-Clause |
| [pyzbar](https://github.com/NaturalHistoryMuseum/pyzbar) | QR code decoding (Stage 2/3 of the QR orientation fallback chain) | MIT |
| [zxingcpp](https://github.com/zxing-cpp/zxing-cpp) (Python bindings) | QR code decoding (Stage 2/3 fallback, alternate decoder) | Apache 2.0 |

Python standard library modules (`argparse`, `pathlib`, `random`, `shutil`,
`itertools`, `math`, `json`, `os`, `sys`) are used throughout and are not
separately licensed.

**Note on Ultralytics' AGPL-3.0 license:** AGPL-3.0 is a strong copyleft
license. If this project (or a service built on it) is distributed or
offered as a network service to others, review Ultralytics' licensing
terms (including their commercial license option) to confirm compliance
before deployment.

## Datasets

The initial labeled training data (~300 images) was from bosch factory's own 
product images, and was **not** sourced from a public dataset. This data
originates from an internal factory quality-control process and may be
proprietary to the organization it was collected for.


## Pretrained Weights

`yolo11s.pt` (the pretrained YOLOv11-small backbone used as the starting
point for both fine-tuning runs, per `src/yolo_pipeline/4_train.py` and
`4b_train_large.py`) is downloaded automatically by Ultralytics on first
run, from Ultralytics' own release assets. See the Ultralytics repository
for the corresponding license/terms of that checkpoint.
