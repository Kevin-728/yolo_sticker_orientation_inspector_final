# Sticker Orientation & Presence Inspector

An industrial computer-vision system that inspects high-resolution factory
images of EV sub-assemblies ("truck" products) and reports a PASS / FAIL /
UNKNOWN quality-control verdict for each product's sticker orientation and
label presence.

## What it Does

The system takes a 4080×3060 photo of up to four assembled products and
determines, per product, whether each required sticker — a QR code, a
yellow warning triangle, a green compliance label, and a fuse cover — is
present and correctly oriented. It solves this with a two-model YOLOv11
detection pipeline: a tiled + 2× upsampled model that recovers detectability
for stickers as small as ~10 px raw (qrcode, yellow_triangle), and a
separate full-image model for stickers that are far larger than any
reasonable tiling window (green_label, fuse_cover). The raw detections are
grouped into individual products by nearest-fuse-cover assignment, and each
component's orientation is checked against the product's own "up" direction
— derived from the fuse cover's geometry and the QR codes' position — using
a fallback chain of finder-pattern geometry and multiple QR decoders for
QR codes, and a projection-based apex detector for the triangles. A missing
green_label is treated as a real defect. The final output is a compact
per-product summary table (or JSON) suitable for a factory QC line.

## Quick Start

```bash
# 1. Clone and set up the environment (see SETUP.md for full detail)
git clone <this-repo-url>
cd <this-repo>
conda env create -f environment.yml
conda activate sticker-yolo

# 2. Run detection on a raw image (requires trained weights — see models/)
python src/yolo_pipeline/5_predict.py \
    --model       models/sticker_v1_best.pt \
    --model_large models/sticker_large_v1_best.pt \
    --img         path/to/image.jpg
# -> writes path/to/image.txt (YOLO-format detections)

# 3. Run the orientation/presence inspection on that image + label pair
python src/inspection_pipeline/main.py path/to/image.jpg path/to/image.txt
```

See [`SETUP.md`](SETUP.md) for full installation instructions, dataset
preparation, training, and how to test the system without training your
own weights. See [`src/yolo_pipeline/README.md`](src/yolo_pipeline/README.md)
for a detailed technical writeup of the detection pipeline (why two
models, the tiling/upsampling strategy, and every bug fix that shaped it).

## Evaluation

Successfully identify all the intended objects and put into industrial 
product line usage.

