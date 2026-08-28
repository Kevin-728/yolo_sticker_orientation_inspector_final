# yolo_converter.py
"""
Convert YOLO-format axis-aligned bounding boxes into polygon dicts
required by orientation_inspector.py.

Image size (always 4080 x 3060 for this project).

Class mapping:
    0 -> qrcode
    1 -> yellow_triangle
    2 -> green_label   (ignored)
    3 -> fuse_cover
"""

IMG_W = 4080
IMG_H = 3060

CLASS_MAP = {
    0: "qrcode",
    1: "yellow_triangle",
    2: "green_label",
    3: "fuse_cover",
}


def yolo_box_to_polygon(cx_n, cy_n, w_n, h_n, img_w=IMG_W, img_h=IMG_H):
    """
    Convert normalised YOLO bbox to 4-corner polygon.

    Output corner order:
        top-left, top-right, bottom-right, bottom-left
    """
    cx = cx_n * img_w
    cy = cy_n * img_h
    w  = w_n  * img_w
    h  = h_n  * img_h

    x1 = round(cx - w / 2)
    y1 = round(cy - h / 2)
    x2 = round(cx + w / 2)
    y2 = round(cy + h / 2)

    return [
        (x1, y1),   # top-left
        (x2, y1),   # top-right
        (x2, y2),   # bottom-right
        (x1, y2),   # bottom-left
    ]


def load_yolo_labels(txt_path, img_w=IMG_W, img_h=IMG_H):
    """
    Read a YOLO label txt and return a flat detections dict:

        {
            "fuse_cover":      [polygon, ...],   # each polygon = [(x,y), ...]
            "qrcode":          [polygon, ...],
            "yellow_triangle": [polygon, ...],
        }
    """
    detections = {
        "fuse_cover":      [],
        "qrcode":          [],
        "yellow_triangle": [],
    }

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:5])

            cls_name = CLASS_MAP.get(cls_id)
            if cls_name is None or cls_name == "green_label":
                continue

            polygon = yolo_box_to_polygon(cx, cy, w, h, img_w, img_h)
            detections[cls_name].append(polygon)

    return detections
