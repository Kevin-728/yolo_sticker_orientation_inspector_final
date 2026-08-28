# orientation_inspector.py
"""
Sticker Orientation Validation – core inspection logic.

Components inspected
--------------------
  fuse_cover      : primary orientation reference (long axis = product vertical)
  qrcode          : validated via multi-stage orientation detection
  yellow_triangle : validated via project-and-spread apex detection

QR detection stages
-------------------
  Stage 1 – Finder-pattern geometry (primary, decode-free):
             Detects the 3 nested-square QR finder patterns in the
             upscaled+CLAHE crop, identifies TL/TR/BL from geometry, derives
             qr_up without any decoder corner-ordering assumptions.
             Result is ONLY used when the angle is unambiguous:
               |angle| < 45°  → clear PASS
               |angle| > 135° → clear FAIL
             Otherwise the result is discarded (ambiguous detection) and
             Stage 2 is attempted.

  Stage 2 – 4-library decoder chain on original crop (all quiet-zone margins):
             Methods tried in order: pyzbar, zxingcpp, opencv_detect,
             opencv_decode.  pyzbar and zxingcpp carry explicit corner
             labels (top_left / top_right / …) so they handle QR rotation
             correctly.  opencv methods are tried as a final fallback.

  Stage 3 – Same decoder chain on preprocessed (4 times upscale + CLAHE) crop.

Pass/fail logic
---------------
  relative_angle ≈   0° → PASS
  relative_angle ≈ 180° → FAIL
  component missing      → UNKNOWN

Overall product status priority: FAIL > UNKNOWN > PASS
"""

import cv2
import itertools
import math
import numpy as np

# ── Optional QR libraries ──────────────────────────────────────────────────────
try:
    from pyzbar import pyzbar as _pyzbar
    _PYZBAR_OK = True
except ImportError:
    _PYZBAR_OK = False

try:
    import zxingcpp as _zxing
    _ZXING_OK = True
except ImportError:
    _ZXING_OK = False


# ── Configuration ──────────────────────────────────────────────────────────────
THRESHOLD_QR            = 60.0   # degrees – pass if angular distance from 0° < this
THRESHOLD_TRIANGLE      = 60.0   # degrees – pass if angular distance from 0° < this
BLACK_PIXEL_THRESH      = 80     # grayscale ≤ this → black pixel (fixed threshold)
BLACK_MIN_PIXELS        = 5      # fewer pixels found → UNKNOWN
BLACK_FIXED_MIN         = 20     # min pixels required to trust fixed threshold;
                                  # below this, fall back to Otsu (handles stickers
                                  # whose black triangle sits in the 80-149 gray band)
QR_DETECT_MARGINS       = list(range(0, 55, 5))
QR_PREPROCESS_SCALE     = 4      # upscale factor for Stages 1 & 3
FP_MIN_AREA_RATIO       = 0.10   # discard finder-pattern candidates smaller than
                                  # this fraction of the largest candidate area
FP_CLEAR_PASS_THRESH    = 45.0   # Stage 1 result accepted as PASS if |angle| < this
FP_CLEAR_FAIL_THRESH    = 135.0  # Stage 1 result accepted as FAIL if |angle| > this


# ── Geometry helpers ───────────────────────────────────────────────────────────

def parse_polygon(points):
    return np.array(points, dtype=np.float32)


def _fuse_long_axis(fuse_pts):
    pts_cv = fuse_pts.reshape(-1, 1, 2).astype(np.float32)
    box    = cv2.boxPoints(cv2.minAreaRect(pts_cv))
    e1, e2 = box[1] - box[0], box[2] - box[1]
    long   = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    return long / np.linalg.norm(long)


def get_fuse_cover_frame(fuse_pts, qr_pts_list):
    """
    Compute fuse_cover coordinate frame.
    Primary:  QR centroid direction (away from QRs = product up).
    Fallback: minAreaRect long axis (when no QR data available).
    """
    fuse_center  = np.mean(fuse_pts, axis=0)
    qr_centroids = [np.mean(qr, axis=0) for qr in qr_pts_list if qr is not None]

    if qr_centroids:
        to_qr = np.mean(qr_centroids, axis=0) - fuse_center
        norm  = np.linalg.norm(to_qr)
        up_vec = -to_qr / norm if norm > 1e-6 else _fuse_long_axis(fuse_pts)
    else:
        up_vec = _fuse_long_axis(fuse_pts)

    right_vec = np.array([-up_vec[1], up_vec[0]])
    return fuse_center, up_vec, right_vec


def signed_angle_deg(v1, v2):
    cross = float(v1[0]) * float(v2[1]) - float(v1[1]) * float(v2[0])
    dot   = float(v1[0]) * float(v2[0]) + float(v1[1]) * float(v2[1])
    return math.degrees(math.atan2(cross, dot))


def angular_distance(angle_deg):
    d = abs(angle_deg)
    return min(d, 360.0 - d)


def passes(angle_deg, threshold):
    return angular_distance(angle_deg) < threshold


# ── Image helpers ──────────────────────────────────────────────────────────────

def crop_bbox(img, polygon):
    pts = np.array(polygon, dtype=np.int32)
    x1, y1 = pts.min(axis=0);  x2, y2 = pts.max(axis=0)
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(img.shape[1] - 1, x2);  y2 = min(img.shape[0] - 1, y2)
    return img[y1:y2 + 1, x1:x2 + 1].copy(), int(x1), int(y1)


def black_threshold_for_crop(gray):
    """
    Choose the best binarisation threshold for isolating dark pixels.

    Strategy
    --------
    1. Count pixels below the fixed threshold (BLACK_PIXEL_THRESH = 80).
    2. If that count is >= BLACK_FIXED_MIN (20), the sticker has clearly dark
       pixels - use the fixed threshold as-is.
    3. Otherwise the triangle is drawn in the mid-grey band (80-149 typical for
       a coloured sticker on a bright background).  Fall back to Otsu so that
       the bimodal split between sticker and background is used.
    """
    n_fixed = int(np.sum(gray <= BLACK_PIXEL_THRESH))
    if n_fixed >= BLACK_FIXED_MIN:
        return BLACK_PIXEL_THRESH
    thr, _ = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return int(thr)


# ── Triangle orientation ───────────────────────────────────────────────────────

def estimate_triangle_orientation(img, polygon, up_vec, right_vec, fuse_center):
    """
    Project-and-spread apex detection.
    Narrower half along right_vec = apex side.
    relative_angle ≈ 0° → PASS;  ≈ 180° → FAIL.
    """
    crop, ox, oy = crop_bbox(img, polygon)
    if crop.size == 0:
        return None, "UNKNOWN"

    gray = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if len(crop.shape) == 3 else crop.copy())
    thr = black_threshold_for_crop(gray)
    _, black_mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

    yy, xx = np.where(black_mask > 0)
    if len(yy) < BLACK_MIN_PIXELS:
        return None, "UNKNOWN"

    pts   = np.column_stack([xx.astype(float), yy.astype(float)])
    pts_c = pts - pts.mean(axis=0)

    proj_up    = pts_c @ up_vec
    proj_right = pts_c @ right_vec

    upper_mask = proj_up >  0
    lower_mask = proj_up <= 0

    def perp_spread(mask):
        sub = proj_right[mask]
        return float(np.std(sub)) if mask.sum() >= 2 else 1e9

    sp_upper = perp_spread(upper_mask)
    sp_lower = perp_spread(lower_mask)

    apex_pts = pts_c[upper_mask] if sp_upper <= sp_lower else pts_c[lower_mask]
    if len(apex_pts) < 2:
        return None, "UNKNOWN"

    apex_dir = apex_pts.mean(axis=0)
    norm = np.linalg.norm(apex_dir)
    if norm < 1e-6:
        return None, "UNKNOWN"
    apex_dir /= norm

    rel_angle = signed_angle_deg(apex_dir, up_vec)
    status = "PASS" if passes(rel_angle, THRESHOLD_TRIANGLE) else "FAIL"
    return round(rel_angle, 2), status


# ── QR orientation ─────────────────────────────────────────────────────────────

# ── Preprocessing ──────────────────────────────────────────────────────────────

def _preprocess_qr_crop(crop):
    """4× upscale + CLAHE. Returns (bgr, scale)."""
    s    = QR_PREPROCESS_SCALE
    h, w = crop.shape[:2]
    big  = cv2.resize(crop, (w * s, h * s), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY) if len(big.shape) == 3 else big
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), s


# ── Stage 1: Finder-pattern geometry ──────────────────────────────────────────

def _fp_candidates(gray):
    """
    Return (cx, cy, area) for innermost nested-square contours —
    these are the inner cores of QR finder patterns.
    """
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    h = hierarchy[0]
    out = []
    for i, cnt in enumerate(contours):
        if h[i][3] < 0 or h[h[i][3]][3] < 0:   # needs 2 ancestors
            continue
        x, y, w, hh = cv2.boundingRect(cnt)
        if min(w, hh) < 3 or max(w, hh) / max(min(w, hh), 1) > 2.5:
            continue
        M = cv2.moments(cnt)
        if M['m00'] < 1e-6:
            continue
        out.append([M['m10'] / M['m00'], M['m01'] / M['m00'], w * hh])
    return out


def _cluster_points(pts, thr):
    """Merge points within *thr* distance; return ndarray of cluster centres."""
    used = [False] * len(pts);  out = []
    for i, p in enumerate(pts):
        if used[i]:
            continue
        grp = [p];  used[i] = True
        for j in range(i + 1, len(pts)):
            if not used[j] and np.linalg.norm(p - pts[j]) < thr:
                grp.append(pts[j]);  used[j] = True
        out.append(np.mean(grp, axis=0))
    return np.array(out)


def _score_triangle(pts3):
    """High score = one near-90° corner, two roughly equal legs."""
    best_score = -1;  best_i = -1
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        v1 = pts3[j] - pts3[i];  v2 = pts3[k] - pts3[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        score = (1.0 - abs(np.dot(v1, v2) / (n1 * n2))) * (min(n1, n2) / max(n1, n2))
        if score > best_score:
            best_score = score;  best_i = i
    return best_score, best_i


def _corners_finder_pattern(crop):
    """
    Estimate QR corners from finder-pattern geometry (no decoder required).

    Returns canonical (4,2) corners [TL, TR, BR, BL] in *crop* coordinates,
    or None if detection failed.
    """
    preprocessed, scale = _preprocess_qr_crop(crop)
    gray  = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
    cands = _fp_candidates(gray)
    if not cands:
        return None

    max_area = max(a for _, _, a in cands)
    cands    = [[x, y] for x, y, a in cands if a >= FP_MIN_AREA_RATIO * max_area]
    if len(cands) < 3:
        return None

    cs = _cluster_points(np.array(cands, dtype=np.float32),
                          min(preprocessed.shape[:2]) * 0.15)
    if len(cs) < 3:
        return None

    best_score = -1;  best_trio = None;  best_corner = -1
    for combo in itertools.combinations(range(len(cs)), 3):
        pts3 = cs[list(combo)]
        sc, ci = _score_triangle(pts3)
        if sc > best_score:
            best_score  = sc;  best_trio = pts3;  best_corner = ci

    if best_trio is None or best_corner < 0:
        return None

    tl = best_trio[best_corner]
    p  = best_trio[(best_corner + 1) % 3]
    q  = best_trio[(best_corner + 2) % 3]
    cross = (p[0] - tl[0]) * (q[1] - tl[1]) - (p[1] - tl[1]) * (q[0] - tl[0])
    tr, bl = (p, q) if cross > 0 else (q, p)
    br = tr + bl - tl

    return np.array([tl, tr, br, bl], dtype=np.float32) / scale


# ── Stage 2 / 3: decoder chain ─────────────────────────────────────────────────
# Methods ordered by corner-label reliability:
#   pyzbar   – explicit CCW polygon (reordered to canonical below)
#   zxingcpp – explicit top_left / top_right / bottom_right / bottom_left
#   opencv_detect  – corner detection only, sometimes correct ordering
#   opencv_decode  – last resort, ordering depends on QR rotation

def _corners_pyzbar(crop):
    if not _PYZBAR_OK:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    results = _pyzbar.decode(gray)
    if not results:
        return None
    poly = results[0].polygon
    if len(poly) != 4:
        return None
    tl, bl, br, tr = (np.array([p.x, p.y], dtype=np.float32) for p in poly)
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _corners_zxing(crop):
    if not _ZXING_OK:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    results = _zxing.read_barcodes(gray)
    for r in results:
        pos = r.position
        tl = np.array([pos.top_left.x,     pos.top_left.y],     dtype=np.float32)
        tr = np.array([pos.top_right.x,    pos.top_right.y],    dtype=np.float32)
        br = np.array([pos.bottom_right.x, pos.bottom_right.y], dtype=np.float32)
        bl = np.array([pos.bottom_left.x,  pos.bottom_left.y],  dtype=np.float32)
        return np.array([tl, tr, br, bl], dtype=np.float32)
    return None


def _corners_opencv_detect(crop):
    detector = cv2.QRCodeDetector()
    retval, points = detector.detectMulti(crop)
    if retval and points is not None and len(points) > 0:
        c = np.array(points[0], dtype=np.float32)
        if c.shape == (4, 2):
            return c
    return None


def _corners_opencv_decode(crop):
    detector = cv2.QRCodeDetector()
    retval, _, points, _ = detector.detectAndDecodeMulti(crop)
    if retval and points is not None and len(points) > 0:
        c = np.array(points[0], dtype=np.float32)
        if c.shape == (4, 2):
            return c
    return None


_QR_DECODER_METHODS = [
    _corners_pyzbar,
    _corners_zxing,
    _corners_opencv_detect,
    _corners_opencv_decode,   # last: may have wrong ordering for rotated QRs
]


def _try_decoder_chain(crop):
    for method in _QR_DECODER_METHODS:
        c = method(crop)
        if c is not None:
            return c
    return None


# ── Angle from corners ─────────────────────────────────────────────────────────

def _qr_angle_from_corners(corners_raw, up_vec, right_vec):
    p0, p1, p2, p3 = corners_raw
    qr_up = (p0 + p1) / 2.0 - (p2 + p3) / 2.0
    norm  = np.linalg.norm(qr_up)
    if norm < 1e-6:
        return None, "UNKNOWN"
    qr_up /= norm

    qr_up_fc = np.array([np.dot(qr_up, right_vec), np.dot(qr_up, up_vec)])
    nf = np.linalg.norm(qr_up_fc)
    if nf < 1e-6:
        return None, "UNKNOWN"
    qr_up_fc /= nf

    rel_angle = signed_angle_deg(qr_up_fc, np.array([0.0, 1.0]))
    status    = "PASS" if passes(rel_angle, THRESHOLD_QR) else "FAIL"
    return round(rel_angle, 2), status


# ── Main QR orientation estimator ─────────────────────────────────────────────

def estimate_qr_orientation(img, polygon, up_vec, right_vec, fuse_center):
    """
    Three-stage QR orientation estimation.

    Stage 1 – Finder-pattern geometry on preprocessed base crop.
              Result accepted only when angle is unambiguous
              (|angle| < FP_CLEAR_PASS_THRESH  or  |angle| > FP_CLEAR_FAIL_THRESH).
              Otherwise discarded; decoder stages are attempted.

    Stage 2 – Decoder chain (pyzbar → zxingcpp → detect → decode)
              on original crop with progressively wider quiet-zone margins.

    Stage 3 – Same decoder chain on 4× upscaled + CLAHE crop.
    """
    pts_bbox = np.array(polygon, dtype=np.int32)
    bx1, by1 = pts_bbox.min(axis=0)
    bx2, by2 = pts_bbox.max(axis=0)
    bx1 = max(0, bx1);  by1 = max(0, by1)
    bx2 = min(img.shape[1] - 1, bx2);  by2 = min(img.shape[0] - 1, by2)

    base_crop   = img[by1:by2 + 1, bx1:bx2 + 1]
    base_offset = np.array([bx1, by1], dtype=np.float32)

    # ── Stage 1: finder-pattern ────────────────────────────────────────────────
    fp_corners = _corners_finder_pattern(base_crop)
    if fp_corners is not None:
        corners_raw = fp_corners + base_offset
        rel_angle, status = _qr_angle_from_corners(corners_raw, up_vec, right_vec)
        if rel_angle is not None:
            ad = angular_distance(rel_angle)
            if ad < FP_CLEAR_PASS_THRESH or ad > FP_CLEAR_FAIL_THRESH:
                return rel_angle, status   # unambiguous result

    # ── Stage 2: decoder chain on original crop ────────────────────────────────
    for margin in QR_DETECT_MARGINS:
        x1 = max(0, bx1 - margin);  y1 = max(0, by1 - margin)
        x2 = min(img.shape[1] - 1, bx2 + margin)
        y2 = min(img.shape[0] - 1, by2 + margin)
        crop   = img[y1:y2 + 1, x1:x2 + 1]
        offset = np.array([x1, y1], dtype=np.float32)
        c = _try_decoder_chain(crop)
        if c is not None:
            return _qr_angle_from_corners(c + offset, up_vec, right_vec)

    # ── Stage 3: decoder chain on preprocessed crop ───────────────────────────
    for margin in QR_DETECT_MARGINS:
        x1 = max(0, bx1 - margin);  y1 = max(0, by1 - margin)
        x2 = min(img.shape[1] - 1, bx2 + margin)
        y2 = min(img.shape[0] - 1, by2 + margin)
        crop   = img[y1:y2 + 1, x1:x2 + 1]
        offset = np.array([x1, y1], dtype=np.float32)
        preprocessed, scale = _preprocess_qr_crop(crop)
        c = _try_decoder_chain(preprocessed)
        if c is not None:
            return _qr_angle_from_corners(c / scale + offset, up_vec, right_vec)

    return None, "UNKNOWN"


# ── Overall status helper ──────────────────────────────────────────────────────

def _combine_status(*statuses):
    if "FAIL"    in statuses: return "FAIL"
    if "UNKNOWN" in statuses: return "UNKNOWN"
    return "PASS"


# ── Per-product inspection ─────────────────────────────────────────────────────

def inspect_product(img, product):
    fuse_poly = parse_polygon(product["fuse_cover"])
    qr_parsed = [parse_polygon(p) for p in product["qrcode"] if p is not None]
    fuse_center, up_vec, right_vec = get_fuse_cover_frame(fuse_poly, qr_parsed)

    result = {
        "fuse_cover": {
            "up_vector": [float(up_vec[0]), float(up_vec[1])],
            "status":    "REFERENCE",
        },
        "qrcode":          [],
        "yellow_triangle": [],
        "overall":         "PASS",
    }
    all_statuses = []

    for i, poly in enumerate(product["qrcode"]):
        if poly is None:
            entry = {"id": i + 1, "status": "UNKNOWN", "note": "not detected"}
            all_statuses.append("UNKNOWN")
        else:
            rel_angle, status = estimate_qr_orientation(
                img, poly, up_vec, right_vec, fuse_center)
            entry = {"id": i + 1, "status": status}
            if rel_angle is not None:
                entry["relative_angle"] = rel_angle
            all_statuses.append(status)
        result["qrcode"].append(entry)

    for i, poly in enumerate(product["yellow_triangle"]):
        if poly is None:
            entry = {"id": i + 1, "status": "UNKNOWN", "note": "not detected"}
            all_statuses.append("UNKNOWN")
        else:
            rel_angle, status = estimate_triangle_orientation(
                img, poly, up_vec, right_vec, fuse_center)
            entry = {"id": i + 1, "status": status}
            if rel_angle is not None:
                entry["relative_angle"] = rel_angle
            all_statuses.append(status)
        result["yellow_triangle"].append(entry)

    result["overall"] = _combine_status(*all_statuses)
    return result


def run_inspection(img_path, products):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot load image: {img_path}")
    results = []
    for i, product in enumerate(products):
        res = inspect_product(img, product)
        res["product_id"] = i + 1
        results.append(res)
    return results
