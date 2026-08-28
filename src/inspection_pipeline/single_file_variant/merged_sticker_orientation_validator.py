# sticker_orientation_validator.py
"""
Sticker Orientation Validation – single-class integration.

Merges yolo_converter.py, product_grouper.py, and orientation_inspector.py
into one self-contained class.

Detailed logic can be found under each function.

Usage
-----
    import cv2
    from sticker_orientation_validator import StickerOrientationValidator

    results = StickerOrientationValidator("product.jpg", "product.txt").run()
    for r in results:
        print(r["product_id"], r["overall"], r["failures"])

Input
-----
    img_path : path to the raw product image (jpg / png / bmp).
    txt_path : path to the matching YOLO-format label file (.txt).

Output schema (one row per product)
-------------------------------------
    {
        "product_id":      int,
        "overall":         "PASS" | "FAIL" | "UNKNOWN",
        "failures":        [str, ...],   # non-empty when overall != PASS
    }

Output example
--------------
    1 UNKNOWN ['qrcode_1: UNKNOWN', 'qrcode_2: UNKNOWN']
    2 PASS []
"""

import cv2
import itertools
import math

import numpy as np

# ── Optional QR libraries (graceful degradation if not installed) ──────────────
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


class StickerOrientationValidator:
    """
    Validates QR-code and yellow warning-triangle sticker orientations.

    All logic is identical to the original three source files:
      yolo_converter.py     – YOLO label parsing
      product_grouper.py    – component-to-product assignment
      orientation_inspector.py – orientation estimation & pass/fail

    QR detection pipeline (3 stages per QR code)
    ---------------------------------------------
    Stage 1 – Finder-pattern geometry (decode-free).  Accepted only when
              unambiguous: |angle| < 45° (PASS) or > 135° (FAIL).
    Stage 2 – Decoder chain on original crop with widening quiet-zone
              margins: pyzbar → zxingcpp → opencv_detect → opencv_decode.
    Stage 3 – Same chain on 4× upscaled + CLAHE crop.

    Triangle orientation (project-and-spread apex detection)
    --------------------------------------------------------
    Black pixels extracted → mean-centred → split along up_vec.
    Narrower lateral half = apex side.  Signed angle of apex_dir vs up_vec.

    Fuse-cover frame
    ----------------
    Primary:  QR centroid direction (away from QRs = product up).
    Fallback: minAreaRect long axis (when no QR data available).

    Pass / fail
    -----------
    relative_angle ≈   0° → PASS
    relative_angle ≈ 180° → FAIL
    undetectable         → UNKNOWN
    Overall:  FAIL > UNKNOWN > PASS
    """

    # ── Constants from yolo_converter.py ─────────────────────────────────────
    IMG_W = 4080
    IMG_H = 3060
    CLASS_MAP = {
        0: "qrcode",
        1: "yellow_triangle",
        2: "green_label",
        3: "fuse_cover",
    }

    # ── Constants from orientation_inspector.py ───────────────────────────────
    THRESHOLD_QR         = 60.0   # degrees – PASS if angular_distance(rel_angle) < this
    THRESHOLD_TRIANGLE   = 60.0   # degrees – PASS if angular_distance(rel_angle) < this
    BLACK_PIXEL_THRESH   = 80     # grayscale ≤ this → black pixel (fixed threshold)
    BLACK_MIN_PIXELS     = 5      # fewer qualifying pixels → UNKNOWN
    BLACK_FIXED_MIN      = 20     # min pixels to trust fixed threshold; else use Otsu
    QR_DETECT_MARGINS    = list(range(0, 55, 5))
    QR_PREPROCESS_SCALE  = 4      # upscale factor for Stages 1 & 3
    FP_MIN_AREA_RATIO    = 0.10   # discard finder-pattern candidates below this ratio
    FP_CLEAR_PASS_THRESH = 45.0   # Stage 1 accepted as PASS if |angle| < this
    FP_CLEAR_FAIL_THRESH = 135.0  # Stage 1 accepted as FAIL if |angle| > this

    # ── Constants from product_grouper.py ────────────────────────────────────
    QR_PER_PRODUCT  = 2
    TRI_PER_PRODUCT = 3

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, img_path: str, txt_path: str) -> None:
        """
        Input Parameters
        ----------
        img_path : path to the raw product image (jpg / png / bmp).
        txt_path : path to the matching YOLO-format label file (.txt).
                   Image dimensions are assumed to be IMG_W × IMG_H (4080 × 3060).
        """
        self._img: np.ndarray = cv2.imread(img_path)
        if self._img is None:
            raise ValueError(f"Cannot load image: {img_path!r}")
        self._detections = self._load_yolo_labels(txt_path)

    # ══════════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════════

    def run(self) -> list:
        """
        Run the full pipeline: group → inspect → return results.

        Returns one dict per product (see module docstring for schema).
        Per-product exceptions are caught and returned as UNKNOWN + error key
        so that one bad product does not abort the batch.
        """
        products = self._group_by_fuse_cover(self._detections)
        if not products:
            return []
        results = []
        for idx, product in enumerate(products):
            try:
                result = self._inspect_product(product)
            except Exception as exc:
                result = {
                    "fuse_cover":      {"up_vector": [0.0, -1.0], "status": "REFERENCE"},
                    "qrcode":          [],
                    "yellow_triangle": [],
                    "overall":         "UNKNOWN",
                    "failures":        [],
                    "error":           str(exc),
                }
            result["product_id"] = idx + 1
            results.append(result)
        return results

    # ══════════════════════════════════════════════════════════════════════════
    # yolo_converter.py logic
    # ══════════════════════════════════════════════════════════════════════════

    def _yolo_box_to_polygon(self, cx_n, cy_n, w_n, h_n):
        """
        Convert normalised YOLO bbox to 4-corner polygon.
        Identical to yolo_converter.yolo_box_to_polygon().

        Output corner order: top-left, top-right, bottom-right, bottom-left.
        """
        cx = cx_n * self.IMG_W
        cy = cy_n * self.IMG_H
        w  = w_n  * self.IMG_W
        h  = h_n  * self.IMG_H

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

    def _load_yolo_labels(self, txt_path: str) -> dict:
        """
        Read a YOLO label txt and return a flat detections dict.
        Identical to yolo_converter.load_yolo_labels().

        Returns
        -------
        {
            "fuse_cover":      [polygon, ...],
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
                parts  = line.split()
                cls_id = int(parts[0])
                cx, cy, w, h = (float(p) for p in parts[1:5])

                cls_name = self.CLASS_MAP.get(cls_id)
                if cls_name is None or cls_name == "green_label":
                    continue

                polygon = self._yolo_box_to_polygon(cx, cy, w, h)
                detections[cls_name].append(polygon)

        return detections

    # ══════════════════════════════════════════════════════════════════════════
    # product_grouper.py logic
    # ══════════════════════════════════════════════════════════════════════════

    def _group_by_fuse_cover(self, detections: dict) -> list:
        """
        Assign QR codes and yellow triangles to their nearest fuse cover.
        Identical to product_grouper.group_by_fuse_cover().
        """
        fuse_polys = detections.get("fuse_cover", [])
        qr_polys   = list(detections.get("qrcode", []))
        tri_polys  = list(detections.get("yellow_triangle", []))

        if not fuse_polys:
            return []

        fuse_centroids = [self._centroid(p) for p in fuse_polys]
        qr_centroids   = [self._centroid(p) for p in qr_polys]
        tri_centroids  = [self._centroid(p) for p in tri_polys]

        # Track which components are still unassigned
        qr_available  = list(range(len(qr_polys)))
        tri_available = list(range(len(tri_polys)))

        products = []
        for fuse_idx, fuse_center in enumerate(fuse_centroids):

            # Assign QR codes
            assigned_qr = self._assign_nearest(
                fuse_center, qr_centroids, qr_available, self.QR_PER_PRODUCT
            )
            qr_slots = [qr_polys[i] if i is not None else None
                        for i in assigned_qr]

            # Assign yellow triangles
            assigned_tri = self._assign_nearest(
                fuse_center, tri_centroids, tri_available, self.TRI_PER_PRODUCT
            )
            tri_slots = [tri_polys[i] if i is not None else None
                         for i in assigned_tri]

            products.append({
                "fuse_cover":      fuse_polys[fuse_idx],
                "qrcode":          qr_slots,
                "yellow_triangle": tri_slots,
            })

        return products

    @staticmethod
    def _assign_nearest(fuse_center, all_centroids, available_indices, n_needed):
        """
        From available_indices pick n_needed entries closest to fuse_center.
        Removes selected indices from available_indices in-place.
        Returns list of length n_needed (None for missing slots).
        Identical to product_grouper._assign_nearest().
        """
        if not available_indices:
            return [None] * n_needed

        distances = [
            (np.linalg.norm(all_centroids[i] - fuse_center), i)
            for i in available_indices
        ]
        distances.sort(key=lambda t: t[0])

        selected = []
        for _, idx in distances[:n_needed]:
            selected.append(idx)
            available_indices.remove(idx)

        # Pad with None if not enough components were available
        while len(selected) < n_needed:
            selected.append(None)

        return selected

    @staticmethod
    def _centroid(polygon) -> np.ndarray:
        """Mean of polygon vertices as float array (2,).
        Identical to product_grouper._centroid()."""
        return np.mean(np.array(polygon, dtype=np.float64), axis=0)

    # ══════════════════════════════════════════════════════════════════════════
    # orientation_inspector.py logic – geometry helpers
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_polygon(points):
        return np.array(points, dtype=np.float32)

    @staticmethod
    def _fuse_long_axis(fuse_pts):
        pts_cv = fuse_pts.reshape(-1, 1, 2).astype(np.float32)
        box    = cv2.boxPoints(cv2.minAreaRect(pts_cv))
        e1, e2 = box[1] - box[0], box[2] - box[1]
        long   = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        norm   = np.linalg.norm(long)
        if norm < 1e-6:             # degenerate polygon – safe fallback
            return np.array([0.0, -1.0])
        return long / norm

    def _get_fuse_cover_frame(self, fuse_pts, qr_pts_list):
        """
        Compute fuse_cover coordinate frame.
        Primary:  QR centroid direction (away from QRs = product up).
        Fallback: minAreaRect long axis (when no QR data available).
        Identical to orientation_inspector.get_fuse_cover_frame().
        """
        fuse_center  = np.mean(fuse_pts, axis=0)
        qr_centroids = [np.mean(qr, axis=0) for qr in qr_pts_list if qr is not None]

        if qr_centroids:
            to_qr  = np.mean(qr_centroids, axis=0) - fuse_center
            norm   = np.linalg.norm(to_qr)
            up_vec = -to_qr / norm if norm > 1e-6 else self._fuse_long_axis(fuse_pts)
        else:
            up_vec = self._fuse_long_axis(fuse_pts)

        right_vec = np.array([-up_vec[1], up_vec[0]])
        return fuse_center, up_vec, right_vec

    @staticmethod
    def _signed_angle_deg(v1, v2):
        cross = float(v1[0]) * float(v2[1]) - float(v1[1]) * float(v2[0])
        dot   = float(v1[0]) * float(v2[0]) + float(v1[1]) * float(v2[1])
        return math.degrees(math.atan2(cross, dot))

    @staticmethod
    def _angular_distance(angle_deg):
        d = abs(angle_deg)
        return min(d, 360.0 - d)

    def _passes(self, angle_deg, threshold):
        return self._angular_distance(angle_deg) < threshold

    # ══════════════════════════════════════════════════════════════════════════
    # orientation_inspector.py logic – image helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _crop_bbox(self, polygon):
        """crop_bbox() from orientation_inspector.py, using self._img."""
        pts = np.array(polygon, dtype=np.int32)
        x1, y1 = pts.min(axis=0);  x2, y2 = pts.max(axis=0)
        x1 = max(0, x1);           y1 = max(0, y1)
        x2 = min(self._img.shape[1] - 1, x2)
        y2 = min(self._img.shape[0] - 1, y2)
        return self._img[y1:y2 + 1, x1:x2 + 1].copy(), int(x1), int(y1)

    def _black_threshold_for_crop(self, gray):
        """
        Choose the best binarisation threshold for isolating dark pixels.
        Identical to orientation_inspector.black_threshold_for_crop().

        1. Count pixels below BLACK_PIXEL_THRESH (80).
        2. If count >= BLACK_FIXED_MIN (20), use fixed threshold.
        3. Otherwise fall back to Otsu (handles mid-grey triangles, 80-149 band).
        """
        n_fixed = int(np.sum(gray <= self.BLACK_PIXEL_THRESH))
        if n_fixed >= self.BLACK_FIXED_MIN:
            return self.BLACK_PIXEL_THRESH
        thr, _ = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return int(thr)

    # ══════════════════════════════════════════════════════════════════════════
    # orientation_inspector.py logic – triangle orientation
    # ══════════════════════════════════════════════════════════════════════════

    def _estimate_triangle_orientation(self, polygon, up_vec, right_vec, fuse_center):
        """
        Project-and-spread apex detection.
        Narrower half along right_vec = apex side.
        relative_angle ≈ 0° → PASS;  ≈ 180° → FAIL.
        Identical to orientation_inspector.estimate_triangle_orientation().
        """
        crop, ox, oy = self._crop_bbox(polygon)
        if crop.size == 0:
            return None, "UNKNOWN"

        gray = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if len(crop.shape) == 3 else crop.copy())
        thr = self._black_threshold_for_crop(gray)
        _, black_mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

        yy, xx = np.where(black_mask > 0)
        if len(yy) < self.BLACK_MIN_PIXELS:
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

        rel_angle = self._signed_angle_deg(apex_dir, up_vec)
        status    = "PASS" if self._passes(rel_angle, self.THRESHOLD_TRIANGLE) else "FAIL"
        return round(rel_angle, 2), status

    # ══════════════════════════════════════════════════════════════════════════
    # orientation_inspector.py logic – QR preprocessing
    # ══════════════════════════════════════════════════════════════════════════

    def _preprocess_qr_crop(self, crop):
        """4× upscale + CLAHE. Returns (bgr, scale).
        Identical to orientation_inspector._preprocess_qr_crop()."""
        s    = self.QR_PREPROCESS_SCALE
        h, w = crop.shape[:2]
        big  = cv2.resize(crop, (w * s, h * s), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY) if len(big.shape) == 3 else big
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), s

    # ══════════════════════════════════════════════════════════════════════════
    # orientation_inspector.py logic – Stage 1: finder-pattern geometry
    # ══════════════════════════════════════════════════════════════════════════

    def _fp_candidates(self, gray):
        """
        Return (cx, cy, area) for innermost nested-square contours —
        inner cores of QR finder patterns.
        Identical to orientation_inspector._fp_candidates().
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

    @staticmethod
    def _cluster_points(pts, thr):
        """Merge points within thr distance; return ndarray of cluster centres.
        Identical to orientation_inspector._cluster_points()."""
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

    @staticmethod
    def _score_triangle(pts3):
        """High score = one near-90° corner, two roughly equal legs.
        Identical to orientation_inspector._score_triangle()."""
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

    def _corners_finder_pattern(self, crop):
        """
        Estimate QR corners from finder-pattern geometry (no decoder required).
        Returns canonical (4,2) corners [TL, TR, BR, BL] in crop coordinates,
        or None if detection failed.
        Identical to orientation_inspector._corners_finder_pattern().
        """
        preprocessed, scale = self._preprocess_qr_crop(crop)
        gray  = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
        cands = self._fp_candidates(gray)
        if not cands:
            return None

        max_area = max(a for _, _, a in cands)
        cands    = [[x, y] for x, y, a in cands if a >= self.FP_MIN_AREA_RATIO * max_area]
        if len(cands) < 3:
            return None

        cs = self._cluster_points(np.array(cands, dtype=np.float32),
                                   min(preprocessed.shape[:2]) * 0.15)
        if len(cs) < 3:
            return None

        best_score = -1;  best_trio = None;  best_corner = -1
        for combo in itertools.combinations(range(len(cs)), 3):
            pts3 = cs[list(combo)]
            sc, ci = self._score_triangle(pts3)
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

    # ══════════════════════════════════════════════════════════════════════════
    # orientation_inspector.py logic – Stages 2 & 3: decoder chain
    # Methods ordered by corner-label reliability:
    #   pyzbar        – explicit CCW polygon (reordered to canonical below)
    #   zxingcpp      – explicit top_left / top_right / bottom_right / bottom_left
    #   opencv_detect – corner detection only, sometimes correct ordering
    #   opencv_decode – last resort, ordering depends on QR rotation
    # ══════════════════════════════════════════════════════════════════════════

    def _corners_pyzbar(self, crop):
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

    def _corners_zxing(self, crop):
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

    def _corners_opencv_detect(self, crop):
        detector = cv2.QRCodeDetector()
        retval, points = detector.detectMulti(crop)
        if retval and points is not None and len(points) > 0:
            c = np.array(points[0], dtype=np.float32)
            if c.shape == (4, 2):
                return c
        return None

    def _corners_opencv_decode(self, crop):
        detector = cv2.QRCodeDetector()
        retval, _, points, _ = detector.detectAndDecodeMulti(crop)
        if retval and points is not None and len(points) > 0:
            c = np.array(points[0], dtype=np.float32)
            if c.shape == (4, 2):
                return c
        return None

    def _try_decoder_chain(self, crop):
        """Try all decoder methods in reliability order; return first success.
        Identical to orientation_inspector._try_decoder_chain()."""
        for method in (self._corners_pyzbar,
                       self._corners_zxing,
                       self._corners_opencv_detect,
                       self._corners_opencv_decode):
            c = method(crop)
            if c is not None:
                return c
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # orientation_inspector.py logic – angle from corners
    # ══════════════════════════════════════════════════════════════════════════

    def _qr_angle_from_corners(self, corners_raw, up_vec, right_vec):
        """Identical to orientation_inspector._qr_angle_from_corners()."""
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

        rel_angle = self._signed_angle_deg(qr_up_fc, np.array([0.0, 1.0]))
        status    = "PASS" if self._passes(rel_angle, self.THRESHOLD_QR) else "FAIL"
        return round(rel_angle, 2), status

    # ══════════════════════════════════════════════════════════════════════════
    # orientation_inspector.py logic – main QR orientation estimator
    # ══════════════════════════════════════════════════════════════════════════

    def _estimate_qr_orientation(self, polygon, up_vec, right_vec, fuse_center):
        """
        Three-stage QR orientation estimation.
        Identical to orientation_inspector.estimate_qr_orientation()
        with self._img replacing the img parameter.
        """
        pts_bbox = np.array(polygon, dtype=np.int32)
        bx1, by1 = pts_bbox.min(axis=0)
        bx2, by2 = pts_bbox.max(axis=0)
        bx1 = max(0, bx1);  by1 = max(0, by1)
        bx2 = min(self._img.shape[1] - 1, bx2)
        by2 = min(self._img.shape[0] - 1, by2)

        base_crop   = self._img[by1:by2 + 1, bx1:bx2 + 1]
        base_offset = np.array([bx1, by1], dtype=np.float32)

        # Stage 1: finder-pattern
        fp_corners = self._corners_finder_pattern(base_crop)
        if fp_corners is not None:
            corners_raw = fp_corners + base_offset
            rel_angle, status = self._qr_angle_from_corners(corners_raw, up_vec, right_vec)
            if rel_angle is not None:
                ad = self._angular_distance(rel_angle)
                if ad < self.FP_CLEAR_PASS_THRESH or ad > self.FP_CLEAR_FAIL_THRESH:
                    return rel_angle, status   # unambiguous result

        # Stage 2: decoder chain on original crop
        for margin in self.QR_DETECT_MARGINS:
            x1 = max(0, bx1 - margin);  y1 = max(0, by1 - margin)
            x2 = min(self._img.shape[1] - 1, bx2 + margin)
            y2 = min(self._img.shape[0] - 1, by2 + margin)
            crop   = self._img[y1:y2 + 1, x1:x2 + 1]
            offset = np.array([x1, y1], dtype=np.float32)
            c = self._try_decoder_chain(crop)
            if c is not None:
                return self._qr_angle_from_corners(c + offset, up_vec, right_vec)

        # Stage 3: decoder chain on preprocessed crop
        for margin in self.QR_DETECT_MARGINS:
            x1 = max(0, bx1 - margin);  y1 = max(0, by1 - margin)
            x2 = min(self._img.shape[1] - 1, bx2 + margin)
            y2 = min(self._img.shape[0] - 1, by2 + margin)
            crop   = self._img[y1:y2 + 1, x1:x2 + 1]
            offset = np.array([x1, y1], dtype=np.float32)
            preprocessed, scale = self._preprocess_qr_crop(crop)
            c = self._try_decoder_chain(preprocessed)
            if c is not None:
                return self._qr_angle_from_corners(c / scale + offset, up_vec, right_vec)

        return None, "UNKNOWN"

    # ══════════════════════════════════════════════════════════════════════════
    # orientation_inspector.py logic – per-product inspection
    # ══════════════════════════════════════════════════════════════════════════

    def _inspect_product(self, product: dict) -> dict:
        """
        Inspect one grouped product dict.
        Identical to orientation_inspector.inspect_product()
        with self._img replacing the img parameter.
        """
        fuse_poly = self._parse_polygon(product["fuse_cover"])
        qr_parsed = [self._parse_polygon(p) for p in product["qrcode"] if p is not None]
        fuse_center, up_vec, right_vec = self._get_fuse_cover_frame(fuse_poly, qr_parsed)

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
                rel_angle, status = self._estimate_qr_orientation(
                    self._parse_polygon(poly), up_vec, right_vec, fuse_center)
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
                rel_angle, status = self._estimate_triangle_orientation(
                    self._parse_polygon(poly), up_vec, right_vec, fuse_center)
                entry = {"id": i + 1, "status": status}
                if rel_angle is not None:
                    entry["relative_angle"] = rel_angle
                all_statuses.append(status)
            result["yellow_triangle"].append(entry)

        result["overall"]  = self._combine_status(*all_statuses)
        result["failures"] = self._format_failures(result)
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # orientation_inspector.py logic – overall status helper
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _combine_status(*statuses):
        """Identical to orientation_inspector._combine_status()."""
        if "FAIL"    in statuses: return "FAIL"
        if "UNKNOWN" in statuses: return "UNKNOWN"
        return "PASS"

    # ══════════════════════════════════════════════════════════════════════════
    # Output helper (not in original files)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _format_failures(result: dict) -> list:
        """
        Collect non-PASS component entries into readable strings.
        Example: ["qrcode_2: FAIL(+179.4°)", "yellow_triangle_1: UNKNOWN"]
        """
        failures = []
        for key, label in [("qrcode", "qrcode"), ("yellow_triangle", "yellow_triangle")]:
            for entry in result.get(key, []):
                if entry.get("status") == "PASS":
                    continue
                status = entry.get("status", "UNKNOWN")
                angle  = entry.get("relative_angle")
                tag    = (f"{label}_{entry['id']}: {status}({angle:+.1f}°)"
                          if angle is not None
                          else f"{label}_{entry['id']}: {status}")
                failures.append(tag)
        return failures
    

# Example Run
results = StickerOrientationValidator(
    "C:/Users/KUA4SZH/Desktop/inspector_test/IMG_20250630_130952_1.jpg",
    "C:/Users/KUA4SZH/Desktop/inspector_test/IMG_20250630_130952_1.txt"
    ).run()
for r in results:
    print(r["product_id"], r["overall"], r["failures"])

