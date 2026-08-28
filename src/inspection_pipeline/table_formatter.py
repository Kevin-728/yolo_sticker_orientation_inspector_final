# table_formatter.py
"""
Formats inspection results as a plain-ASCII table for terminal output.

One table per image. Each row = one product. Columns:
  id | up_vector | qr_1 | qr_2 | tri_1 | tri_2 | tri_3 | overall

Cell format:
  PASS( -4.7°)   – status with angle
  FAIL(179.1°)   – status with angle
  UNKNOWN        – no angle shown
"""

# Column widths (chars)
_W_ID      =  4
_W_UP      = 16
_W_COMP    = 13   # qr / triangle cells
_W_OVERALL =  7

_SEP = " | "


def _cell_comp(entry):
    """Format one qr/triangle entry dict → fixed-width string."""
    status = entry.get("status", "UNKNOWN")
    angle  = entry.get("relative_angle")
    if status == "UNKNOWN" or angle is None:
        text = "UNKNOWN"
    else:
        text = f"{status}({angle:+.1f}°)"
    return text.ljust(_W_COMP)


def _cell_overall(status):
    return status.ljust(_W_OVERALL)


def _cell_up(up_vector):
    x, y = up_vector
    return f"[{x:+.2f},{y:+.2f}]".ljust(_W_UP)


def _header():
    cols = [
        "id".center(_W_ID),
        "up_vector".ljust(_W_UP),
        "qr_1".ljust(_W_COMP),
        "qr_2".ljust(_W_COMP),
        "tri_1".ljust(_W_COMP),
        "tri_2".ljust(_W_COMP),
        "tri_3".ljust(_W_COMP),
        "overall".ljust(_W_OVERALL),
    ]
    return _SEP.join(cols)


def _divider(header):
    return "-" * len(header)


def _row(product):
    pid    = str(product.get("product_id", "?")).center(_W_ID)
    up     = _cell_up(product["fuse_cover"]["up_vector"])
    qrs    = product.get("qrcode", [])
    tris   = product.get("yellow_triangle", [])

    def get_comp(lst, idx):
        if idx < len(lst):
            return _cell_comp(lst[idx])
        return "N/A".ljust(_W_COMP)

    cols = [
        pid,
        up,
        get_comp(qrs,  0),
        get_comp(qrs,  1),
        get_comp(tris, 0),
        get_comp(tris, 1),
        get_comp(tris, 2),
        _cell_overall(product.get("overall", "UNKNOWN")),
    ]
    return _SEP.join(cols)


def format_results(all_results):
    """
    Parameters
    ----------
    all_results : list of dicts, one per image
        Each dict: { "image": str, "products": [product_dict, ...] }

    Returns
    -------
    str  – formatted table string ready to print
    """
    lines = []
    for image_result in all_results:
        img_path = image_result.get("image", "?")
        products = image_result.get("products", [])
        error    = image_result.get("error")

        lines.append(f"image: {img_path}")

        if error:
            lines.append(f"  ERROR: {error}")
            lines.append("")
            continue

        if not products:
            lines.append("  (no products detected)")
            lines.append("")
            continue

        hdr = _header()
        div = _divider(hdr)
        lines.append(hdr)
        lines.append(div)
        for prod in products:
            lines.append(_row(prod))
        lines.append("")   # blank line between images

    return "\n".join(lines)
