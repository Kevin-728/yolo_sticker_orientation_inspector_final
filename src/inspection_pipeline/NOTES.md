# Inspection pipeline — two implementations

This folder contains **two** implementations of the same
orientation/presence inspection logic, kept as-is (unmodified) from
development:

1. **Modular version (default / primary):** `main.py` +
   `yolo_converter.py` + `product_grouper.py` +
   `orientation_inspector.py` + `table_formatter.py`. This is the
   version referenced in the top-level README's Quick Start and in
   `SETUP.md`.

2. **Single-file variant:** `single_file_variant/merged_sticker_orientation_validator.py`
   — the same logic merged into one `StickerOrientationValidator` class,
   useful for embedding in another project as a single importable file.
   Its bottom-of-file example run block has a hardcoded local Windows
   path from development and is left untouched; treat it as a usage
   example, not something to run as-is.

Both are kept in the repo since they were both already in active use;
consolidate later if you decide you only need one going forward.
