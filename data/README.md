# Data

## Layout

```
data/
  raw_dataset/              # flat folder: image001.jpg, image001.txt, ...
  split_dataset/            # output of 1_split_dataset.py
    images/{train,val}/
    labels/{train,val}/
  tiled_dataset/             # output of 3_tile_dataset.py (small-object model)
  large_object_dataset/      # output of 3b_prepare_large_dataset.py (large-object model)
```

Class IDs (fixed across the whole pipeline):

```
0 = qrcode
1 = yellow_triangle
2 = green_label
3 = fuse_cover
```

See [`../src/yolo_pipeline/README.md`](../src/yolo_pipeline/README.md)
for the exact commands that build each of these from raw labeled images.


