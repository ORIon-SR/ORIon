# ORIon

## Patch Creator

`scripts/patch_creator.py` extracts fixed-size HR patches from APOD images, split into train/val/test sets.

### Expected file structure

```
apod_data/
├── images/          # Raw APOD images (any format PIL can read)
├── splits/          # One CSV per split
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── dataset/         # Generated output (created by the script)
    ├── train/
    │   ├── HR/
    │   └── LR/
    ├── val/
    │   ├── HR/
    │   └── LR/
    └── test/
        ├── HR/
        └── LR/
```

### Split CSV format

Each CSV in `apod_data/splits/` must have a header row and three columns:

```
path,width,height
apod_data/images/2012-11-30.jpg,2000,2000
apod_data/images/2001-06-02.jpg,1607,1595
```

- **path** — relative path from the project root to the image file. Both `/` and `\` separators are accepted.
- **width**, **height** — image dimensions in pixels. Images smaller than the patch size are skipped automatically.

### Usage

```bash
# HR patches only (512x512 default)
python scripts/patch_creator.py

# With LR patches for super-resolution (4x downscale -> 128x128 LR)
python scripts/patch_creator.py --downscale-factor 4

```
