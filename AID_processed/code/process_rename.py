#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import json
import shutil
from pathlib import Path
from typing import List

# ---- I/O paths ----
ORIGINAL_PATH = Path("path/to/source_dataset")             # e.g. AID/<class_name>/*
FLAT_DIR      = Path("path/to/flattened_dataset")          # e.g. dataset_flat/image_00001.jpg
INDEX_JSON    = FLAT_DIR / "aid_label_index.json"

# Acceptable image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Fixed category order -> numeric labels (1..30)
CATEGORY_ORDER = [
    "BareLand","BaseballField","Beach","Bridge","Center","Church","Commercial",
    "DenseResidential","Desert","Farmland","Forest","Industrial","Meadow","MediumResidential",
    "Mountain","Park","Parking","Playground","Pond","Port","RailwayStation","Resort","River",
    "School","SparseResidential","Square","Stadium","StorageTanks","Viaduct"
]

def natsort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def list_images(dir_path: Path) -> List[Path]:
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: natsort_key(p.name))
    return files

def main():
    src = ORIGINAL_PATH
    if not src.exists():
        raise FileNotFoundError(f"Original path not found: {ORIGINAL_PATH}")

    # recreate flat dir
    flat = FLAT_DIR
    if flat.exists():
        print(f"[WARN] {FLAT_DIR} exists. Deleting and recreating ...")
        shutil.rmtree(flat)
    flat.mkdir(parents=True, exist_ok=True)

    # build mapping: class_name -> label_id (1..30) if exists
    mapping = {}
    for i, cname in enumerate(CATEGORY_ORDER, 1):
        cpath = src / cname
        if cpath.exists() and cpath.is_dir():
            mapping[cname] = i
        else:
            print(f"[WARN] Missing class folder: {cname} (label {i})")

    print(f"[INFO] Found {len(mapping)}/{len(CATEGORY_ORDER)} class folders present.")

    # flatten with global running index; also build label index list
    global_idx = 0
    total_copied = 0
    index_list = []  # each: {"file": "<path>", "label": "<1..30>"}

    for cname in CATEGORY_ORDER:
        if cname not in mapping:
            continue
        label_id = mapping[cname]
        imgs = list_images(src / cname)
        print(f"[CLASS] {cname} -> {label_id} | {len(imgs)} files")
        for img in imgs:
            global_idx += 1
            new_name = f"image_{global_idx:05d}{img.suffix.lower()}"
            dst_path = flat / new_name
            shutil.copy2(img, dst_path)
            index_list.append({
                "file": str(dst_path),
                "label": str(label_id),
                "orig_class": cname,
                "orig_name": img.name
            })
            total_copied += 1
            if total_copied % 500 == 0:
                print(f"  copied {total_copied} files ...")

    # write index json
    with open(INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump({"samples": index_list}, f, ensure_ascii=False, indent=2)

    print("\n[DONE] Flatten completed.")
    print(f"  total files: {total_copied}")
    print(f"  flat dir   : {FLAT_DIR}")
    print(f"  index json : {INDEX_JSON}")
    print("\n[Mapping] numeric labels (fixed):")
    for i, cname in enumerate(CATEGORY_ORDER, 1):
        print(f"  {i}: {cname}")

if __name__ == "__main__":
    main()
