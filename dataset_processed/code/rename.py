# -*- coding: utf-8 -*-
import json
import os
import re
import shutil
from typing import List, Dict, Any
from collections import defaultdict

INPUT_JSON = "/root/openset/dataset_processed/output_json/output_images_annotations.json"
RENAMED_FINAL_DIR = "/root/openset/dataset/renamed_final"
OUTPUT_JSON = "/root/openset/dataset_processed/output_json/renamed_output_images_annotation.json"
IMAGE_CATEGORY_MAPPING_JSON = os.path.join(RENAMED_FINAL_DIR, "image_category_mapping.json")

CATEGORY_PREFIX = "category"
IMAGE_PREFIX = "image"

SUFFIX_WHITELIST = {
    "sand_beach", "city_building", "circular_farmland",
    "dry_farm", "green_farmland", "rectangular_farmland",
    "sparse_forest", "ground_track_field", "stadium",
    "tennis_court", "tenniscourt", "airport_runway",
}

ALIAS_MAP = {
    "tenniscourt": "tennis_court",
    "groundtrackfield": "ground_track_field",
    "sandbeach": "sand_beach",
    "citybuilding": "city_building",
    "green_farm_land": "green_farmland",
    "rectangular_farm_land": "rectangular_farmland",
    "circular_farm_land": "circular_farmland",
}

def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def normalize_label(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[ \t\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def cleanup_suffix(s: str) -> str:
    s = re.sub(r"([_-]?\d+)$", "", s).rstrip("_-. ")
    return s

def parse_category(file_path: str) -> str:
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    if "__" not in name:
        raise ValueError(f"No '__' in filename: {base}")
    prefix, raw_suffix = name.split("__", 1)
    cleaned_suffix = cleanup_suffix(raw_suffix)
    norm_suffix = normalize_label(cleaned_suffix)
    norm_suffix = ALIAS_MAP.get(norm_suffix, norm_suffix)
    return norm_suffix if norm_suffix in SUFFIX_WHITELIST else prefix

def digits(n: int, min_width: int = 6) -> int:
    return max(min_width, len(str(n)))

def copy_file(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

def should_skip(file_path: str) -> bool:
    return bool(re.search(r"airfield|airplane|airport", file_path, re.IGNORECASE))

def main():
    items = load_json(INPUT_JSON)
    category_to_items = defaultdict(list)

    for item in items:
        if should_skip(item["file_path"]):
            continue
        category = parse_category(item["file_path"])
        category_to_items[category].append(item)

    categories_sorted = sorted(category_to_items.keys())
    category_id_map = {cat: f"{CATEGORY_PREFIX}{str(idx).zfill(4)}"
                       for idx, cat in enumerate(categories_sorted, start=1)}

    total_images = sum(len(v) for v in category_to_items.values())
    img_digits = digits(total_images)

    mapping_json = []
    updated_items = []
    image_counter = 1
    category_counts = defaultdict(int)

    os.makedirs(RENAMED_FINAL_DIR, exist_ok=True)

    for cat in categories_sorted:
        cat_id = category_id_map[cat]
        cat_items = category_to_items[cat]
        for item in cat_items:
            ext = os.path.splitext(item["file_path"])[-1] or ".tif"
            new_filename = f"{IMAGE_PREFIX}{str(image_counter).zfill(img_digits)}{ext}"
            new_filepath = os.path.join(RENAMED_FINAL_DIR, new_filename)

            copy_file(item["file_path"], new_filepath)

            mapping_json.append({
                "new_filename": new_filename,
                "original_path": item["file_path"],
                "new_path": new_filepath,
                "category": cat_id
            })

            item_updated = dict(item)
            item_updated["file_path"] = new_filepath
            updated_items.append(item_updated)

            image_counter += 1
            category_counts[cat_id] += 1

    save_json(mapping_json, IMAGE_CATEGORY_MAPPING_JSON)
    save_json(updated_items, OUTPUT_JSON)

    print("=== Processing Summary ===")
    for cat in categories_sorted:
        cat_id = category_id_map[cat]
        count = category_counts[cat_id]
        print(f"{cat_id} ({cat}): {count} images")

    print(f"\nTotal images processed: {total_images}")
    print(f"Unified images stored at: {RENAMED_FINAL_DIR}")
    print(f"Detailed mapping JSON saved at: {IMAGE_CATEGORY_MAPPING_JSON}")
    print(f"Updated annotations JSON saved at: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
