# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Dict, List, Any

INPUT_JSON = Path("path/to/instance_object_only.json")  # update to your input JSON path
IMG_BASE_DIR = Path("path/to/flat_out")                 # update to your flattened image directory
OUTPUT_JSON = Path("path/to/output_images_annotations.json")  # update to your desired output file path


def load_input_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_ann_index(annotations: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Cluster annotations by image_id for easy merging."""
    idx: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        img_id = int(ann.get("image_id"))
        idx.setdefault(img_id, []).append(ann)
    return idx


def to_file_path(file_name: str, base_dir: Path) -> str:
    """Convert file_name to full file_path by joining with base_dir."""
    return str(base_dir / file_name)


def merge_images_and_annotations(data: Dict[str, Any], base_dir: Path) -> List[Dict[str, Any]]:
    """核心合并：对每个 image, 挂上它的 annotations, 并把 file_name -> file_path。"""
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    ann_index = build_ann_index(annotations)

    merged: List[Dict[str, Any]] = []
    for img in images:
        img_id = int(img["id"])
        # 生成 file_path，保留其它图像元数据
        item = {
            "id": img_id,
            "file_path": to_file_path(img["file_name"], base_dir),
            "width": img.get("width"),
            "height": img.get("height"),
            "annotations": ann_index.get(img_id, [])
        }
        merged.append(item)
        print(f"Merged image {img_id} with {len(item['annotations'])} annotations.")
    return merged


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    data = load_input_json(INPUT_JSON)
    merged = merge_images_and_annotations(data, IMG_BASE_DIR)
    save_json(merged, OUTPUT_JSON)
    total_anns = sum(len(x["annotations"]) for x in merged)
    print(f"[OK] Combined Confirmed: {len(merged)} images, {total_anns} annotations -> {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
