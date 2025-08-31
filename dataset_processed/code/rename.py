# -*- coding: utf-8 -*-
import json
import os
import shutil
from typing import List, Dict, Any


INPUT_JSON = "/root/openset/dataset_processed/output_json/output_images_annotations.json"    # Modify to your own output_images_annotations.json
RENAMED_DIR = "/root/openset/dataset/flat_out_renamed"           # Please create flat_out_renamed directory first in the dataset folder
OUTPUT_JSON = "/root/openset/dataset_processed/output_json/renamed_output_images_annotations.json"   # Modify to your own output path of this file
NAME_PREFIX = "image"   


def load_list_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def digits_for(n: int) -> int:
    """根据数量决定零填充位数，至少 2 位。"""
    if n <= 0:
        return 2
    return max(2, len(str(n - 1)))


def copy_and_rename(items: List[Dict[str, Any]],
                    out_dir: str,
                    prefix: str = "image") -> List[Dict[str, Any]]:
    """
    逐条复制并重命名；保持 items 顺序与一一对应关系。
    返回更新完 file_path 的新列表（其余字段不变）。
    """
    ensure_dir(out_dir)
    n = len(items)
    z = digits_for(n)
    updated: List[Dict[str, Any]] = []

    for i, item in enumerate(items):
        src = item.get("file_path")
        if not src:
            raise ValueError(f"The {i} row lacks file_path tag: {item}")
        if not os.path.exists(src):
            raise FileNotFoundError(f"Could not find source image: {src}")

        _, ext = os.path.splitext(src)
        if not ext:
            ext = ".tif"  # 兜底：若没有扩展名，默认用 .tif

        new_name = f"{prefix}{str(i).zfill(z)}{ext}"
        dst = os.path.join(out_dir, new_name)

        if os.path.exists(dst):
            raise FileExistsError(f"Target exists, avoid being covered: {dst}")

        # 复制并保留元数据
        shutil.copy2(src, dst)

        # 仅更新 file_path，其余内容保持不变（浅拷贝）
        new_item = dict(item)
        new_item["file_path"] = dst
        updated.append(new_item)

        print(f"[{i+1}/{n}] {src}  ->  {dst}")

    return updated


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    items = load_list_json(INPUT_JSON)
    print(f"Already read {len(items)} image records. Copy and renamed to: {RENAMED_DIR}")
    updated = copy_and_rename(items, RENAMED_DIR, NAME_PREFIX)
    save_json(updated, OUTPUT_JSON)
    print(f"[OK] Already written to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
