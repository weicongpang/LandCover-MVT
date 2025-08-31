# -*- coding: utf-8 -*-
import json
import os
from typing import Any, Dict, List


INPUT_JSON = "/root/openset/dataset_processed/output_json/renamed_output_images_annotations.json"   # Modify to your own renamed_output_images_annotations.json path    
OUTPUT_DIR = "/root/openset/llama_factory/LLaMA-Factory/data"       
DATASET_FILE = "rs_open_tag_infer.jsonl"      # Generated Alpaca-format dataset

# 是否在 input 中附带 RLE（segmentation.counts）。RLE 很长，默认 False。
INCLUDE_RLE = True

# prompt control (still kept in rules; model must output ONE label per instance)
MIN_SCORE_HINT = 0.30


def load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_instruction() -> str:
    """
    Pixel-level, open-set naming. No label hints are provided.
    Exactly ONE label per instance. Strict JSON-only output.
    """
    return (
        "You are a remote-sensing imagery expert. <image>\n"
        "The user provides an image and a JSON list of pixel-level segmented instances (see user input). "
        "Each instance contains: id, bounding box [x,y,w,h], area, iscrowd, an optional detection score, "
        "and a segmentation mask (RLE with size).\n\n"

        "Task (pixel-level, open-set): For EACH instance, decide EXACTLY ONE semantic label (top-1) that best "
        "describes the object inside the mask region. Treat the segmentation mask (mask==1) as the definitive "
        "region of interest; the bounding box is only a coarse locator. Base your judgment primarily on pixels "
        "inside the mask; surrounding context may be used only as auxiliary evidence. "
        f"If your confidence is below {MIN_SCORE_HINT:.2f} for an instance, output the label \"unknown\".\n\n"

        "Constraints:\n"
        "1) Output for THIS image only.\n"
        "2) The output must contain the SAME NUMBER of segments and the SAME ORDER as the input instances.\n"
        "3) For each segment, return exactly ONE English label (a noun or short noun phrase) and a confidence_score in [0,1].\n"
        "4) Do NOT infer labels from file names, paths, or any meta fields. Do NOT invent or remove segments. Do NOT include any hints or examples.\n"
        "5) Return JSON ONLY and include NO extra keys beyond those specified below.\n\n"

        "Please strictly output in the following format (JSON only, no extra text):\n"
        "{\n"
        '  \"image_id\": <int>,\n'
        '  \"segments\": [\n'
        '    {\"id\": <int>, \"label\": \"<string>\", \"confidence_score\": 0.xx},\n'
        "    ... one entry per input instance, same order ...\n"
        "  ]\n"
        "}\n"
        "Return JSON ONLY."
    )


'''
Output sample:
{
  "image_id": 101,
  "segments": [
    {"id": 599, "label": "aircraft", "score": 0.97},
    {"id": 600, "label": "aircraft", "score": 0.95},
    {"id": 601, "label": "aircraft", "score": 0.93}
  ]
}
'''


def build_input_payload(item: Dict[str, Any]) -> str:
    instances = []
    for ann in item.get("annotations", []):
        one = {
            "id": int(ann.get("id")),
            "bbox_xywh": ann.get("bbox"),     
            "area": ann.get("area"),
            "iscrowd": ann.get("iscrowd", 0),
        }
        if "score" in ann:
            one["score"] = ann["score"]
        
        if INCLUDE_RLE and "segmentation" in ann:
            seg = ann.get("segmentation", {})
            one["segmentation"] = {
                "size": seg.get("size"),
                "counts": seg.get("counts"),
            }
        instances.append(one)

    payload = {
        "image_meta": {
            "image_id": int(item.get("id")),
            "width": item.get("width"),
            "height": item.get("height"),
            "path": item.get("file_path"),
            "instance_count": len(instances)
        },
        "instances": instances,
        "rules": {
            "labels_per_instance": 1,          # exactly one label per instance
            "enforce_one_label": True,
            "preserve_order": True,
            "language": "en",
            "min_score_hint": MIN_SCORE_HINT
        }
    }
    return json.dumps(payload, ensure_ascii=False)


def convert_to_alpaca(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert each per-image record into an Alpaca-style sample."""
    inst = build_instruction()
    outputs: List[Dict[str, Any]] = []
    for it in items:
        sample = {
            "instruction": inst,
            "input": build_input_payload(it),
            "output": "",                # 零样本推理：留空, 如果做 SFT，则要填真实 JSON 标注
            "images": [it["file_path"]]  # exactly 1 image per sample
        }
        outputs.append(sample)
    return outputs


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    items = load_items(INPUT_JSON)
    print(f"[LOAD] Read {len(items)} image records.")
    samples = convert_to_alpaca(items)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, DATASET_FILE)
    save_jsonl(samples, out_path)
    print(f"[SAVE] Already write jsons to: {out_path}")

if __name__ == "__main__":
    main()
