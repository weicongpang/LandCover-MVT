#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import json
import random
from pathlib import Path
from typing import List, Dict

# ---- Expert prompt: exactly ONE <image> ----
PROMPT = (
    "<image>\n"
    "You are a senior remote-sensing analyst. Carefully examine the remote-sensing RGB image at multiple scales and infer its numeric label ID (1-30) strictly from visual evidence.\n"
    "Internally consider: global spatial layout and landform; geometry, size, and alignment of man-made structures; texture repetitiveness and granularity; surface/material cues; linear networks (roads, tracks, embankments, shorelines); density and relative scale indicators; color/tonal/spectral contrasts; cast shadows and illumination; contextual boundaries and transitions.\n"
    "Open-set constraint: NEVER state or imply any category or scene name, and do not verbalize your rationale.\n"
    "Output format: return ONLY the integer label in [1, 30] as plain text with no extra words, symbols, or punctuation."
)

INPUT_TXT = (
    "Return one integer in [1, 30]. Do not include any words, labels, or explanations."
)

# safety: exactly one <image>
if len(re.findall(r"<image>", PROMPT)) != 1:
    raise ValueError("PROMPT must contain exactly 1 <image> tag!")

# ---- I/O ----
INDEX_JSON = Path("path/to/aid_label_index.json")  # produced by process_rename.py
OUTPUT_JSON = Path("path/to/0909AID_dataset.jsonl")
RANDOM_SEED  = 20250909  # set None for fully random order

def load_index(index_path: Path) -> List[Dict]:
    if not index_path.exists():
        raise FileNotFoundError(f"Index JSON not found: {index_path}")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
    elif isinstance(data, list):
        samples = data
    else:
        raise ValueError("Index JSON format must be a list or {'samples': [...]}")

    # filter only records with existing image file
    kept = []
    for it in samples:
        fp = it.get("file", "")
        lb = it.get("label", "")
        if not fp or not lb:
            continue
        fp_path = Path(fp)
        resolved_fp = fp_path if fp_path.is_absolute() else (index_path.parent / fp_path)
        if resolved_fp.exists():
            kept.append({"file": str(fp_path), "label": str(lb)})
    if not kept:
        raise RuntimeError("No valid (file, label) pairs found in index.")
    return kept

def count_image_tokens(*texts: str) -> int:
    return sum(len(re.findall(r"<image>", t)) for t in texts if isinstance(t, str))

def build_records(pairs: List[Dict]) -> List[dict]:
    records = []
    for it in pairs:
        img_path = it["file"]
        label    = it["label"]   # "1".."30"
        rec = {
            "instruction": PROMPT,
            "input": INPUT_TXT,
            "output": str(label),
            "images": [img_path],   # exactly one image; must match <image> count (=1)
        }
        tok_cnt = count_image_tokens(rec["instruction"], rec["input"], rec["output"])
        if tok_cnt != len(rec["images"]):
            raise ValueError(
                f"Mismatch: <image> tokens = {tok_cnt}, images = {len(rec['images'])} for file {img_path}"
            )
        records.append(rec)
    return records

def save_json(records: List[dict], out_json: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {len(records)} samples to {out_json}")


def main():
    print(f"[INFO] Loading index: {INDEX_JSON}")
    pairs = load_index(INDEX_JSON)
    print(f"[INFO] Loaded {len(pairs)} (file, label) pairs")

    # build then shuffle
    records = build_records(pairs)
    if RANDOM_SEED is not None:
        random.Random(RANDOM_SEED).shuffle(records)
    else:
        random.shuffle(records)

    # quick sanity check
    for i, r in enumerate(records[:5]):
        tok_cnt = count_image_tokens(r["instruction"], r["input"], r["output"])
        if tok_cnt != len(r["images"]):
            raise AssertionError(f"Post-shuffle <image>:images check failed at idx {i}")

    save_json(records, OUTPUT_JSON)

if __name__ == "__main__":
    main()
