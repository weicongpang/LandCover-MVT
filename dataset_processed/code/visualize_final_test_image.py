#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
from typing import Optional, Tuple, List

from PIL import Image, ImageDraw, ImageFont

# -------------------- Hard-coded paths --------------------
DATASET_DIR = "/root/openset/dataset_eval/Test_processed"
JSONL_PATH  = "/root/openset/llama_factory/LLaMA-Factory/outputs/no-finetune-pixtral_test_2025-09-14/generated_predictions.jsonl"
OUTPUT_DIR  = "/root/openset/dataset_eval/no-finetune-pixtral-result"

# -------------------- Regex helpers (robust to commas, etc.) --------------------
IMG_PATH_REGEXES = [
    re.compile(r"\((/[^)]+\.(?:png|jpg|jpeg|bmp|gif|tif|tiff))\)", re.IGNORECASE),
    re.compile(r"image\s*[:=]\s*(/[^,\s]+\.(?:png|jpg|jpeg|bmp|gif|tif|tiff))", re.IGNORECASE),
]

LEVEL1_REGEXES = [
    re.compile(r"Level-1\s*category\s*[: ]\s*(.+?)(?=\.|\n|$)", re.IGNORECASE),
    re.compile(r"belongs to\s+Level-1\s*category\s+(.+?)(?=\.|,|\n|$)", re.IGNORECASE),
]

LEVEL2_REGEXES = [
    re.compile(r"Level-2\s*subclass\s*[: ]\s*(.+?)(?=\.|\n|$)", re.IGNORECASE),
    re.compile(r"falls under\s+Level-2\s*subclass\s+(.+?)(?=\.|,|\n|$)", re.IGNORECASE),
]

DESC_REGEXES = [
    re.compile(r"(?:The\s+reason\s+for\s+this\s+classification\s+is\s+as\s+follows:)\s*(.*)$",
               re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:Reason\s*:)\s*(.*)$", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:because\s*:?)\s*(.*)$", re.IGNORECASE | re.DOTALL),
]

def extract_first(patterns: List[re.Pattern], text: str) -> Optional[str]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    return None

def clean_markdown_spans(s: str) -> str:
    if not s:
        return s
    # 去掉代码块围栏
    s = re.sub(r"^```[\s\S]*?\n", "", s.strip())
    s = re.sub(r"```$", "", s.strip())
    # 去掉外层引号
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()
    # 去掉加粗/反引号/方括号包裹
    s = s.replace("**", "").replace("`", "").strip()
    s = re.sub(r"^\[(.+)\]$", r"\1", s)
    return s

def parse_description(raw: str) -> str:
    # 1) 常见格式：Reason...: <desc>
    for pat in DESC_REGEXES:
        m = pat.search(raw)
        if m:
            desc = m.group(1).strip()
            break
    else:
        # 2) 退路：截取二级类别句子后的剩余文本
        m = re.search(
            r"Specifically,\s*it\s+falls\s+under\s+Level-2\s*subclass\s+.+?\.?\s*(.*)$",
            raw,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            desc = m.group(1).strip()
        else:
            # 3) 再退路：从 “The image shows ...” 开始
            m = re.search(
                r"(The\s+image\s+shows\b.*)$",
                raw,
                re.IGNORECASE | re.DOTALL,
            )
            desc = m.group(1).strip() if m else ""

    desc = re.sub(r"\s+", " ", desc).strip()
    desc = clean_markdown_spans(desc)
    return desc

def parse_raw_prediction(raw: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """
    Returns (img_path, level1, level2, description)
    """
    raw = clean_markdown_spans(raw)

    img_path = extract_first(IMG_PATH_REGEXES, raw)
    level1   = clean_markdown_spans(extract_first(LEVEL1_REGEXES, raw) or "")
    level2   = clean_markdown_spans(extract_first(LEVEL2_REGEXES, raw) or "")
    desc     = parse_description(raw)

    level1 = level1 or None
    level2 = level2 or None
    return img_path, level1, level2, desc

# -------------------- Imaging helpers --------------------
def try_load_font(font_size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                pass
    return ImageFont.load_default()

def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    cur = words[0]
    for w in words[1:]:
        trial = cur + " " + w
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines

def lines_height(draw: ImageDraw.ImageDraw, lines: List[str], font, line_spacing: int) -> int:
    if not lines:
        return 0
    h_sum = 0
    for t in lines:
        bbox = draw.textbbox((0,0), t, font=font)
        h_sum += (bbox[3] - bbox[1]) + line_spacing
    return h_sum - line_spacing

def annotate_image(img_path: str, level1: str, level2: str, desc: str, out_path: str) -> None:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        W, H = im.size

        scale_factor = 0.9
        new_W, new_H = int(W * scale_factor), int(H * scale_factor)
        im_resized = im.resize((new_W, new_H), Image.LANCZOS)

        base_size = max(14, min(40, int(W * 0.025)))
        font_h = try_load_font(base_size)

        desc_size = max(10, int(base_size * 0.75))
        font_d = try_load_font(desc_size)

        left = max(20, int(W * 0.05))
        right = left
        top_pad = max(10, int(base_size * 0.3))
        mid_pad = max(6, int(base_size * 0.3))
        bottom_pad = max(12, int(base_size * 0.4))
        line_space_h = max(4, int(base_size * 0.15))
        def line_space_d(sz): return max(3, int(sz * 0.15))
        max_text_w = W - left - right

        tmp = ImageDraw.Draw(im_resized)

        l12_text = f"Level-1: {level1}, Level-2: {level2}"
        l12 = wrap_text(tmp, l12_text, font_h, max_text_w)
        h12 = lines_height(tmp, l12, font_h, line_space_h)

        max_footer_h = H - new_H + int(H * 0.25)
        while True:
            font_d = try_load_font(desc_size)
            l3 = wrap_text(tmp, f"Description: {desc}", font_d, max_text_w)
            h3 = lines_height(tmp, l3, font_d, line_space_d(desc_size))
            footer_h = top_pad + h12 + mid_pad + h3 + bottom_pad
            if footer_h <= max_footer_h or desc_size <= 8:
                break
            desc_size -= 1

        total_H = new_H + footer_h
        out = Image.new("RGB", (W, total_H), "white")

        img_x = (W - new_W) // 2
        out.paste(im_resized, (img_x, 0))
        draw = ImageDraw.Draw(out)
        sep_th = max(1, W // 800)
        draw.line([(0, new_H), (W, new_H)], fill=(200,200,200), width=sep_th)

        def draw_lines(lines, y, font, ls):
            for t in lines:
                draw.text((left, y), t, font=font, fill="black")
                y = draw.textbbox((left, y), t, font=font)[3] + ls
            return y

        y0 = new_H + top_pad
        y0 = draw_lines(l12, y0, font_h, line_space_h)
        y0 += (mid_pad - line_space_h)
        draw_lines(l3, y0, font_d, line_space_d(desc_size))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_path = os.path.splitext(out_path)[0] + ".jpg"
        out.save(out_path, format="JPEG", quality=95)

# -------------------- Main pipeline --------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(JSONL_PATH):
        raise FileNotFoundError(f"JSONL file not found: {JSONL_PATH}")

    # Count non-empty lines for progress bar upper bound (minimal logs)
    total_lines = 0
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total_lines += 1

    processed_index = 0  # 仅用于展示“Processing i/N”
    saved = 0

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            raw_line = line.strip()
            if not raw_line:
                continue

            processed_index += 1
            print(f"Processing {processed_index}/{total_lines}", flush=True)

            # 解析 JSONL
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            # 只要 predict，不要 prompt；同时兼容旧键名
            raw = obj.get("predict") or obj.get("raw_prediction") or obj.get("text") or obj.get("output")
            if not raw or not isinstance(raw, str):
                continue
            raw = clean_markdown_spans(raw)

            img_path, level1, level2, desc = parse_raw_prediction(raw)
            if not (level1 and level2):
                continue

            # Resolve image path（优先绝对路径，其次 DATASET_DIR/basename）
            candidate_paths = []
            if img_path and os.path.exists(img_path):
                candidate_paths.append(img_path)
            if img_path:
                candidate_paths.append(os.path.join(DATASET_DIR, os.path.basename(img_path)))
            else:
                m = re.search(r"(\d+\.(?:png|jpg|jpeg|bmp|gif|tif|tiff))", raw, re.IGNORECASE)
                if m:
                    candidate_paths.append(os.path.join(DATASET_DIR, m.group(1)))

            resolved = None
            for p in candidate_paths:
                if p and os.path.exists(p):
                    resolved = p
                    break
            if not resolved:
                continue

            out_path = os.path.join(OUTPUT_DIR, os.path.basename(resolved))
            try:
                annotate_image(resolved, level1, level2, desc, out_path)
                saved += 1
            except Exception:
                continue

    print(f"Done. Saved {saved} images to {OUTPUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
