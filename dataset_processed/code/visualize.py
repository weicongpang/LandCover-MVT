# -*- coding: utf-8 -*-
import os
import json
from typing import Any, Dict, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ======== 可按需修改 ========
INPUT_JSON = "/root/openset/dataset_processed/sample_json/final_annotations.json"
OUTPUT_DIR = "/root/openset/dataset_processed/visualize_folder"
ALPHA = 0.55                   # 透明度↑ 更“深”
DRAW_ID = True                 # 是否在实例中心写 id
BASE_FONT_SIZE = 9             # 目标字号（DejaVu/Arial 找不到则回退默认字号）
TEXT_FILL = (255, 255, 255)    # 文字颜色
TEXT_STROKE = (0, 0, 0)        # 1px 黑色描边，增强可读
# ============================

# 更深的配色（饱和）
DEEP_COLORS = [
    (220, 20, 60),    # Crimson
    (65, 105, 225),   # RoyalBlue
    (46, 139, 87),    # SeaGreen
    (255, 140, 0),    # DarkOrange
    (138, 43, 226),   # BlueViolet
    (218, 165, 32),   # Goldenrod
    (0, 191, 255),    # DeepSkyBlue
    (244, 164, 96),   # SandyBrown
]

# 尝试导入 pycocotools
try:
    from pycocotools import mask as maskUtils
except Exception as e:
    raise RuntimeError("缺少 pycocotools，请先：pip install pycocotools\n" + str(e))


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_rle_to_mask(seg: Dict[str, Any]) -> np.ndarray:
    size = seg.get("size")
    counts = seg.get("counts")
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    rle = {"size": size, "counts": counts}
    m = maskUtils.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def overlay_mask_on_image(img: Image.Image, mask: np.ndarray, color: tuple, alpha: float) -> Image.Image:
    img_np = np.array(img).astype(np.float32)
    h, w = mask.shape
    if img_np.shape[0] != h or img_np.shape[1] != w:
        mask = np.array(Image.fromarray(mask).resize((img_np.shape[1], img_np.shape[0]), Image.NEAREST))

    color_np = np.zeros_like(img_np)
    color_np[:, :, 0] = color[0]
    color_np[:, :, 1] = color[1]
    color_np[:, :, 2] = color[2]

    m3 = np.repeat(mask[:, :, None], 3, axis=2)
    img_np[m3 == 1] = img_np[m3 == 1] * (1 - alpha) + color_np[m3 == 1] * alpha
    return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8), mode="RGB")


def _load_font(target_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """优先加载可控字号的 TrueType 字体；缺失时退回默认位图字体。"""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=target_size)
            except Exception:
                pass
    return ImageFont.load_default()  # 退回默认（字号不可控，通常≈10px）


def draw_segment_id(img: Image.Image, mask: np.ndarray, seg_id: int) -> Image.Image:
    if not DRAW_ID:
        return img
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return img
    cx, cy = int(xs.mean()), int(ys.mean())

    # 动态字号：在小图上保持更小；在大图上略放大
    img_min_side = min(img.width, img.height)
    fs = max(8, min(14, int(img_min_side / 32)))  # 256px -> 8，512px -> 16(被限制到14)
    fs = max(fs, BASE_FONT_SIZE)

    font = _load_font(fs)
    draw = ImageDraw.Draw(img)

    # 仅文字，无白底；用 1px 描边保证可读
    draw.text(
        (cx, cy),
        str(seg_id),
        fill=TEXT_FILL,
        font=font,
        anchor="mm",              # 以文本中心对齐到质心
        stroke_width=1,
        stroke_fill=TEXT_STROKE,
    )
    return img


def visualize_item(item: Dict[str, Any], out_dir: str, alpha: float = ALPHA) -> str:
    img_path = item["file_path"]
    img = Image.open(img_path).convert("RGB")
    anns = item.get("annotations", [])

    for k, ann in enumerate(anns):
        seg = ann.get("segmentation")
        if not seg or "counts" not in seg or "size" not in seg:
            continue
        mask = decode_rle_to_mask(seg)
        color = DEEP_COLORS[k % len(DEEP_COLORS)]
        img = overlay_mask_on_image(img, mask, color=color, alpha=alpha)
        img = draw_segment_id(img, mask, seg_id=int(ann.get("id", k + 1)))

    base = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(out_dir, f"{base}_vis.jpg")
    img.save(save_path, format="JPEG", quality=95)
    return save_path


def main():
    ensure_dir(OUTPUT_DIR)
    items = load_items(INPUT_JSON)
    print(f"[LOAD] {len(items)} items")
    saved = []
    for item in items:
        try:
            p = visualize_item(item, OUTPUT_DIR, alpha=ALPHA)
            saved.append(p)
            print(f"[OK] saved: {p}")
        except Exception as e:
            print(f"[WARN] image_id={item.get('id')}: {e}")
    print(f"\n[SUMMARY] {len(saved)} images saved -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
