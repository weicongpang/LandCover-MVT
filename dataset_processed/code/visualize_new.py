# =========================
# 用户配置：原图路径
# =========================
# 请修改成你实际的原图路径，比如:
# IMG_PATH = "/mnt/data/airfield__airplane_001.jpg"
IMG_PATH = "/root/openset/dataset/flat_out/airfield__airplane_001.jpg"   # 默认放在脚本同目录

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# =========================
# 可视化核心函数
# =========================
def map_points_to_image(
    points_src: List[Dict],
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
    scale_limit: Optional[int] = 512,
) -> Tuple[List[Dict], Tuple[int, int], float]:
    w_src, h_src = src_size
    W, H = dst_size

    mapped = []
    for s in points_src:
        x_src, y_src = s["xy"]
        x = float(x_src) * (W / float(w_src))
        y = float(y_src) * (H / float(h_src))
        mapped.append({"xy": (x, y), "score": float(s.get("score", 1.0))})

    scale = 1.0
    if scale_limit is not None and max(W, H) > scale_limit:
        scale = (scale_limit / float(W)) if W >= H else (scale_limit / float(H))

    new_size = (max(1, int(round(W * scale))), max(1, int(round(H * scale))))

    mapped_scaled = []
    for m in mapped:
        x, y = m["xy"]
        mapped_scaled.append({"xy": (x * scale, y * scale), "score": m["score"]})

    return mapped_scaled, new_size, scale


def _light_colormap(v: np.ndarray) -> np.ndarray:
    c1 = np.array([180, 210, 255], dtype=np.float32) / 255.0
    c2 = np.array([255, 200, 220], dtype=np.float32) / 255.0
    return c1 * (1.0 - v[..., None]) + c2 * v[..., None]


def make_soft_overlay(
    disp_W: int,
    disp_H: int,
    points: List[Dict],
    max_alpha: float = 0.45,
) -> np.ndarray:
    yy, xx = np.mgrid[0:disp_H, 0:disp_W]
    heat = np.zeros((disp_H, disp_W), dtype=np.float32)

    sigma = max(disp_W, disp_H) / 18.0
    two_sigma2 = 2.0 * (sigma ** 2)

    for m in points:
        (cx, cy) = m["xy"]
        score = float(m.get("score", 1.0))
        amp = max(0.25, min(1.0, score))
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / two_sigma2) * amp
        heat = np.maximum(heat, g)

    if heat.max() > 0:
        heat_norm = heat / heat.max()
    else:
        heat_norm = heat

    rgba = np.zeros((disp_H, disp_W, 4), dtype=np.float32)
    for y in range(disp_H):
        rgb_row = _light_colormap(heat_norm[y, :])
        rgba[y, :, :3] = rgb_row
    rgba[:, :, 3] = heat_norm * max_alpha
    return rgba


def overlay_rgba_on_image(base_img: Image.Image, overlay_rgba: np.ndarray) -> Image.Image:
    base = np.asarray(base_img).astype(np.float32) / 255.0
    H, W = base.shape[:2]
    ov = overlay_rgba
    alpha = ov[:, :, 3:4]
    color = ov[:, :, :3]
    comp = base * (1.0 - alpha) + color * alpha
    comp_img = (np.clip(comp, 0, 1) * 255.0).astype(np.uint8)
    return Image.fromarray(comp_img)


def draw_points_with_labels(
    pil_img: Image.Image,
    points: List[Dict],
    radius: int = 5,
    fill=(255, 255, 255),
    outline=(0, 0, 0),
) -> Image.Image:
    """
    在图上绘制点与分数标签（无背景框，文字为深色小号字体）。
    """
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)

    # 尝试加载小号字体
    font = None
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, size=7)  # 字体更小
                break
            except Exception:
                font = None

    for i, m in enumerate(points, 1):
        x, y = m["xy"]
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, fill=fill, outline=outline, width=2)

        txt = f"#{i}  score={float(m.get('score', 0.0)):.3f}"
        draw.text((x + 6, y - 8), txt, fill=(0, 0, 0), font=font)  # 深色文字，无背景框

    return img


def visualize_dino_points_on_image(
    img_path: Path,
    seeds: List[Dict],
    src_wh: Tuple[int, int] = (256, 256),
    display_limit: int = 512,
    out_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    if out_dir is None:
        out_dir = Path(".")

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    mapped_scaled, new_size, scale = map_points_to_image(
        points_src=seeds,
        src_size=src_wh,
        dst_size=(W, H),
        scale_limit=display_limit,
    )

    if new_size != (W, H):
        img_disp = img.resize(new_size, Image.LANCZOS)
    else:
        img_disp = img.copy()

    disp_W, disp_H = img_disp.size
    overlay_rgba = make_soft_overlay(disp_W, disp_H, mapped_scaled, max_alpha=0.7)
    comp_img = overlay_rgba_on_image(img_disp, overlay_rgba)
    vis_with_overlay = draw_points_with_labels(comp_img, mapped_scaled, radius=6)
    vis_points_only = draw_points_with_labels(img_disp, mapped_scaled, radius=6)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_overlay = out_dir / "overlay.png"
    out_points = out_dir / "overlay_points_only.png"
    vis_with_overlay.save(out_overlay)
    vis_points_only.save(out_points)

    return out_overlay, out_points


# =========================
# 脚本入口
# =========================
if __name__ == "__main__":
    seeds = [
        {"type": "point", "xy": [76.0, 108.0], "score": 0.5095531344413757},
        {"type": "point", "xy": [52.0, 204.0], "score": 0.4820469915866852},
        {"type": "point", "xy": [188.0, 36.0], "score": 0.4234660267829895},
    ]
    src_w, src_h = 256, 256
    display_limit = 512

    img_path = Path(IMG_PATH)
    if not img_path.exists():
        print("❌ 没找到原图，请检查 IMG_PATH 设置：", IMG_PATH)
    else:
        overlay_path, points_path = visualize_dino_points_on_image(
            img_path=img_path,
            seeds=seeds,
            src_wh=(src_w, src_h),
            display_limit=display_limit,
            out_dir=Path("."),
        )
        print("✅ 可视化完成：")
        print("带浅色分割区域：", overlay_path.resolve())
        print("仅点与标签：  ", points_path.resolve())
