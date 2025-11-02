"""
可视化工具：为评估结果生成可读图像。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image

__all__ = [
    "overlay_mask",
    "stack_overlays",
    "save_overlay",
]


def _to_image(image: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGBA")
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[2] == 3:
        alpha = np.full((image.shape[0], image.shape[1], 1), 255, dtype=image.dtype)
        image = np.concatenate([image, alpha], axis=2)
    return Image.fromarray(image.astype(np.uint8), mode="RGBA")


def _mask_to_rgba(mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> Image.Image:
    mask_bool = mask.astype(bool)
    overlay = np.zeros((*mask_bool.shape, 4), dtype=np.uint8)
    overlay[..., :3] = color
    overlay[..., 3] = (mask_bool * int(alpha * 255)).astype(np.uint8)
    return Image.fromarray(overlay, mode="RGBA")


def overlay_mask(
    image: np.ndarray | Image.Image,
    mask: np.ndarray,
    *,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5,
) -> Image.Image:
    """
    将掩码半透明叠加在原图上。
    """
    base = _to_image(image)
    overlay = _mask_to_rgba(mask, color=color, alpha=alpha)
    return Image.alpha_composite(base, overlay)


def stack_overlays(overlays: Iterable[Image.Image]) -> Image.Image:
    """
    将多张可视化结果垂直拼接，方便对比。
    """
    overlay_list = list(overlays)
    if not overlay_list:
        raise ValueError("没有可拼接的图像。")
    widths = [img.width for img in overlay_list]
    heights = [img.height for img in overlay_list]
    canvas = Image.new("RGBA", (max(widths), sum(heights)), (0, 0, 0, 0))
    y_offset = 0
    for img in overlay_list:
        canvas.paste(img, (0, y_offset))
        y_offset += img.height
    return canvas


def save_overlay(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGBA").save(path)
