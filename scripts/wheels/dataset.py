"""
数据相关“轮子”模块：封装 DAVIS 数据读取与预测结果规范化的常用函数。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch

_FRAME_EXTENSIONS = [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]


def find_frame_path(rgb_dir: Path, frame_token: str) -> Path:
    """
    在常见后缀中查找指定帧的图像文件。
    """
    for ext in _FRAME_EXTENSIONS:
        candidate = rgb_dir / f"{frame_token}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"序列 {rgb_dir.parent.name} 的帧 {frame_token} 文件不存在。")


def normalize_object_ids(object_ids: Iterable) -> list[int]:
    """
    将预测返回的对象 ID 标准化为 Python int 列表。
    """
    if isinstance(object_ids, torch.Tensor):
        if object_ids.ndim == 0:
            return [int(object_ids.item())]
        return [int(item) for item in object_ids.detach().cpu().tolist()]
    return [int(item) for item in object_ids]


def normalize_masks(masks: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """
    将预测返回的掩码张量标准化为列表形式。
    """
    if isinstance(masks, torch.Tensor):
        if masks.ndim == 2:
            return [masks]
        return [masks[idx] for idx in range(masks.shape[0])]
    return [torch.as_tensor(mask) for mask in masks]


__all__ = ["find_frame_path", "normalize_object_ids", "normalize_masks"]
