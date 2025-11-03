"""
根据 GT 掩码生成 SAM2 推理所需的 prompt。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PromptBundle",
    "mask_to_point_prompt",
    "mask_to_box_prompt",
    "build_prompt_bundle",
]


@dataclass(slots=True)
class PromptBundle:
    """
    SAM2 交互式推理所需的 prompt 组合。
    """

    points: np.ndarray
    point_labels: np.ndarray
    boxes: np.ndarray | None = None
    mask_input: np.ndarray | None = None

    def as_dict(self) -> dict[str, np.ndarray | None]:
        return {
            "points": self.points,
            "point_labels": self.point_labels,
            "boxes": self.boxes,
            "mask_inputs": self.mask_input,
        }


def _ensure_bool_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.bool_:
        return mask.astype(bool)
    return mask


def mask_to_point_prompt(mask: np.ndarray) -> np.ndarray:
    """
    计算掩码的几何中心作为正样本点。
    """
    mask = _ensure_bool_mask(mask)
    if mask.sum() == 0:
        raise ValueError("GT 掩码为空，无法生成前景点。")
    indices = np.argwhere(mask)
    yx = indices.mean(axis=0)
    # 顺序为 (x, y)
    return np.array([[yx[1], yx[0]]], dtype=np.float32)


def mask_to_box_prompt(mask: np.ndarray) -> np.ndarray:
    """
    生成掩码的包围盒，用于 box prompt。
    返回 (x_min, y_min, x_max, y_max)。
    """
    mask = _ensure_bool_mask(mask)
    if mask.sum() == 0:
        raise ValueError("GT 掩码为空，无法生成包围盒。")
    indices = np.argwhere(mask)
    y_min, x_min = indices.min(axis=0)
    y_max, x_max = indices.max(axis=0)
    return np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)


def _sample_background_points(
    mask: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    mask = _ensure_bool_mask(mask)
    inverse = np.logical_not(mask)
    candidates = np.argwhere(inverse)
    if candidates.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    if count >= candidates.shape[0]:
        sampled = candidates
    else:
        indices = rng.choice(candidates.shape[0], size=count, replace=False)
        sampled = candidates[indices]
    return np.stack([sampled[:, 1], sampled[:, 0]], axis=1).astype(np.float32)


def build_prompt_bundle(
    mask: np.ndarray,
    *,
    include_points: bool = True,
    include_box: bool = True,
    background_points: int = 0,
    rng: np.random.Generator | None = None,
    mask_input: np.ndarray | None = None,
) -> PromptBundle:
    """
    组合点、框和可选的 mask 提示，便于直接输送至 SAM2。
    """
    rng = rng or np.random.default_rng()
    if include_points:
        pos_points = mask_to_point_prompt(mask)
        point_labels = np.ones(pos_points.shape[0], dtype=np.int32)

        neg_points = _sample_background_points(mask, background_points, rng)
        if neg_points.size > 0:
            points = np.concatenate([pos_points, neg_points], axis=0)
            neg_labels = np.zeros(neg_points.shape[0], dtype=np.int32)
            point_labels = np.concatenate([point_labels, neg_labels], axis=0)
        else:
            points = pos_points
    else:
        points = np.empty((0, 2), dtype=np.float32)
        point_labels = np.empty((0,), dtype=np.int32)

    boxes = mask_to_box_prompt(mask) if include_box else None
    return PromptBundle(points=points, point_labels=point_labels, boxes=boxes, mask_input=mask_input)
