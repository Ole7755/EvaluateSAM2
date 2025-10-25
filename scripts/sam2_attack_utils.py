"""
通用的 SAM2 攻击辅助函数与日志工具。

该模块负责：
1. 加载与保存图像 / 掩码张量；
2. 组织实验输出目录；
3. 计算常用的分割指标；
4. 记录攻击配置与结果，便于后续复现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import shutil

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

from .evaluate_sam2_metrics import compute_iou_and_dice

# ImageNet 归一化常量（与 SAM2 训练流程保持一致）
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def ensure_dir(path: Path) -> Path:
    """确保目录存在，并返回该路径。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_rgb_tensor(image_path: Path, device: torch.device, normalize: bool = True) -> torch.Tensor:
    """
    将磁盘上的 RGB 图像读取为张量。

    参数：
        image_path: 图像路径。
        device: 目标设备（cpu 或 cuda）。
        normalize: 是否归一化到 [0, 1] 区间。

    返回：
        shape 为 (3, H, W) 的 torch.Tensor。
    """
    image = Image.open(image_path).convert("RGB")
    array = np.array(image, dtype=np.float32)
    if normalize:
        array /= 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(device)
    return tensor.contiguous()


def save_rgb_tensor(tensor: torch.Tensor, save_path: Path) -> None:
    """
    将张量保存为 PNG 图像。

    注意：
        - 函数会自动将值裁剪到 [0, 1] 区间；
        - 输入张量 shape 应为 (3, H, W)。
    """
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    array = (tensor.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(save_path)


def save_perturbation_image(tensor: torch.Tensor, save_path: Path) -> None:
    """
    将扰动张量可视化后保存为 PNG。

    采用零点居中显示，将最大绝对值映射为 1。
    """
    if tensor.ndim == 4:
        if tensor.size(0) != 1:
            raise ValueError("暂不支持批量扰动可视化。")
        tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu()
    max_abs = float(tensor.abs().max().item())
    if max_abs <= 1e-8:
        max_abs = 1.0
    scaled = (tensor / (2 * max_abs)) + 0.5
    scaled = scaled.clamp(0.0, 1.0)
    array = (scaled.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(save_path)


def load_mask_tensor(mask_path: Path, device: torch.device) -> torch.Tensor:
    """
    加载首帧掩码（支持多实例标签）。

    返回：
        torch.Tensor，dtype=float32，shape=(H, W)。
    """
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return torch.from_numpy(mask.astype(np.float32)).to(device)


def mask_to_binary(mask_tensor: torch.Tensor, label: Optional[int] = None, threshold: float = 0.0) -> torch.Tensor:
    """
    将原始掩码转换为二值张量。

    参数：
        mask_tensor: 原始掩码张量。
        label: 若提供，则仅保留对应标签。
        threshold: 当 label 未指定时，以阈值划分前景和背景。
    """
    if label is not None:
        binary = (mask_tensor == float(label)).to(mask_tensor.dtype)
    else:
        binary = (mask_tensor > threshold).to(mask_tensor.dtype)
    return binary


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    计算 Dice Loss（1 - Dice 系数）。

    参数：
        pred: 预测概率，范围建议在 [0, 1]。
        target: 目标二值掩码。
        eps: 防止分母为 0 的微小常数。
    """
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return torch.clamp(1.0 - dice, min=0.0, max=1.0)


def bce_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """对掩码概率与目标之间计算二值交叉熵。"""
    pred = torch.clamp(pred, eps, 1.0 - eps)
    loss = -(target * pred.log() + (1 - target) * (1 - pred).log())
    return loss.mean()


def eval_masks_numpy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    """
    对 numpy 掩码计算 mIoU 与 Dice。

    参数：
        pred_mask: 二值预测。
        gt_mask: 二值真值。
    """
    return compute_iou_and_dice(pred_mask.astype(bool), gt_mask.astype(bool))


def resize_image_tensor(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """
    将图像张量缩放到 (size, size) 的分辨率。

    注意：
        - 输入 tensor 形状应为 (C, H, W) 或 (B, C, H, W)；
        - 使用双线性插值，保持像素值的平滑性。
    """
    need_squeeze = False
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        need_squeeze = True
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    if need_squeeze:
        resized = resized.squeeze(0)
    return resized


def resize_mask_tensor(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """
    将掩码张量缩放到 (size, size) 并保持二值性。

    采用最近邻插值以避免引入灰度值。
    """
    need_squeeze = False
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        need_squeeze = True
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    resized = F.interpolate(tensor, size=(size, size), mode="nearest")
    if need_squeeze:
        resized = resized.squeeze(0).squeeze(0)
    return resized


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    将经过 ImageNet 归一化的张量还原到 [0, 1] 范围。

    用于保存攻前 / 攻后图像。
    """
    mean = IMAGENET_MEAN.to(tensor.device)
    std = IMAGENET_STD.to(tensor.device)
    tensor = tensor * std + mean
    return tensor.clamp(0.0, 1.0)


def compute_perturbation_norms(perturbation: torch.Tensor) -> Dict[str, float]:
    """
    计算扰动张量的常见范数，用于日志记录。

    返回：
        - l2: L2 范数
        - linf: L_inf 范数
        - l1: L1 范数
    """
    flat = perturbation.view(perturbation.size(0), -1)
    norms = {
        "l2": float(torch.linalg.vector_norm(flat, ord=2, dim=1).mean().item()),
        "linf": float(torch.linalg.vector_norm(flat, ord=float("inf"), dim=1).mean().item()),
        "l1": float(torch.linalg.vector_norm(flat, ord=1, dim=1).mean().item()),
    }
    return norms


@dataclass
class AttackConfig:
    """记录攻击的核心超参数，用于写入日志。"""

    attack_name: str
    epsilon: float
    step_size: float
    steps: int
    random_start: bool = False
    cw_confidence: Optional[float] = None
    cw_learning_rate: Optional[float] = None
    cw_binary_steps: Optional[int] = None


@dataclass
class AttackSummary:
    """记录一次攻击的概览信息。"""

    attack_name: str
    sequence: str
    frame_idx: int
    obj_id: int
    gt_label: Optional[int]
    clean_iou: float
    clean_dice: float
    adv_iou: float
    adv_dice: float
    perturbation_norm: Dict[str, float]


class AttackLogger:
    """
    简易日志工具：
    - 将配置与结果写入 JSON；
    - 保存 UAP 张量为 .pt，方便复现。
    """

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = ensure_dir(log_dir)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def save_config(self, config: AttackConfig) -> Path:
        """将攻击配置保存到日志目录中。"""
        path = self.log_dir / f"{config.attack_name}_config_{self._timestamp()}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(config), f, ensure_ascii=False, indent=2)
        return path

    def save_summary(self, summary: AttackSummary, extra: Optional[Dict[str, Any]] = None) -> Path:
        """保存攻击结果概要。"""
        payload: Dict[str, Any] = asdict(summary)
        if extra:
            payload.update(extra)
        path = self.log_dir / f"{summary.sequence}_{summary.obj_id}_{summary.frame_idx}_{self._timestamp()}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def save_tensor(self, tensor: torch.Tensor, name: str) -> Path:
        """保存张量（例如通用扰动）为 .pt 文件。"""
        path = self.log_dir / f"{name}_{self._timestamp()}.pt"
        torch.save(tensor.detach().cpu(), path)
        return path


class BestWorstTracker:
    """维护同一攻击类型下最佳 / 最差案例，并保存相关图像。"""

    def __init__(
        self,
        record_path: Path,
        best_dir: Path,
        worst_dir: Path,
        min_clean_iou: float = 0.5,
    ) -> None:
        self.record_path = record_path
        self.best_root = ensure_dir(best_dir)
        self.worst_root = ensure_dir(worst_dir)
        self.state = self._load_state()
        self.min_clean_iou = float(min_clean_iou)

    def _load_state(self) -> Dict[str, Dict[str, Any]]:
        if self.record_path.exists():
            with self.record_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_state(self) -> None:
        ensure_dir(self.record_path.parent)
        with self.record_path.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    def _remove_artifacts(self, artifacts: Dict[str, str]) -> None:
        for path_str in artifacts.values():
            path = Path(path_str)
            if path.exists():
                path.unlink()

    def _format_prefix(self, summary: AttackSummary) -> str:
        return (
            f"{summary.attack_name}_"
            f"{summary.sequence}_"
            f"frame{summary.frame_idx:05d}_"
            f"obj{summary.obj_id}"
        )

    def update(
        self,
        summary: AttackSummary,
        artifacts: Dict[str, Path],
        attack_name: Optional[str] = None,
    ) -> Dict[str, bool]:
        attack_key = attack_name or summary.attack_name
        container = self.state.setdefault(attack_key, {"best": None, "worst": None})

        if summary.clean_iou < self.min_clean_iou:
            return {"best": False, "worst": False, "skipped": True}

        score_value = summary.clean_iou - summary.adv_iou

        best_updated = False
        worst_updated = False

        if container["best"] is None or score_value > container["best"]["score"]:
            best_updated = True
            container["best"] = self._store_record(
                summary,
                artifacts,
                score_value,
                target_root=self.best_root / attack_key,
                existing=container["best"],
            )

        if container["worst"] is None or score_value < container["worst"]["score"]:
            worst_updated = True
            container["worst"] = self._store_record(
                summary,
                artifacts,
                score_value,
                target_root=self.worst_root / attack_key,
                existing=container["worst"],
            )

        if best_updated or worst_updated:
            self._save_state()

        return {"best": best_updated, "worst": worst_updated, "skipped": False}

    def _store_record(
        self,
        summary: AttackSummary,
        artifacts: Dict[str, Path],
        score: float,
        target_root: Path,
        existing: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        target_root = ensure_dir(target_root)
        if existing is not None and "artifacts" in existing:
            self._remove_artifacts(existing["artifacts"])

        prefix = self._format_prefix(summary)
        stored_paths: Dict[str, str] = {}
        for name, src in artifacts.items():
            src_path = Path(src)
            if not src_path.exists():
                continue
            dest_path = target_root / f"{prefix}_{name}{src_path.suffix}"
            shutil.copy2(src_path, dest_path)
            stored_paths[name] = str(dest_path)

        record = {
            "score": score,
            "clean_iou": summary.clean_iou,
            "adv_iou": summary.adv_iou,
            "summary": asdict(summary),
            "artifacts": stored_paths,
        }
        return record
