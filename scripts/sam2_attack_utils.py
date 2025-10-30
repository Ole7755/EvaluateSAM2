"""
Shared utilities for SAM2 adversarial experiments.

The implementation fuses our original tooling with helper patterns extracted from
the reference project “Vanish into Thin Air: Cross-prompt Universal Adversarial
Attacks for SAM2”.  Core responsibilities:

* IO helpers for RGB frames / mask tensors plus optional resize bookkeeping;
* metric utilities (IoU, Dice) and perturbation statistics;
* lightweight logging helpers for attacks;
* convenience routines (frame index parsing, overlay visualisation) ported from
  the reference wheels.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .evaluate_sam2_metrics import compute_iou_and_dice

# ImageNet normalisation constants (match SAM2 training pipeline)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_rgb_tensor(image_path: Path, device: torch.device, normalize: bool = True) -> torch.Tensor:
    """
    Load an RGB frame as a contiguous tensor.

    Args:
        image_path: RGB frame path.
        device: torch device to place the tensor.
        normalize: divide by 255 if True.
    """
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image, dtype=np.float32)
    if normalize:
        array /= 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(device)
    return tensor.contiguous()


def save_rgb_tensor(tensor: torch.Tensor, save_path: Path) -> None:
    """Persist a tensor in [0,1] range as PNG."""
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    array = (tensor.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(save_path)


def load_mask_tensor(mask_path: Path, device: torch.device) -> torch.Tensor:
    """Load a mask image (possibly multi-label) as float tensor."""
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return torch.from_numpy(mask.astype(np.float32)).to(device)


def mask_to_binary(mask_tensor: torch.Tensor, label: Optional[int] = None, threshold: float = 0.0) -> torch.Tensor:
    """
    Convert a raw mask tensor to a binary tensor.

    Args:
        mask_tensor: raw mask (float).
        label: keep only this label id if provided.
        threshold: threshold when label is None.
    """
    if label is not None:
        binary = (mask_tensor == float(label)).to(mask_tensor.dtype)
    else:
        binary = (mask_tensor > threshold).to(mask_tensor.dtype)
    return binary


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute Dice loss (1 - Dice coefficient)."""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return torch.clamp(1.0 - dice, min=0.0, max=1.0)


def bce_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary cross entropy between probabilities and targets."""
    pred = torch.clamp(pred, eps, 1.0 - eps)
    loss = -(target * pred.log() + (1 - target) * (1 - pred).log())
    return loss.mean()


def eval_masks_numpy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    """Return mIoU and Dice for numpy masks."""
    return compute_iou_and_dice(pred_mask.astype(bool), gt_mask.astype(bool))


@dataclass
class ResizePadInfo:
    """Keep resize metadata aligned with image pre-processing."""

    orig_height: int
    orig_width: int
    resized_height: int
    resized_width: int
    target_size: int
    scale_y: float
    scale_x: float
    pad_bottom: int
    pad_right: int
    keep_aspect_ratio: bool


def resize_image_tensor(
    tensor: torch.Tensor,
    target_size: int,
    keep_aspect_ratio: bool = False,
) -> tuple[torch.Tensor, ResizePadInfo]:
    """
    Resize an image tensor to SAM2 square input.

    When keep_aspect_ratio=True we follow the reference wheel: resize by long
    side, then pad bottom/right with zeros.
    """
    if tensor.ndim not in (3, 4):
        raise ValueError("resize_image_tensor expects (C,H,W) or (B,C,H,W).")

    orig_height, orig_width = tensor.shape[-2], tensor.shape[-1]
    need_squeeze = tensor.ndim == 3
    batch_tensor = tensor.unsqueeze(0) if need_squeeze else tensor

    if keep_aspect_ratio:
        scale = float(target_size) / float(max(orig_height, orig_width))
        resized_height = max(int(round(orig_height * scale)), 1)
        resized_width = max(int(round(orig_width * scale)), 1)
        resized = F.interpolate(
            batch_tensor,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        padded = torch.zeros(
            (resized.size(0), resized.size(1), target_size, target_size),
            dtype=resized.dtype,
            device=resized.device,
        )
        padded[..., :resized_height, :resized_width] = resized
        output = padded
        pad_bottom = target_size - resized_height
        pad_right = target_size - resized_width
        scale_y = scale_x = scale
    else:
        resized_height = target_size
        resized_width = target_size
        resized = F.interpolate(
            batch_tensor,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        output = resized
        pad_bottom = 0
        pad_right = 0
        scale_y = float(target_size) / float(orig_height)
        scale_x = float(target_size) / float(orig_width)

    if need_squeeze:
        output = output.squeeze(0)

    info = ResizePadInfo(
        orig_height=orig_height,
        orig_width=orig_width,
        resized_height=resized_height,
        resized_width=resized_width,
        target_size=target_size,
        scale_y=scale_y,
        scale_x=scale_x,
        pad_bottom=pad_bottom,
        pad_right=pad_right,
        keep_aspect_ratio=keep_aspect_ratio,
    )
    return output.contiguous(), info


def resize_mask_tensor(tensor: torch.Tensor, info: ResizePadInfo) -> torch.Tensor:
    """Resize/pad mask according to resize metadata."""
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = tensor.unsqueeze(0)
        else:
            raise ValueError("3D mask tensor must be (1,H,W).")
    elif tensor.ndim != 4:
        raise ValueError("resize_mask_tensor expects (H,W), (1,H,W) or (B,1,H,W).")

    tensor = tensor.float()
    resized = F.interpolate(
        tensor,
        size=(info.resized_height, info.resized_width),
        mode="bilinear",
        align_corners=False,
    )
    resized = (resized >= 0.5).float()
    if info.keep_aspect_ratio:
        padded = torch.zeros(
            (resized.size(0), resized.size(1), info.target_size, info.target_size),
            dtype=resized.dtype,
            device=resized.device,
        )
        padded[..., : info.resized_height, : info.resized_width] = resized
        return padded
    return resized


def _crop_to_resized_region(tensor: torch.Tensor, info: ResizePadInfo) -> torch.Tensor:
    return tensor[..., : info.resized_height, : info.resized_width]


def restore_image_tensor(
    tensor: torch.Tensor,
    info: ResizePadInfo,
    output_hw: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """Restore resized image/perturbation back to the original spatial size."""
    if tensor.ndim not in (3, 4):
        raise ValueError("restore_image_tensor expects (C,H,W) or (B,C,H,W).")

    need_squeeze = tensor.ndim == 3
    if need_squeeze:
        tensor = tensor.unsqueeze(0)

    cropped = _crop_to_resized_region(tensor, info)
    align = mode in {"bilinear", "bicubic"}
    restored = F.interpolate(
        cropped,
        size=output_hw,
        mode=mode,
        align_corners=False if align else None,
    )
    if need_squeeze:
        restored = restored.squeeze(0)
    return restored


def mask_probs_to_numpy(
    probs: torch.Tensor,
    info: ResizePadInfo,
    output_hw: Tuple[int, int],
    threshold: float,
) -> np.ndarray:
    """Map SAM2 mask probabilities back to original resolution boolean array."""
    if probs.ndim == 3:
        probs = probs.unsqueeze(0)
    if probs.ndim != 4 or probs.size(1) != 1:
        raise ValueError("mask_probs_to_numpy expects shape (B,1,H,W).")

    binary = (probs > threshold).float()
    cropped = _crop_to_resized_region(binary, info)
    restored = F.interpolate(cropped, size=output_hw, mode="nearest")
    mask = restored.squeeze(0).squeeze(0).detach().cpu().numpy()
    return mask.astype(bool)


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation."""
    mean = IMAGENET_MEAN.to(tensor.device)
    std = IMAGENET_STD.to(tensor.device)
    tensor = tensor * std + mean
    return tensor.clamp(0.0, 1.0)


def compute_perturbation_norms(perturbation: torch.Tensor) -> Dict[str, float]:
    """Return L1/L2/L_inf norms averaged over batch."""
    flat = perturbation.view(perturbation.size(0), -1)
    norms = {
        "l2": float(torch.linalg.vector_norm(flat, ord=2, dim=1).mean().item()),
        "linf": float(torch.linalg.vector_norm(flat, ord=float("inf"), dim=1).mean().item()),
        "l1": float(torch.linalg.vector_norm(flat, ord=1, dim=1).mean().item()),
    }
    return norms


@dataclass
class AttackConfig:
    """Attack configuration stub stored alongside logs."""

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
    """Key metrics from one attack run."""

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
    """Handle JSON summaries and tensor artefacts for an attack."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = ensure_dir(log_dir)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def save_config(self, config: AttackConfig) -> Path:
        path = self.log_dir / f"{config.attack_name}_config_{self._timestamp()}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(config), f, ensure_ascii=False, indent=2)
        return path

    def save_summary(self, summary: AttackSummary, extra: Optional[Dict[str, Any]] = None) -> Path:
        payload: Dict[str, Any] = asdict(summary)
        if extra:
            payload.update(extra)
        path = self.log_dir / f"{summary.sequence}_{summary.obj_id}_{summary.frame_idx}_{self._timestamp()}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def save_tensor(self, tensor: torch.Tensor, name: str) -> Path:
        path = self.log_dir / f"{name}_{self._timestamp()}.pt"
        torch.save(tensor.detach().cpu(), path)
        return path


class BestWorstTracker:
    """
    Track best / worst cases for a given attack type and persist artefacts.

    Inspired by the evaluation routines bundled with the reference project,
    but integrated with our logging directories.
    """

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


def make_print_to_file(path: Path, filename_prefix: str = "") -> Path:
    """
    Create a timestamped log file and redirect stdout to it.

    Matches the logging helper seen in cross_prompts_alluap_test.py but keeps
    configuration minimal for local experiments.
    """
    ensure_dir(path)
    timestamp = datetime.now().strftime("%Y_%m_%d")
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    logfile = path / f"{prefix}{timestamp}.log"
    logfile.touch(exist_ok=True)
    return logfile


def compute_clean_adv_metrics(
    clean_mask: np.ndarray,
    adv_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> Dict[str, float]:
    """Utility bundled with evaluation loops to compare clean vs adversarial masks."""
    clean_iou, clean_dice = eval_masks_numpy(clean_mask, gt_mask)
    adv_iou, adv_dice = eval_masks_numpy(adv_mask, gt_mask)
    return {
        "clean_iou": clean_iou,
        "clean_dice": clean_dice,
        "adv_iou": adv_iou,
        "adv_dice": adv_dice,
        "delta_iou": clean_iou - adv_iou,
        "delta_dice": clean_dice - adv_dice,
    }


def get_frame_index(img_id: str) -> int:
    """Parse frame index tokens such as 00000 or 00012_id0."""
    stem = Path(img_id).stem
    token = stem.split("_", maxsplit=1)[0]
    if not token.isdigit():
        raise ValueError(f"Unexpected frame token: {img_id}")
    return int(token)


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.3) -> np.ndarray:
    """Composite a binary mask onto an RGB image for quick visualisation."""
    if cv2 is None:
        raise RuntimeError("overlay_mask_on_image requires OpenCV (cv2) to be installed.")
    mask_bool = mask.astype(bool)
    overlay = np.zeros_like(image)
    overlay[mask_bool] = color
    return cv2.addWeighted(image, 1.0, overlay, alpha, 0.0)


try:
    import cv2  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore[assignment]


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "ResizePadInfo",
    "AttackConfig",
    "AttackLogger",
    "AttackSummary",
    "BestWorstTracker",
    "bce_loss",
    "dice_loss",
    "denormalize_image",
    "compute_perturbation_norms",
    "compute_clean_adv_metrics",
    "ensure_dir",
    "eval_masks_numpy",
    "load_mask_tensor",
    "load_rgb_tensor",
    "mask_probs_to_numpy",
    "mask_to_binary",
    "resize_image_tensor",
    "resize_mask_tensor",
    "restore_image_tensor",
    "save_rgb_tensor",
    "make_print_to_file",
    "get_frame_index",
    "overlay_mask_on_image",
]
