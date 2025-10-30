"""
Universal adversarial patch trainer utilities.

The implementation keeps our original training pipeline while incorporating
ideas from the cross-prompt SAM2 attack reference (“Vanish into Thin Air...”):
reusable resize metadata, batched prompt handling via ``SAM2ForwardHelper`` and
consistent metrics tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .sam2_attack_utils import (
    ResizePadInfo,
    bce_loss,
    compute_clean_adv_metrics,
    dice_loss,
    load_mask_tensor,
    load_rgb_tensor,
    mask_probs_to_numpy,
    mask_to_binary,
    resize_image_tensor,
    resize_mask_tensor,
)
from .uap_attacks import SAM2ForwardHelper


_FRAME_EXTENSIONS = [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]


@dataclass
class UAPSample:
    """Per-sequence sample used for universal patch optimisation."""

    sequence: str
    frame_token: str
    frame_path: Path
    mask_path: Path
    resized_image: torch.Tensor  # (3, S, S)
    resized_mask: torch.Tensor  # (1, 1, S, S)
    resize_info: ResizePadInfo
    orig_hw: Tuple[int, int]
    gt_mask: np.ndarray  # bool


@dataclass
class StepRecord:
    """Statistics for one optimisation step."""

    step: int
    loss: float
    per_sample_losses: List[float]


@dataclass
class SampleEvaluation:
    """Per-sample evaluation metrics."""

    split: str
    sequence: str
    frame_token: str
    clean_iou: float
    clean_dice: float
    adv_iou: float
    adv_dice: float
    delta_iou: float
    delta_dice: float
    frame_path: str
    mask_path: str


@dataclass
class AggregateMetrics:
    """Aggregate metrics for a set of sample evaluations."""

    split: str
    num_samples: int
    mean_clean_iou: float
    mean_adv_iou: float
    mean_delta_iou: float
    min_adv_iou: float
    mean_clean_dice: float
    mean_adv_dice: float
    mean_delta_dice: float
    min_adv_dice: float
    max_delta_iou: float
    max_delta_dice: float

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "num_samples": self.num_samples,
            "mean_clean_iou": self.mean_clean_iou,
            "mean_adv_iou": self.mean_adv_iou,
            "mean_delta_iou": self.mean_delta_iou,
            "min_adv_iou": self.min_adv_iou,
            "mean_clean_dice": self.mean_clean_dice,
            "mean_adv_dice": self.mean_adv_dice,
            "mean_delta_dice": self.mean_delta_dice,
            "min_adv_dice": self.min_adv_dice,
            "max_delta_iou": self.max_delta_iou,
            "max_delta_dice": self.max_delta_dice,
        }

    @classmethod
    def from_samples(cls, split: str, samples: Sequence[SampleEvaluation]) -> AggregateMetrics:
        if not samples:
            raise ValueError("samples must not be empty.")
        num = len(samples)
        clean_iou_vals = [item.clean_iou for item in samples]
        adv_iou_vals = [item.adv_iou for item in samples]
        delta_iou_vals = [item.delta_iou for item in samples]
        clean_dice_vals = [item.clean_dice for item in samples]
        adv_dice_vals = [item.adv_dice for item in samples]
        delta_dice_vals = [item.delta_dice for item in samples]
        return cls(
            split=split,
            num_samples=num,
            mean_clean_iou=sum(clean_iou_vals) / num,
            mean_adv_iou=sum(adv_iou_vals) / num,
            mean_delta_iou=sum(delta_iou_vals) / num,
            min_adv_iou=min(adv_iou_vals),
            mean_clean_dice=sum(clean_dice_vals) / num,
            mean_adv_dice=sum(adv_dice_vals) / num,
            mean_delta_dice=sum(delta_dice_vals) / num,
            min_adv_dice=min(adv_dice_vals),
            max_delta_iou=max(delta_iou_vals),
            max_delta_dice=max(delta_dice_vals),
        )


@dataclass
class EvaluationSummary:
    """Return type for evaluation."""

    samples: List[SampleEvaluation]
    aggregate: Optional[AggregateMetrics]


def _find_frame_path(rgb_dir: Path, frame_token: str) -> Path:
    for ext in _FRAME_EXTENSIONS:
        candidate = rgb_dir / f"{frame_token}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Frame {frame_token} not found under {rgb_dir}.")


def load_uap_samples(
    sequences: Sequence[str],
    frame_token: str,
    data_root: Path,
    mask_subdir: str,
    device: torch.device,
    input_size: int,
    keep_aspect_ratio: bool,
    gt_label: Optional[int],
) -> List[UAPSample]:
    samples: List[UAPSample] = []
    if not sequences:
        return samples

    for seq in sequences:
        rgb_dir = data_root / "JPEGImages" / "480p" / seq
        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory missing: {rgb_dir}")
        mask_dir = data_root / mask_subdir / "480p" / seq
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory missing: {mask_dir}")

        frame_path = _find_frame_path(rgb_dir, frame_token)
        mask_path = mask_dir / f"{frame_token}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask missing: {mask_path}")

        image_tensor = load_rgb_tensor(frame_path, device=device)
        mask_tensor = load_mask_tensor(mask_path, device=device)
        binary_mask = mask_to_binary(mask_tensor, label=gt_label)

        resized_image, resize_info = resize_image_tensor(
            image_tensor, input_size, keep_aspect_ratio=keep_aspect_ratio
        )
        resized_mask = resize_mask_tensor(binary_mask, resize_info).to(device)

        samples.append(
            UAPSample(
                sequence=seq,
                frame_token=frame_token,
                frame_path=frame_path,
                mask_path=mask_path,
                resized_image=resized_image.contiguous(),
                resized_mask=resized_mask.contiguous(),
                resize_info=resize_info,
                orig_hw=(int(image_tensor.shape[-2]), int(image_tensor.shape[-1])),
                gt_mask=(binary_mask.detach().cpu().numpy() > 0.5),
            )
        )
    return samples


def match_sample(
    samples: Iterable[UAPSample],
    sequence: str,
    frame_token: str,
) -> Optional[UAPSample]:
    """Search cached samples by sequence and frame token."""
    for sample in samples:
        if sample.sequence == sequence and sample.frame_token == frame_token:
            return sample
    return None


class UniversalPatchTrainer:
    """Train universal patches across multiple sequences."""

    def __init__(
        self,
        helper: SAM2ForwardHelper,
        samples: Sequence[UAPSample],
        epsilon: float,
        loss_type: str = "dice",
        mask_threshold: float = 0.5,
    ) -> None:
        if not samples:
            raise ValueError("samples must not be empty.")
        if loss_type not in {"dice", "bce"}:
            raise ValueError("loss_type must be one of {'dice', 'bce'}.")

        self.helper = helper
        self.device = helper.device
        self.samples = list(samples)
        self.epsilon = float(epsilon)
        self.mask_threshold = float(mask_threshold)
        self.loss_type = loss_type
        self.loss_fn = dice_loss if loss_type == "dice" else bce_loss
        self.target_size = self.samples[0].resize_info.target_size

        for param in self.helper.predictor.parameters():
            param.requires_grad_(False)

    def _loss_with_patch(self, patch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        losses: List[torch.Tensor] = []
        for sample in self.samples:
            clean = sample.resized_image.unsqueeze(0)
            adv = torch.clamp(clean + patch, 0.0, 1.0)
            output = self.helper.forward(adv, prompt_mask=sample.resized_mask)
            loss = self.loss_fn(output.probs, sample.resized_mask)
            losses.append(loss)
        stacked = torch.stack(losses)
        return stacked.mean(), stacked

    def _compute_gradient(self, patch: torch.Tensor) -> Tuple[torch.Tensor, float, List[float]]:
        patch_var = patch.clone().detach().requires_grad_(True)
        mean_loss, per_sample = self._loss_with_patch(patch_var)
        mean_loss.backward()
        grad = patch_var.grad.detach()
        losses = per_sample.detach().cpu().tolist()
        return grad, float(mean_loss.item()), losses

    def _init_patch(self, random_start: bool) -> torch.Tensor:
        patch = torch.zeros((1, 3, self.target_size, self.target_size), device=self.device)
        if random_start and self.epsilon > 0:
            patch.uniform_(-self.epsilon, self.epsilon)
        return patch

    def train(
        self,
        steps: int,
        step_size: float,
        random_start: bool = False,
        attack_type: str = "pgd",
        cw_lr: float = 0.01,
        cw_confidence: float = 0.0,
        cw_binary_steps: int = 5,
    ) -> Tuple[torch.Tensor, List[StepRecord]]:
        attack_type = attack_type.lower()
        if attack_type == "cw":
            return self._train_cw(
                steps=steps,
                lr=cw_lr,
                confidence=cw_confidence,
                binary_steps=cw_binary_steps,
            )
        if attack_type in {"pgd", "bim", "fgsm"}:
            return self._train_iterative(
                steps=steps,
                step_size=step_size,
                random_start=random_start,
                attack_type=attack_type,
            )
        raise ValueError(f"Unsupported attack_type: {attack_type}")

    def _train_iterative(
        self,
        steps: int,
        step_size: float,
        random_start: bool,
        attack_type: str,
    ) -> Tuple[torch.Tensor, List[StepRecord]]:
        patch = self._init_patch(random_start)

        patch = torch.clamp(patch, -self.epsilon, self.epsilon).detach()
        history: List[StepRecord] = []

        for idx in range(int(steps)):
            grad, loss, per_sample = self._compute_gradient(patch)
            if attack_type == "fgsm" and idx > 0:
                break
            update = step_size * torch.sign(grad)
            patch = torch.clamp(patch + update, -self.epsilon, self.epsilon).detach()
            history.append(StepRecord(step=idx, loss=loss, per_sample_losses=list(per_sample)))

        return patch, history

    def _train_cw(
        self,
        steps: int,
        lr: float,
        confidence: float,
        binary_steps: int,
    ) -> Tuple[torch.Tensor, List[StepRecord]]:
        w = torch.zeros((1, 3, self.target_size, self.target_size), device=self.device, requires_grad=True)
        const_lower = torch.tensor(0.0, device=self.device)
        const_upper = torch.tensor(1e4, device=self.device)
        const = torch.tensor(1.0, device=self.device)

        best_patch = torch.zeros_like(w)
        best_adv = float("inf")
        history: List[StepRecord] = []

        for binary_idx in range(binary_steps):
            optimizer = torch.optim.Adam([w], lr=float(lr))
            binary_best = float("inf")
            binary_patch = best_patch.clone()

            for step_idx in range(int(steps)):
                patch = torch.tanh(w) * self.epsilon
                mean_loss, per_sample = self._loss_with_patch(patch)
                loss_total = const * (mean_loss - confidence) + torch.sum(patch * patch)

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                adv_loss = float(mean_loss.item())
                losses = per_sample.detach().cpu().tolist()
                global_step = binary_idx * int(steps) + step_idx
                history.append(StepRecord(step=global_step, loss=adv_loss, per_sample_losses=losses))

                if adv_loss < binary_best:
                    binary_best = adv_loss
                    binary_patch = patch.detach()

                if adv_loss < best_adv:
                    best_adv = adv_loss
                    best_patch = patch.detach()

            with torch.no_grad():
                if binary_best <= confidence:
                    const_upper = torch.min(const_upper, const)
                    const = (const_lower + const_upper) / 2
                else:
                    const_lower = torch.max(const_lower, const)
                    if const_upper < 10_000:
                        const = (const_lower + const_upper) / 2
                    else:
                        const = const * 2

                w.data = torch.atanh(torch.clamp(binary_patch / max(self.epsilon, 1e-6), -0.999999, 0.999999))

        return best_patch.detach(), history

    @torch.no_grad()
    def evaluate(
        self,
        patch: torch.Tensor,
        samples: Sequence[UAPSample],
        split: str,
    ) -> EvaluationSummary:
        if not samples:
            return EvaluationSummary(samples=[], aggregate=None)

        patch = patch.to(self.device)
        records: List[SampleEvaluation] = []
        for sample in samples:
            clean = sample.resized_image.unsqueeze(0)
            clean_out = self.helper.forward(clean, prompt_mask=sample.resized_mask)
            clean_mask_np = mask_probs_to_numpy(
                clean_out.probs, sample.resize_info, sample.orig_hw, self.mask_threshold
            )

            adv = torch.clamp(clean + patch, 0.0, 1.0)
            adv_out = self.helper.forward(adv, prompt_mask=sample.resized_mask)
            adv_mask_np = mask_probs_to_numpy(
                adv_out.probs, sample.resize_info, sample.orig_hw, self.mask_threshold
            )

            metrics = compute_clean_adv_metrics(clean_mask_np, adv_mask_np, sample.gt_mask)
            records.append(
                SampleEvaluation(
                    split=split,
                    sequence=sample.sequence,
                    frame_token=sample.frame_token,
                    clean_iou=float(metrics["clean_iou"]),
                    clean_dice=float(metrics["clean_dice"]),
                    adv_iou=float(metrics["adv_iou"]),
                    adv_dice=float(metrics["adv_dice"]),
                    delta_iou=float(metrics["delta_iou"]),
                    delta_dice=float(metrics["delta_dice"]),
                    frame_path=str(sample.frame_path),
                    mask_path=str(sample.mask_path),
                )
            )

        aggregate = AggregateMetrics.from_samples(split, records)
        return EvaluationSummary(samples=records, aggregate=aggregate)
