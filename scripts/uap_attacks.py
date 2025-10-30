"""
Universal adversarial attacks for SAM2.

Design borrowed from the reference project “Vanish into Thin Air: Cross-prompt
Universal Adversarial Attacks for SAM2”, while keeping our original forward
helper API.  We expose FGSM / BIM / PGD / C&W attacks plus a differentiable
SAM2 forward wrapper that mirrors the prompt-handling logic from the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .sam2_attack_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    bce_loss,
    dice_loss,
)


@dataclass
class ForwardOutput:
    """SAM2 forward pass output bundle."""

    logits: torch.Tensor
    probs: torch.Tensor


class ForwardHelperBase:
    """Abstract base class wrapping SAM2 inference for gradient back-prop."""

    def forward(self, image_tensor: torch.Tensor, **kwargs) -> ForwardOutput:  # pragma: no cover - interface
        raise NotImplementedError


class BaseUAPAttack:
    """Common helper functions shared by all UAP variants."""

    def __init__(
        self,
        epsilon: float,
        step_size: float,
        steps: int,
        loss_type: str = "dice",
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        self.epsilon = epsilon
        self.step_size = step_size
        self.steps = steps
        self.loss_type = loss_type
        self.clip_min = clip_min
        self.clip_max = clip_max

    def _loss(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "dice":
            return dice_loss(probs, target)
        if self.loss_type == "bce":
            return bce_loss(probs, target)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _project(self, perturbation: torch.Tensor) -> torch.Tensor:
        if self.epsilon <= 0:
            return torch.zeros_like(perturbation)
        return torch.clamp(perturbation, -self.epsilon, self.epsilon)

    def generate(  # pragma: no cover - interface
        self,
        helper: ForwardHelperBase,
        clean_image: torch.Tensor,
        target_mask: torch.Tensor,
        random_start: bool = False,
        **forward_kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


class FGSMAttack(BaseUAPAttack):
    """Single-step FGSM attack."""

    def generate(
        self,
        helper: ForwardHelperBase,
        clean_image: torch.Tensor,
        target_mask: torch.Tensor,
        random_start: bool = False,
        **forward_kwargs,
    ) -> torch.Tensor:
        image = clean_image.detach()
        if random_start and self.epsilon > 0:
            delta = torch.empty_like(image).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.zeros_like(image)

        adv = (image + delta).clamp(self.clip_min, self.clip_max)
        adv.requires_grad_(True)

        output = helper.forward(adv, prompt_mask=target_mask, **forward_kwargs)
        loss = self._loss(output.probs, target_mask)
        loss.backward()

        perturbation = self.epsilon * adv.grad.data.sign()
        perturbation = self._project(perturbation)
        adv_image = (image + perturbation).clamp(self.clip_min, self.clip_max)
        return adv_image.detach() - image


class BIMAttack(BaseUAPAttack):
    """Iterative FGSM (a deterministic PGD variant)."""

    def generate(
        self,
        helper: ForwardHelperBase,
        clean_image: torch.Tensor,
        target_mask: torch.Tensor,
        random_start: bool = False,
        **forward_kwargs,
    ) -> torch.Tensor:
        image = clean_image.detach()
        delta = torch.zeros_like(image)

        for _ in range(self.steps):
            adv = (image + delta).clamp(self.clip_min, self.clip_max)
            adv.requires_grad_(True)

            output = helper.forward(adv, prompt_mask=target_mask, **forward_kwargs)
            loss = self._loss(output.probs, target_mask)
            loss.backward()

            delta = delta + self.step_size * adv.grad.detach().sign()
            delta = self._project(delta).detach()

        adv_image = (image + delta).clamp(self.clip_min, self.clip_max)
        return adv_image - image


class PGDAttack(BaseUAPAttack):
    """Projected gradient descent with optional random start."""

    def generate(
        self,
        helper: ForwardHelperBase,
        clean_image: torch.Tensor,
        target_mask: torch.Tensor,
        random_start: bool = False,
        **forward_kwargs,
    ) -> torch.Tensor:
        image = clean_image.detach()
        if random_start and self.epsilon > 0:
            delta = torch.empty_like(image).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.zeros_like(image)

        for _ in range(self.steps):
            adv = (image + delta).clamp(self.clip_min, self.clip_max)
            adv.requires_grad_(True)

            output = helper.forward(adv, prompt_mask=target_mask, **forward_kwargs)
            loss = self._loss(output.probs, target_mask)
            loss.backward()

            delta = delta + self.step_size * adv.grad.detach().sign()
            delta = self._project(delta).detach()

        adv_image = (image + delta).clamp(self.clip_min, self.clip_max)
        return adv_image - image


class CarliniWagnerAttack(BaseUAPAttack):
    """L2 Carlini & Wagner attack with tanh-space optimisation."""

    def __init__(
        self,
        epsilon: float,
        step_size: float,
        steps: int,
        confidence: float = 0.0,
        binary_steps: int = 5,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        loss_type: str = "dice",
    ) -> None:
        super().__init__(
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            loss_type=loss_type,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        self.confidence = confidence
        self.binary_steps = binary_steps

    def _to_tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.clip_min) / (self.clip_max - self.clip_min)
        x = torch.clamp(x * 2 - 1, -0.999999, 0.999999)
        return torch.atanh(x)

    def _from_tanh_space(self, w: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(w)
        x = (x + 1) / 2
        return x * (self.clip_max - self.clip_min) + self.clip_min

    def generate(
        self,
        helper: ForwardHelperBase,
        clean_image: torch.Tensor,
        target_mask: torch.Tensor,
        random_start: bool = False,
        **forward_kwargs,
    ) -> torch.Tensor:
        image = clean_image.detach()
        device = image.device

        w = self._to_tanh_space(image).detach()
        w.requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=self.step_size)

        best_adv = image.clone()
        best_loss = torch.full((1,), float("inf"), device=device)

        const_lower = torch.zeros(1, device=device)
        const_upper = torch.full((1,), 1e4, device=device)
        const = torch.ones(1, device=device)

        for _ in range(self.binary_steps):
            w.data = self._to_tanh_space(image)
            for _ in range(self.steps):
                adv_image = self._from_tanh_space(w)
                output = helper.forward(adv_image, prompt_mask=target_mask, **forward_kwargs)
                loss_adv = self._loss(output.probs, target_mask) - self.confidence
                diff = adv_image - image
                loss_dist = torch.sum(diff * diff)
                loss_total = const * loss_adv + loss_dist

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                with torch.no_grad():
                    better = loss_adv < best_loss
                    if better.item():
                        best_loss = loss_adv.detach()
                        best_adv = adv_image.detach()

            with torch.no_grad():
                if best_loss <= 0:
                    const_upper = torch.min(const_upper, const)
                    const = (const_lower + const_upper) / 2
                else:
                    const_lower = torch.max(const_lower, const)
                    if const_upper < 1e9:
                        const = (const_lower + const_upper) / 2
                    else:
                        const = const * 2

        final_adv = best_adv.clamp(self.clip_min, self.clip_max)
        perturbation = final_adv - image
        if self.epsilon > 0:
            norm = torch.norm(perturbation.view(-1), p=float("inf"))
            if norm > self.epsilon:
                perturbation = perturbation * (self.epsilon / norm)
        adv_image = (image + perturbation).clamp(self.clip_min, self.clip_max)
        return adv_image - image


class SAM2ForwardHelper(ForwardHelperBase):
    """
    Differentiable wrapper around SAM2 predictor.

    The code mirrors the helper used in the reference implementation:
    - normalise input with ImageNet statistics;
    - forward through SAM2 image encoder;
    - optionally accept point / box / mask prompts.
    """

    def __init__(self, predictor, device: torch.device) -> None:
        self.predictor = predictor.to(device)
        self.predictor.eval()

        self.prompt_encoder = self.predictor.sam_prompt_encoder
        self.mask_decoder = self.predictor.sam_mask_decoder

        self.prompt_encoder.to(device)
        self.mask_decoder.to(device)

        self.device = device
        self.mean = IMAGENET_MEAN.to(device)
        self.std = IMAGENET_STD.to(device)

    def forward(
        self,
        image_tensor: torch.Tensor,
        prompt_points: Optional[torch.Tensor] = None,
        prompt_boxes: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> ForwardOutput:
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        normalized = (image_tensor - self.mean) / self.std

        backbone_out = self.predictor.forward_image(normalized)
        (
            backbone_out,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self.predictor._prepare_backbone_features(backbone_out)

        reshaped_feats: list[torch.Tensor] = [
            self._reshape_feature(feat, H, W) for feat, (H, W) in zip(current_vision_feats, feat_sizes)
        ]

        target_hw = self.predictor.sam_image_embedding_size

        backbone_features = None
        high_res_candidates: list[torch.Tensor] = []
        for reshaped, (H, W) in zip(reshaped_feats, feat_sizes):
            if H == target_hw:
                backbone_features = reshaped
            elif H > target_hw:
                high_res_candidates.append(reshaped)

        if backbone_features is None:
            if not reshaped_feats:
                raise ValueError("Image encoder returned no features.")
            backbone_features = reshaped_feats[-1]
            backbone_features = F.interpolate(backbone_features, size=(target_hw, target_hw), mode="nearest")

        if backbone_features.ndim != 4:
            raise ValueError(f"Unexpected feature shape: {backbone_features.shape}")
        if backbone_features.size(-1) != target_hw:
            backbone_features = F.interpolate(backbone_features, size=(target_hw, target_hw), mode="nearest")

        if len(high_res_candidates) >= 2:
            high_res_features = (high_res_candidates[0], high_res_candidates[1])
        elif len(high_res_candidates) == 1:
            high_res_features = (high_res_candidates[0], high_res_candidates[0])
        else:
            high_res_features = None

        point_inputs = self._prepare_point_inputs(prompt_points)
        mask_inputs = self._prepare_mask_inputs(prompt_mask)
        box_inputs = self._prepare_box_inputs(prompt_boxes)

        sam_outputs = self.predictor._forward_sam_heads(
            backbone_features=backbone_features,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            boxes=box_inputs,
            high_res_features=high_res_features,
            multimask_output=False,
        )

        low_res_masks = sam_outputs[3]
        logits = F.interpolate(low_res_masks, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits)
        return ForwardOutput(logits=logits, probs=probs)

    def _reshape_feature(self, feat: torch.Tensor, H: int, W: int) -> torch.Tensor:
        if feat.ndim == 4:
            if feat.shape[-2:] == (H, W):
                return feat
            if feat.shape[1:3] == (H, W):
                return feat.permute(0, 3, 1, 2).contiguous()
        elif feat.ndim == 3:
            if feat.shape[1] == H * W:
                batch, tokens, dim = feat.shape
                return feat.transpose(1, 2).contiguous().view(batch, dim, H, W)
            if feat.shape[0] == H * W:
                tokens, batch, dim = feat.shape
                return feat.permute(1, 2, 0).contiguous().view(batch, dim, H, W)
            if feat.shape[2] == H * W:
                batch, dim, tokens = feat.shape
                return feat.view(batch, dim, H, W)
        raise ValueError(f"Cannot reshape feature of shape {tuple(feat.shape)} to [B,C,{H},{W}].")

    def _prepare_point_inputs(
        self,
        prompt_points: Optional[torch.Tensor],
    ) -> Optional[dict[str, torch.Tensor]]:
        if prompt_points is None:
            return None
        if prompt_points.ndim == 2:
            prompt_points = prompt_points.unsqueeze(0)
        coords = prompt_points[..., :2].to(self.device)
        labels = prompt_points[..., 2].to(self.device).to(torch.int32)
        return {"point_coords": coords, "point_labels": labels}

    def _prepare_box_inputs(self, prompt_boxes: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if prompt_boxes is None:
            return None
        box = prompt_boxes.to(self.device)
        if box.ndim == 2:
            box = box.unsqueeze(0)
        if box.ndim != 3:
            raise ValueError("Box prompts must be shaped [B, num_boxes, 4].")
        return box

    def _prepare_mask_inputs(self, prompt_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if prompt_mask is None:
            return None
        mask = prompt_mask.to(self.device)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        if mask.ndim == 4 and mask.shape[1] == 1:
            return mask
        raise ValueError("Mask prompts must be shaped [B,1,H,W].")


__all__ = [
    "ForwardOutput",
    "ForwardHelperBase",
    "BaseUAPAttack",
    "FGSMAttack",
    "BIMAttack",
    "PGDAttack",
    "CarliniWagnerAttack",
    "SAM2ForwardHelper",
]
