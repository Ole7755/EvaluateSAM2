"""
实现针对 SAM2 的通用扰动（UAP）攻击算法。

包含以下攻击：
1. FGSM
2. BIM（迭代 FGSM）
3. PGD
4. C&W

说明：
- 这些实现假设可对输入图像张量求导；
- 实际运行时，需要配合 ``SAM2ForwardHelper``，确保前向传播保留梯度。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .sam2_attack_utils import (
    dice_loss,
    bce_loss,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


@dataclass
class ForwardOutput:
    """
    SAM2ForwardHelper 的输出封装。

    属性：
        logits: SAM2 输出的掩码对数几率（未过 sigmoid）。
        probs: 经过 sigmoid 的概率图（方便直接计算损失）。
    """

    logits: torch.Tensor
    probs: torch.Tensor


# 为攻击器提供可微前向接口的基础类
class ForwardHelperBase:
    """
    将 SAM2 推理流程包装为可求导的函数。

    子类或实例需实现 ``forward`` 方法，接收图像张量并返回 ForwardOutput。
    """

    def forward(self, image_tensor: torch.Tensor) -> ForwardOutput:  # pragma: no cover - 接口方法
        raise NotImplementedError("请在具体环境中实现 forward，用于驱动 SAM2 产生掩码。")


class BaseUAPAttack:
    """UAP 攻击基类，提供通用的损失函数与裁剪逻辑。"""

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
        """
        计算攻击目标的损失函数。

        当前支持：
            - dice: 1 - Dice
            - bce: 二值交叉熵
        """
        if self.loss_type == "dice":
            return dice_loss(probs, target)
        if self.loss_type == "bce":
            return bce_loss(probs, target)
        raise ValueError(f"未知的损失类型：{self.loss_type}")

    def _project(self, perturbation: torch.Tensor) -> torch.Tensor:
        """将扰动限制在 L_inf 球内。"""
        if self.epsilon <= 0:
            return torch.zeros_like(perturbation)
        return torch.clamp(perturbation, -self.epsilon, self.epsilon)

    def generate(
        self,
        helper: ForwardHelperBase,
        clean_image: torch.Tensor,
        target_mask: torch.Tensor,
        random_start: bool = False,
    ) -> torch.Tensor:  # pragma: no cover - 接口方法
        raise NotImplementedError


class FGSMAttack(BaseUAPAttack):
    """单步 FGSM 攻击。"""

    def generate(
        self,
        helper: ForwardHelperBase,
        clean_image: torch.Tensor,
        target_mask: torch.Tensor,
        random_start: bool = False,
    ) -> torch.Tensor:
        image = clean_image.detach()
        if random_start and self.epsilon > 0:
            delta = torch.empty_like(image).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.zeros_like(image)

        adv = (image + delta).clamp(self.clip_min, self.clip_max)
        adv.requires_grad_(True)

        output = helper.forward(adv)
        loss = self._loss(output.probs, target_mask)
        loss.backward()

        grad_sign = adv.grad.data.sign()
        perturbation = self.epsilon * grad_sign
        perturbation = self._project(perturbation)
        adv_image = (image + perturbation).clamp(self.clip_min, self.clip_max)
        adv_image = adv_image.detach()
        return adv_image - image


class BIMAttack(BaseUAPAttack):
    """
    Basic Iterative Method（迭代 FGSM）。

    注意：BIM 是 PGD 的确定性版本，此处实现与 PGD 类似但不包含随机初始点。
    """

    def generate(
        self,
        helper: ForwardHelperBase,
        clean_image: torch.Tensor,
        target_mask: torch.Tensor,
        random_start: bool = False,
    ) -> torch.Tensor:
        image = clean_image.detach()
        delta = torch.zeros_like(image)

        for _ in range(self.steps):
            adv = (image + delta).clamp(self.clip_min, self.clip_max)
            adv.requires_grad_(True)
            output = helper.forward(adv)
            loss = self._loss(output.probs, target_mask)
            loss.backward()

            grad_sign = adv.grad.detach().sign()
            delta = delta + self.step_size * grad_sign
            delta = self._project(delta)
            delta = delta.detach()

        adv_image = (image + delta).clamp(self.clip_min, self.clip_max)
        return adv_image - image


class PGDAttack(BaseUAPAttack):
    """投影梯度下降攻击，可选择随机初始化。"""

    def generate(
        self,
        helper: ForwardHelperBase,
        clean_image: torch.Tensor,
        target_mask: torch.Tensor,
        random_start: bool = False,
    ) -> torch.Tensor:
        image = clean_image.detach()
        if random_start and self.epsilon > 0:
            delta = torch.empty_like(image).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.zeros_like(image)

        for _ in range(self.steps):
            adv = (image + delta).clamp(self.clip_min, self.clip_max)
            adv.requires_grad_(True)
            output = helper.forward(adv)
            loss = self._loss(output.probs, target_mask)
            loss.backward()

            grad_sign = adv.grad.detach().sign()
            delta = delta + self.step_size * grad_sign
            delta = self._project(delta)
            delta = delta.detach()

        adv_image = (image + delta).clamp(self.clip_min, self.clip_max)
        return adv_image - image


class CarliniWagnerAttack(BaseUAPAttack):
    """
    Carlini & Wagner 攻击的 L2 版本。

    该实现遵循原论文的逻辑：通过对变量 w 进行优化，实现对扰动进行隐式裁剪。
    """

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
        """将输入映射到 tanh 空间，便于 C&W 约束。"""
        x = x.clone()
        x = (x - self.clip_min) / (self.clip_max - self.clip_min)
        x = x * 2 - 1
        return torch.atanh(torch.clamp(x, -0.999999, 0.999999))

    def _from_tanh_space(self, w: torch.Tensor) -> torch.Tensor:
        """从 tanh 空间反变换回像素空间。"""
        x = torch.tanh(w)
        x = (x + 1) / 2
        return x * (self.clip_max - self.clip_min) + self.clip_min

    def generate(
        self,
        helper: ForwardHelperBase,
        clean_image: torch.Tensor,
        target_mask: torch.Tensor,
        random_start: bool = False,
    ) -> torch.Tensor:
        image = clean_image.detach()
        batch_shape = image.shape
        device = image.device

        w = self._to_tanh_space(image).detach()
        w.requires_grad_(True)

        optimizer = torch.optim.Adam([w], lr=self.step_size)

        best_adv = image.clone()
        best_loss = torch.full((1,), float("inf"), device=device)

        const_lower = torch.zeros(1, device=device)
        const_upper = torch.full((1,), 1e4, device=device)
        const = torch.ones(1, device=device) * 1.0

        for _ in range(self.binary_steps):
            w.data = self._to_tanh_space(image)
            for _ in range(self.steps):
                adv_image = self._from_tanh_space(w)
                output = helper.forward(adv_image)
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
        return adv_image.view(batch_shape) - image


class SAM2ForwardHelper(ForwardHelperBase):
    """
    SAM2 可微前向辅助类。

    说明：
        - 在 forward 中完成 ImageNet 归一化；
        - 默认仅支持掩码提示，如需扩展点提示可在 _encode_prompts 中补充。
    """

    def __init__(self, predictor, device: torch.device) -> None:
        self.predictor = predictor.to(device)
        self.predictor.eval()

        self.image_encoder = self.predictor.image_encoder
        self.prompt_encoder = self.predictor.sam_prompt_encoder
        self.mask_decoder = self.predictor.sam_mask_decoder

        # 确保子模块位于正确设备
        self.image_encoder.to(device)
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

        backbone_raw = self.image_encoder(normalized)
        (
            backbone_out,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self.predictor._prepare_backbone_features(backbone_raw)

        reshaped_feats: list[torch.Tensor] = [
            self._reshape_feature(feat, H, W) for feat, (H, W) in zip(current_vision_feats, feat_sizes)
        ]

        target_hw = self.predictor.sam_image_embedding_size
        if not hasattr(self, "_debug_logged"):
            debug_info = {
                "target_hw": target_hw,
                "feat_sizes": feat_sizes,
                "reshaped_shapes": [tuple(t.shape) for t in reshaped_feats],
                "backbone_out_keys": list(backbone_out.keys()) if isinstance(backbone_out, dict) else type(backbone_out),
            }
            if isinstance(backbone_out, dict) and "high_res_feats" in backbone_out:
                high_res = backbone_out["high_res_feats"]
                if isinstance(high_res, (list, tuple)):
                    debug_info["high_res_feats_shapes"] = [tuple(t.shape) for t in high_res]
                else:
                    debug_info["high_res_feats_type"] = type(high_res)
            if isinstance(backbone_out, dict) and "backbone_fpn" in backbone_out:
                backbone_fpn = backbone_out["backbone_fpn"]
                if isinstance(backbone_fpn, (list, tuple)):
                    debug_info["backbone_fpn_len"] = len(backbone_fpn)
                    debug_info["backbone_fpn_shapes"] = [tuple(t.shape) for t in backbone_fpn]
                elif isinstance(backbone_fpn, dict):
                    debug_info["backbone_fpn_keys"] = list(backbone_fpn.keys())
                    debug_info["backbone_fpn_shapes"] = {k: tuple(v.shape) for k, v in backbone_fpn.items()}
                else:
                    debug_info["backbone_fpn_type"] = type(backbone_fpn)
            print("[DEBUG] SAM2ForwardHelper feature summary:", debug_info)
            self._debug_logged = True

        backbone_features = None
        high_res_candidates: list[torch.Tensor] = []
        for reshaped, (H, W) in zip(reshaped_feats, feat_sizes):
            if H == target_hw:
                backbone_features = reshaped
            elif H > target_hw:
                high_res_candidates.append(reshaped)

        if backbone_features is None:
            if not reshaped_feats:
                raise ValueError("image encoder 未返回任何特征。")
            backbone_features = reshaped_feats[-1]
            backbone_features = F.interpolate(
                backbone_features,
                size=(target_hw, target_hw),
                mode="nearest",
            )

        if backbone_features.ndim != 4:
            raise ValueError(f"backbone_features 维度异常：{backbone_features.shape}")
        if backbone_features.size(-1) != target_hw:
            backbone_features = F.interpolate(
                backbone_features,
                size=(target_hw, target_hw),
                mode="nearest",
            )

        if len(high_res_candidates) >= 2:
            high_res_features = (high_res_candidates[0], high_res_candidates[1])
        elif len(high_res_candidates) == 1:
            high_res_features = (high_res_candidates[0], high_res_candidates[0])
        else:
            high_res_features = None

        point_inputs = self._prepare_point_inputs(prompt_points)
        mask_inputs = self._prepare_mask_inputs(prompt_mask)

        sam_outputs = self.predictor._forward_sam_heads(
            backbone_features=backbone_features,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            high_res_features=high_res_features,
            multimask_output=False,
        )

        low_res_masks = sam_outputs[3]

        logits = F.interpolate(
            low_res_masks, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        probs = torch.sigmoid(logits)
        return ForwardOutput(logits=logits, probs=probs)

    def _reshape_feature(self, feat: torch.Tensor, H: int, W: int) -> torch.Tensor:
        if feat.ndim == 4:
            if feat.shape[-2:] == (H, W):
                return feat
            if feat.shape[1:3] == (H, W):
                return feat.permute(0, 3, 1, 2).contiguous()
        elif feat.ndim == 3:
            if feat.shape[1] == H * W:  # [B, HW, C]
                batch, tokens, dim = feat.shape
                return feat.transpose(1, 2).contiguous().view(batch, dim, H, W)
            if feat.shape[0] == H * W:  # [HW, B, C]
                tokens, batch, dim = feat.shape
                return feat.permute(1, 2, 0).contiguous().view(batch, dim, H, W)
            if feat.shape[2] == H * W:  # [B, C, HW]
                batch, dim, tokens = feat.shape
                return feat.view(batch, dim, H, W)
        raise ValueError(f"无法重排特征为 [B, C, {H}, {W}]：shape={tuple(feat.shape)}")

    def _prepare_point_inputs(
        self, prompt_points: Optional[torch.Tensor]
    ) -> Optional[dict[str, torch.Tensor]]:
        if prompt_points is None:
            return None
        if prompt_points.ndim == 2:
            prompt_points = prompt_points.unsqueeze(0)
        coords = prompt_points[..., :2].to(self.device)
        labels = prompt_points[..., 2].to(self.device).to(torch.int32)
        return {"point_coords": coords, "point_labels": labels}

    def _prepare_mask_inputs(self, prompt_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if prompt_mask is None:
            return None
        mask = prompt_mask.to(self.device)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)
        if mask.ndim == 4 and mask.shape[1] == 1:
            return mask
        raise ValueError("掩码提示张量必须为 [B, 1, H, W] 格式。")
