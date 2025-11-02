"""
Utility helpers to construct SAM2 image predictors without relying on the original
`build_sam` entry points (which assume the package is installed outside the
repository root).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sam2.sam2_image_predictor import SAM2ImagePredictor

_POSTPROCESSING_OVERRIDES: dict[str, Any] = {
    "model": {
        "sam_mask_decoder_extra_args": {
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        }
    }
}


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = getattr(load_result, "missing_keys", ())
    unexpected = getattr(load_result, "unexpected_keys", ())
    if missing:
        raise RuntimeError(f"Checkpoint missing keys: {sorted(missing)}")
    if unexpected:
        raise RuntimeError(f"Checkpoint has unexpected keys: {sorted(unexpected)}")


def load_image_predictor(
    config_path: Path,
    checkpoint_path: Path,
    *,
    device: str = "cuda",
    mask_threshold: float = 0.0,
    apply_postprocessing: bool = True,
) -> SAM2ImagePredictor:
    """
    Build a SAM2 image predictor given a config file and checkpoint.
    """
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)

    cfg = OmegaConf.load(config_path)
    if apply_postprocessing:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(_POSTPROCESSING_OVERRIDES))
    OmegaConf.resolve(cfg)

    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, checkpoint_path)
    model.to(device)
    model.eval()

    predictor = SAM2ImagePredictor(model, mask_threshold=mask_threshold)
    predictor.model = model  # 确保后续访问到同一模型实例
    return predictor
