from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor


def load_binary_mask(mask_path: Path) -> torch.Tensor:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = (mask > 0).astype(np.float32)
    return torch.from_numpy(binary)


def normalize_object_ids(object_ids: Iterable) -> list[int]:
    if isinstance(object_ids, torch.Tensor):
        if object_ids.ndim == 0:
            return [int(object_ids.item())]
        return [int(item) for item in object_ids.cpu().tolist()]
    return [int(item) for item in object_ids]


def normalize_masks(masks: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
    if isinstance(masks, torch.Tensor):
        if masks.ndim == 2:
            return [masks]
        return [masks[idx] for idx in range(masks.shape[0])]
    return [torch.as_tensor(mask) for mask in masks]


check_point = "sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
predictor = build_sam2_video_predictor(model_cfg, check_point)

seq = "bear"
rgb_dir = Path("DAVIS/JPEGImages/480p/bear")
ann_dir = Path("DAVIS/Annotations_unsupervised/480p/bear")

state = predictor.init_state(rgb_dir.as_posix())

mask_tensor = load_binary_mask(ann_dir / "00000.png")

if hasattr(predictor, "add_new_mask"):
    try:
        initial = predictor.add_new_mask(
            state,
            frame_idx=0,
            obj_id=1,
            mask=mask_tensor,
        )
    except TypeError:
        initial = predictor.add_new_mask(
            state,
            frame_idx=0,
            obj_id=1,
            mask_tensor=mask_tensor,
        )
else:
    raise RuntimeError("当前 SAM2 版本缺少 add_new_mask 接口，无法直接使用掩码提示。")

frame_idx, object_ids, masks = initial
object_ids_list = normalize_object_ids(object_ids)
masks_list = normalize_masks(masks)

out_dir = Path("outputs/bear")
out_dir.mkdir(parents=True, exist_ok=True)

for obj_id, mask in zip(object_ids_list, masks_list):
    mask_u8 = (mask > 0.5).to(torch.uint8).cpu().numpy() * 255
    Image.fromarray(mask_u8).save(out_dir / f"{frame_idx:05d}_id{obj_id}.png")

for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
    object_ids_list = normalize_object_ids(object_ids)
    masks_list = normalize_masks(masks)
    for obj_id, mask in zip(object_ids_list, masks_list):
        mask_u8 = (mask > 0.5).to(torch.uint8).cpu().numpy() * 255
        Image.fromarray(mask_u8).save(out_dir / f"{frame_idx:05d}_id{obj_id}.png")
