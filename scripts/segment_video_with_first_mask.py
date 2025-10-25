from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "sam2_hiera_s.yaml"
WEIGHT_PATH = PROJECT_ROOT / "weights" / "sam2_hiera_small.pt"
DATA_ROOT = PROJECT_ROOT / "data" / "DAVIS"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def load_mask_tensor(mask_path: Path) -> torch.Tensor:
    mask = np.array(Image.open(mask_path))
    print(mask.dtype)
    print(f"掩码最大值: {mask.max()}, 掩码最小值: {mask.min()}")
    if mask.ndim == 3:
        print(f"mask.ndim = {mask.ndim}，使用第 0 通道")
        mask = mask[..., 0]
    return torch.from_numpy(mask.astype(np.int64))


def extract_instance_masks(mask_tensor: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
    mask_np = mask_tensor.numpy()
    unique_labels = [int(v) for v in np.unique(mask_np) if v != 0]
    unique_labels.sort()
    if not unique_labels:
        raise ValueError("未在首帧掩码中检测到任何前景标签。")

    print(f"检测到的对象标签: {unique_labels}")
    instances: list[tuple[int, torch.Tensor]] = []
    for obj_index, label_value in enumerate(unique_labels, start=1):
        binary = (mask_tensor == label_value).to(torch.float32)
        instances.append((obj_index, binary))
        area = int(binary.sum().item())
        print(f"obj_id={obj_index} 对应标签 {label_value}，首帧像素面积 {area}")
    return instances


def save_masks(
    frame_idx: int,
    object_ids: Sequence[int],
    masks: Sequence[torch.Tensor],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for obj_id, mask_tensor in zip(object_ids, masks):
        mask_u8 = (mask_tensor > 0.5).to(torch.uint8).cpu().numpy() * 255
        save_path = output_dir / f"{frame_idx:05d}_id{obj_id}.png"
        Image.fromarray(mask_u8).save(save_path)


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


def sample_points_from_mask(
    mask: torch.Tensor, num_foreground: int = 2, num_background: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    mask_np = mask.numpy()
    ys, xs = np.nonzero(mask_np)
    if ys.size == 0:
        raise ValueError("The provided mask is empty; cannot derive prompts.")

    center = np.array([[xs.mean(), ys.mean()]], dtype=np.float32)
    fg_points = [center[0]]

    extremal_indices = [
        np.argmin(xs),
        np.argmax(xs),
        np.argmin(ys),
        np.argmax(ys),
    ]
    for idx in extremal_indices:
        fg_points.append([xs[idx], ys[idx]])
        if len(fg_points) >= num_foreground:
            break
    fg_points = np.array(fg_points, dtype=np.float32)

    height, width = mask_np.shape
    bg_candidates = np.array(
        [
            [1, 1],
            [width - 2, 1],
            [1, height - 2],
            [width - 2, height - 2],
            [width / 2, 1],
            [width / 2, height - 2],
            [1, height / 2],
            [width - 2, height / 2],
        ],
        dtype=np.float32,
    )
    bg_points = []
    for point in bg_candidates:
        x, y = int(round(point[0])), int(round(point[1]))
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)
        if mask_np[y, x] == 0:
            bg_points.append([float(x), float(y)])
        if len(bg_points) >= num_background:
            break

    if not bg_points:
        raise ValueError("Unable to find background points outside the mask.")

    points = np.concatenate([fg_points, np.array(bg_points, dtype=np.float32)], axis=0)
    labels = np.concatenate(
        [np.ones(len(fg_points), dtype=np.int32), np.zeros(len(bg_points), dtype=np.int32)]
    )
    return points, labels


def main() -> None:
    sequence = "walking"
    rgb_dir = DATA_ROOT / "JPEGImages" / "480p" / sequence
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB frames not found at {rgb_dir}")

    mask_path = DATA_ROOT / "Annotations_unsupervised" / "480p" / sequence / "00000.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"First-frame mask not found at {mask_path}")

    output_dir = OUTPUT_ROOT / sequence

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"未找到 SAM2 配置文件：{CONFIG_PATH}")
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"未找到 SAM2 权重文件：{WEIGHT_PATH}")

    predictor = build_sam2_video_predictor(CONFIG_PATH.as_posix(), WEIGHT_PATH.as_posix())
    state = predictor.init_state(rgb_dir.as_posix())

    first_mask = load_mask_tensor(mask_path)
    instance_masks = extract_instance_masks(first_mask)
    # 只保留前两个对象
    instance_masks = instance_masks[:2]

    last_result = None
    for index, (obj_id, instance_mask) in enumerate(instance_masks):
        if hasattr(predictor, "add_new_mask"):
            try:
                result = predictor.add_new_mask(
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask=instance_mask,
                )
            except TypeError:
                result = predictor.add_new_mask(
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask_tensor=instance_mask,
                )
        else:
            points, labels = sample_points_from_mask(instance_mask)
            result = predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=obj_id,
                points=points,
                labels=labels,
                normalize_coords=False,
                clear_old_points=(index == 0),
            )
        last_result = result

    if last_result is None:
        raise RuntimeError("Failed to initialize any objects in the predictor.")

    frame_idx, object_ids, masks = last_result
    object_ids_list = normalize_object_ids(object_ids)
    masks_list = normalize_masks(masks)
    save_masks(frame_idx, object_ids_list, masks_list, output_dir)

    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        object_ids_list = normalize_object_ids(object_ids)
        masks_list = normalize_masks(masks)
        save_masks(frame_idx, object_ids_list, masks_list, output_dir)


if __name__ == "__main__":
    main()
