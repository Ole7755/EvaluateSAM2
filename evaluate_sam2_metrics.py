from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


def load_mask(path: Path, threshold: float = 0.0, label: Optional[int] = None) -> np.ndarray:
    """Load a mask image and return a binary array."""
    array = np.array(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    if label is not None:
        mask = array == label
    else:
        mask = array > threshold
    return mask.astype(bool)


def compute_iou_and_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    """Compute IoU and Dice between two boolean masks."""
    if pred_mask.shape != gt_mask.shape:
        raise ValueError("Prediction and ground-truth masks must share the same shape.")

    intersection = np.logical_and(pred_mask, gt_mask).sum(dtype=np.float64)
    union = np.logical_or(pred_mask, gt_mask).sum(dtype=np.float64)
    pred_area = pred_mask.sum(dtype=np.float64)
    gt_area = gt_mask.sum(dtype=np.float64)

    if union == 0:
        iou = 1.0
    else:
        iou = float(intersection / union)

    if pred_area + gt_area == 0:
        dice = 1.0
    else:
        dice = float(2.0 * intersection / (pred_area + gt_area))

    return iou, dice


def extract_frame_token(stem: str) -> str:
    """Return the frame token (00000) from names like 00000 or 00000_id1."""
    return stem.split("_", maxsplit=1)[0]


def evaluate_directory(
    pred_dir: Path,
    gt_dir: Path,
    obj_id: Optional[int] = None,
    gt_label: Optional[int] = None,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """Evaluate predictions stored in pred_dir against gt_dir."""
    pred_paths = sorted(pred_dir.glob("*.png"))
    if obj_id is not None:
        suffix = f"_id{obj_id}"
        pred_paths = [path for path in pred_paths if suffix in path.stem]

    if not pred_paths:
        raise FileNotFoundError("No prediction PNG files matched the criteria.")

    per_frame: List[Tuple[str, float, float]] = []

    for pred_path in pred_paths:
        frame_token = extract_frame_token(pred_path.stem)
        gt_path = gt_dir / f"{frame_token}.png"
        if not gt_path.exists():
            print(f"[WARN] Ground truth missing for frame {frame_token}, skipping.")
            continue

        pred_mask = load_mask(pred_path, threshold=threshold)
        gt_mask = load_mask(gt_path, threshold=threshold, label=gt_label)
        iou, dice = compute_iou_and_dice(pred_mask, gt_mask)
        per_frame.append((frame_token, iou, dice))
        print(f"{frame_token}: IoU={iou:.4f}, Dice={dice:.4f}")

    if not per_frame:
        raise RuntimeError("No frames were evaluated. Check prediction and GT directories.")

    iou_values = [item[1] for item in per_frame]
    dice_values = [item[2] for item in per_frame]

    mean_iou = float(np.mean(iou_values))
    mean_dice = float(np.mean(dice_values))

    print(f"\nMean IoU: {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")

    return {
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "frames_evaluated": len(per_frame),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAM2 masks with mIoU and Dice.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory of predicted PNG masks.")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory of ground-truth PNG masks.")
    parser.add_argument(
        "--obj-id",
        type=int,
        default=None,
        help="Only evaluate predictions containing _id{obj_id} (set None for all).",
    )
    parser.add_argument(
        "--gt-label",
        type=int,
        default=None,
        help="Ground-truth label value for the target object (defaults to >0 binary).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Pixel threshold for binarising masks when gt_label is not provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_directory(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        obj_id=args.obj_id,
        gt_label=args.gt_label,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
