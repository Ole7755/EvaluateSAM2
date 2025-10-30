"""
Entry-point to craft universal adversarial perturbations against SAM2.

Refactored to lean on the reusable wheels inspired by the cross-prompt attack
codebase while keeping the original CLI workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image

if __package__ is None:  # pragma: no cover - allow `python scripts/run_uap_attack.py`
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from sam2.build_sam import build_sam2_video_predictor

from scripts.wheels.attacks import (
    BIMAttack,
    CarliniWagnerAttack,
    FGSMAttack,
    PGDAttack,
    SAM2ForwardHelper,
)
from scripts.wheels.dataset import (
    find_frame_path,
    normalize_masks,
    normalize_object_ids,
)
from scripts.wheels.utils import (
    AttackConfig,
    AttackLogger,
    AttackSummary,
    BestWorstTracker,
    compute_perturbation_norms,
    eval_masks_numpy,
    load_mask_tensor,
    load_rgb_tensor,
    mask_probs_to_numpy,
    mask_to_binary,
    ensure_dir,
    resize_image_tensor,
    resize_mask_tensor,
    restore_image_tensor,
    save_perturbation_image,
    save_rgb_tensor,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "sam2_hiera_s.yaml"
WEIGHT_PATH = PROJECT_ROOT / "weights" / "sam2_hiera_small.pt"
DATA_ROOT = PROJECT_ROOT / "data" / "DAVIS"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
LOG_ROOT = PROJECT_ROOT / "logs"

os.environ.setdefault("SAM2_CONFIG_DIR", str(CONFIG_PATH.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply UAP attacks to SAM2 on DAVIS sequences.")
    parser.add_argument("--sequence", type=str, required=True, help="DAVIS sequence name, e.g. bear.")
    parser.add_argument("--frame-token", type=str, default="00000", help="Frame index token (default 00000).")
    parser.add_argument("--gt-label", type=int, default=None, help="Foreground label in the first-frame mask.")
    parser.add_argument("--obj-id", type=int, default=1, help="Object id for logging.")
    parser.add_argument(
        "--attack",
        type=str,
        choices=("fgsm", "pgd", "bim", "cw"),
        required=True,
        help="Attack type to run.",
    )
    parser.add_argument("--epsilon", type=float, default=0.03, help="Max perturbation (L_inf).")
    parser.add_argument("--step-size", type=float, default=0.01, help="Gradient step size.")
    parser.add_argument("--steps", type=int, default=40, help="Number of attack iterations.")
    parser.add_argument("--random-start", action="store_true", help="Enable random initialisation for PGD.")
    parser.add_argument("--input-size", type=int, default=1024, help="Resize frames to this square size.")
    parser.add_argument(
        "--keep-aspect-ratio",
        action="store_true",
        help="Keep aspect ratio when resizing (pads bottom/right).",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Threshold for probability masks.")
    parser.add_argument("--cw-confidence", type=float, default=0.0, help="C&W confidence margin.")
    parser.add_argument("--cw-binary-steps", type=int, default=5, help="C&W binary search steps.")
    parser.add_argument("--cw-lr", type=float, default=0.01, help="C&W optimiser learning rate.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device, e.g. cuda or cpu.")
    return parser.parse_args()


def ensure_prerequisites() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"SAM2 config not found: {CONFIG_PATH}")
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"SAM2 weights not found: {WEIGHT_PATH}")
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DAVIS data root missing: {DATA_ROOT}")


def build_attack(args: argparse.Namespace) -> Tuple[object, AttackConfig]:
    attack_map = {
        "fgsm": FGSMAttack,
        "pgd": PGDAttack,
        "bim": BIMAttack,
        "cw": CarliniWagnerAttack,
    }
    attack_cls = attack_map[args.attack]

    if args.attack == "cw":
        attack = attack_cls(
            epsilon=args.epsilon,
            step_size=args.cw_lr,
            steps=args.steps,
            confidence=args.cw_confidence,
            binary_steps=args.cw_binary_steps,
        )
    else:
        attack = attack_cls(
            epsilon=args.epsilon,
            step_size=args.step_size,
            steps=args.steps,
        )

    config = AttackConfig(
        attack_name=args.attack,
        epsilon=args.epsilon,
        step_size=args.step_size if args.attack != "cw" else args.cw_lr,
        steps=args.steps,
        random_start=args.random_start,
        cw_confidence=args.cw_confidence if args.attack == "cw" else None,
        cw_learning_rate=args.cw_lr if args.attack == "cw" else None,
        cw_binary_steps=args.cw_binary_steps if args.attack == "cw" else None,
    )
    return attack, config


def main() -> None:
    args = parse_args()
    ensure_prerequisites()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    sequence = args.sequence
    frame_token = args.frame_token

    rgb_dir = DATA_ROOT / "JPEGImages" / "480p" / sequence
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory missing: {rgb_dir}")
    ann_dir = DATA_ROOT / "Annotations_unsupervised" / "480p" / sequence
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation directory missing: {ann_dir}")

    frame_path = find_frame_path(rgb_dir, frame_token)
    mask_path = ann_dir / f"{frame_token}.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask missing: {mask_path}")

    # Build predictor and helper
    predictor = build_sam2_video_predictor(str(CONFIG_PATH), str(WEIGHT_PATH), device=device)
    helper = SAM2ForwardHelper(predictor, device=device)

    # Load IO tensors
    image_tensor = load_rgb_tensor(frame_path, device=device)
    mask_tensor = load_mask_tensor(mask_path, device=device)
    binary_mask = mask_to_binary(mask_tensor, label=args.gt_label)

    resized_image, resize_info = resize_image_tensor(
        image_tensor, args.input_size, keep_aspect_ratio=args.keep_aspect_ratio
    )
    resized_mask = resize_mask_tensor(binary_mask, resize_info).to(device)
    origin_hw = (int(image_tensor.shape[-2]), int(image_tensor.shape[-1]))

    clean_input = resized_image.unsqueeze(0)
    clean_output = helper.forward(clean_input, prompt_mask=resized_mask)
    clean_mask_np = mask_probs_to_numpy(clean_output.probs, resize_info, origin_hw, args.mask_threshold)
    gt_mask_numpy = binary_mask.detach().cpu().numpy() > 0.5
    clean_iou, clean_dice = eval_masks_numpy(clean_mask_np, gt_mask_numpy)

    print(f"[INFO] Clean baseline IoU={clean_iou:.6f}, Dice={clean_dice:.6f}")

    attack, attack_config = build_attack(args)

    # Prepare directories
    attack_dir = ensure_output_dirs(sequence, args.attack)
    log_dir = ensure_dir(LOG_ROOT / "uap" / sequence / args.attack)
    logger = AttackLogger(log_dir)
    tracker = BestWorstTracker(
        record_path=log_dir / "best_worst.json",
        best_dir=attack_dir / "best_cases",
        worst_dir=attack_dir / "worst_cases",
    )
    logger.save_config(attack_config)

    clean_input = clean_input.to(device)
    perturbation = attack.generate(
        helper=helper,
        clean_image=clean_input,
        target_mask=resized_mask,
        random_start=args.random_start,
    )
    adv_input = (clean_input + perturbation).clamp(0.0, 1.0)

    adv_output = helper.forward(adv_input, prompt_mask=resized_mask)
    adv_mask_np = mask_probs_to_numpy(adv_output.probs, resize_info, origin_hw, args.mask_threshold)
    adv_iou, adv_dice = eval_masks_numpy(adv_mask_np, gt_mask_numpy)

    print(f"[INFO] Adversarial IoU={adv_iou:.6f}, Dice={adv_dice:.6f}")
    delta_iou = clean_iou - adv_iou
    print(f"[INFO] ΔIoU={delta_iou:.6f}")

    perturbation_norms = compute_perturbation_norms(perturbation)

    clean_vis = restore_image_tensor(clean_input, resize_info, origin_hw).squeeze(0).detach()
    adv_vis = restore_image_tensor(adv_input, resize_info, origin_hw).squeeze(0).detach()
    perturbation_vis = restore_image_tensor(perturbation, resize_info, origin_hw).squeeze(0).detach()

    clean_image_path = attack_dir / f"{frame_token}_clean.png"
    adv_image_path = attack_dir / f"{frame_token}_adv.png"
    perturbation_image_path = attack_dir / f"{frame_token}_perturbation.png"

    save_rgb_tensor(clean_vis, clean_image_path)
    save_rgb_tensor(adv_vis, adv_image_path)
    save_perturbation_image(perturbation_vis, perturbation_image_path)

    summary = AttackSummary(
        attack_name=args.attack,
        sequence=sequence,
        frame_idx=int(frame_token),
        obj_id=args.obj_id,
        gt_label=args.gt_label,
        clean_iou=clean_iou,
        clean_dice=clean_dice,
        adv_iou=adv_iou,
        adv_dice=adv_dice,
        perturbation_norm=perturbation_norms,
    )
    logger.save_summary(
        summary,
        extra={"frame_path": str(frame_path), "mask_path": str(mask_path)},
    )
    logger.save_tensor(perturbation.squeeze(0), name=f"{sequence}_{args.attack}_uap")

    artifacts = {
        "clean": clean_image_path,
        "adv": adv_image_path,
        "perturbation": perturbation_image_path,
    }
    tracker_update = tracker.update(summary, artifacts, attack_name=args.attack)
    if tracker_update.get("skipped"):
        print(
            "[INFO] BestWorstTracker skipped update because clean_iou "
            f"{clean_iou:.4f} < threshold {tracker.min_clean_iou:.2f}"
        )
    else:
        if tracker_update["best"]:
            print(f"[INFO] Updated best case under {attack_dir / 'best_cases'}")
        if tracker_update["worst"]:
            print(f"[INFO] Updated worst case under {attack_dir / 'worst_cases'}")

    mask_dir = ensure_dir(attack_dir / "masks")
    Image.fromarray((clean_mask_np.astype(float) * 255).astype("uint8")).save(mask_dir / f"{frame_token}_clean.png")
    Image.fromarray((adv_mask_np.astype(float) * 255).astype("uint8")).save(mask_dir / f"{frame_token}_adv.png")
    Image.fromarray((gt_mask_numpy.astype(float) * 255).astype("uint8")).save(mask_dir / f"{frame_token}_gt.png")

    report: Dict[str, float] = {
        "clean_iou": clean_iou,
        "clean_dice": clean_dice,
        "adv_iou": adv_iou,
        "adv_dice": adv_dice,
        "uap_linf": perturbation_norms["linf"],
        "uap_l2": perturbation_norms["l2"],
        "uap_l1": perturbation_norms["l1"],
        "delta_iou": delta_iou,
    }
    report_path = attack_dir / f"{frame_token}_{args.attack}_metrics.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[INFO] Attack finished. Key metrics:")
    for key, value in report.items():
        print(f"  {key}: {value:.6f}")


def ensure_output_dirs(sequence: str, attack_name: str) -> Path:
    attack_dir = ensure_dir(OUTPUT_ROOT / sequence / attack_name)
    ensure_dir(LOG_ROOT / "uap" / sequence / attack_name)
    return attack_dir


if __name__ == "__main__":
    main()
