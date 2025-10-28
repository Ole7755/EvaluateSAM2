"""
在 bear 序列上使用 FGSM 训练通用对抗补丁 (UAP)，并在其余序列上批量评估。

脚本流程：
1. 加载 bear 首帧样本，使用 FGSM 训练出补丁；
2. 自动发现 DAVIS 其余序列，或通过命令行指定测试列表；
3. 在测试序列上评估补丁效果，并将所有指标汇总到单个 JSON 文件。

示例用法：
    python3 scripts/run_bear_fgsm_uap_pipeline.py \
        --output logs/uap/fgsm_bear/pipeline_result.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import sys
import torch

from sam2.build_sam import build_sam2_video_predictor


if __package__ is None:  # 兼容直接以 python 执行脚本
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.run_uap_attack import (  # noqa: E402
    find_frame_path,
    normalize_masks,
    normalize_object_ids,
)
from scripts.sam2_attack_utils import (  # noqa: E402
    compute_perturbation_norms,
    eval_masks_numpy,
    ensure_dir,
    load_mask_tensor,
    mask_to_binary,
    save_perturbation_image,
)
from scripts.uap_attacks import SAM2ForwardHelper
from scripts.uap_patch_trainer import (
    EvaluationSummary,
    StepRecord,
    UAPSample,
    UniversalPatchTrainer,
    load_uap_samples,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "sam2_hiera_s.yaml"
WEIGHT_PATH = PROJECT_ROOT / "weights" / "sam2_hiera_small.pt"
DATA_ROOT = PROJECT_ROOT / "data" / "DAVIS"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "uap" / "fgsm_bear"
LOG_ROOT = PROJECT_ROOT / "logs" / "uap" / "fgsm_bear"

os.environ.setdefault("SAM2_CONFIG_DIR", str(CONFIG_PATH.parent))


def ensure_prerequisites() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"未找到配置文件：{CONFIG_PATH}")
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"未找到权重文件：{WEIGHT_PATH}")
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"未找到 DAVIS 数据目录：{DATA_ROOT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 bear 上训练 FGSM UAP，并在其他序列评估。")
    parser.add_argument("--frame-token", type=str, default="00000", help="首帧编号。")
    parser.add_argument(
        "--mask-subdir",
        type=str,
        default="Annotations_unsupervised",
        help="掩码所在子目录。",
    )
    parser.add_argument("--gt-label", type=int, default=None, help="多标签掩码的目标标签。")
    parser.add_argument("--epsilon", type=float, default=0.03, help="L_inf 扰动半径。")
    parser.add_argument("--step-size", type=float, default=0.03, help="FGSM 更新步长。")
    parser.add_argument("--input-size", type=int, default=1024, help="输入缩放尺寸。")
    parser.add_argument("--keep-aspect-ratio", action="store_true", help="保持宽高比缩放。")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="概率转掩码阈值。")
    parser.add_argument("--loss-type", choices=("dice", "bce"), default="dice", help="训练损失类型。")
    parser.add_argument("--obj-id", type=int, default=1, help="目标对象 ID，用于官方 baseline。")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备。")
    parser.add_argument(
        "--test-sequences",
        type=str,
        default="",
        help="逗号分隔的测试序列。留空则自动使用除 bear 以外的所有序列。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="实验结果保存路径（JSON）。",
    )
    parser.add_argument(
        "--patch-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="补丁可视化与张量的保存目录。",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_ROOT,
        help="日志目录，用于存放训练历史等文件。",
    )
    return parser.parse_args()


def discover_sequences(exclude: Sequence[str], resolution: str = "480p") -> List[str]:
    root = DATA_ROOT / "JPEGImages" / resolution
    if not root.exists():
        raise FileNotFoundError(f"RGB 目录不存在：{root}")
    excluded = set(exclude)
    sequences: List[str] = []
    for item in sorted(root.iterdir()):
        if item.is_dir():
            name = item.name
            if name not in excluded:
                sequences.append(name)
    return sequences


def to_sequence_list(raw: str) -> List[str]:
    if not raw:
        return []
    result: List[str] = []
    seen = set()
    for token in raw.split(","):
        name = token.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def load_samples_safe(
    sequences: Sequence[str],
    frame_token: str,
    mask_subdir: str,
    device: torch.device,
    input_size: int,
    keep_aspect: bool,
    gt_label: Optional[int],
) -> tuple[List[UAPSample], List[Dict[str, str]]]:
    samples: List[UAPSample] = []
    skipped: List[Dict[str, str]] = []
    for seq in sequences:
        try:
            loaded = load_uap_samples(
                sequences=[seq],
                frame_token=frame_token,
                data_root=DATA_ROOT,
                mask_subdir=mask_subdir,
                device=device,
                input_size=input_size,
                keep_aspect_ratio=keep_aspect,
                gt_label=gt_label,
            )
        except FileNotFoundError as exc:
            skipped.append({"sequence": seq, "reason": str(exc)})
            continue
        if loaded:
            samples.extend(loaded)
        else:
            skipped.append({"sequence": seq, "reason": "未加载到首帧样本"})
    return samples, skipped


def summary_to_dict(summary: EvaluationSummary) -> Dict[str, object]:
    aggregate = summary.aggregate.to_dict() if summary.aggregate is not None else None
    samples = [
        {
            "split": item.split,
            "sequence": item.sequence,
            "frame_token": item.frame_token,
            "clean_iou": item.clean_iou,
            "clean_dice": item.clean_dice,
            "adv_iou": item.adv_iou,
            "adv_dice": item.adv_dice,
            "delta_iou": item.delta_iou,
            "delta_dice": item.delta_dice,
            "frame_path": item.frame_path,
            "mask_path": item.mask_path,
        }
        for item in summary.samples
    ]
    return {"aggregate": aggregate, "samples": samples}


def history_to_list(history: Sequence[StepRecord]) -> List[Dict[str, object]]:
    return [
        {"step": record.step, "loss": record.loss, "per_sample_losses": record.per_sample_losses}
        for record in history
    ]


def compute_clean_baseline(
    predictor,
    sequence: str,
    frame_token: str,
    obj_id: int,
    mask_subdir: str,
    gt_label: Optional[int],
    mask_threshold: float,
    device: torch.device,
) -> Dict[str, object]:
    rgb_dir = DATA_ROOT / "JPEGImages" / "480p" / sequence
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB 目录不存在：{rgb_dir}")

    mask_dir = DATA_ROOT / mask_subdir / "480p" / sequence
    if not mask_dir.exists():
        raise FileNotFoundError(f"掩码目录不存在：{mask_dir}")

    frame_path = find_frame_path(rgb_dir, frame_token)
    mask_path = mask_dir / f"{frame_token}.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"未找到掩码：{mask_path}")

    raw_mask = load_mask_tensor(mask_path, device=device)
    binary_mask = mask_to_binary(raw_mask, label=gt_label)
    gt_mask_np = binary_mask.detach().cpu().numpy().astype(bool)

    state = predictor.init_state(rgb_dir.as_posix())
    try:
        initial = predictor.add_new_mask(
            state,
            frame_idx=0,
            obj_id=obj_id,
            mask=binary_mask,
        )
    except TypeError:
        initial = predictor.add_new_mask(
            state,
            frame_idx=0,
            obj_id=obj_id,
            mask_tensor=binary_mask,
        )

    _, object_ids, masks = initial
    object_ids_list = normalize_object_ids(object_ids)
    masks_list = normalize_masks(masks)

    clean_mask = None
    for oid, mask in zip(object_ids_list, masks_list):
        if oid == obj_id:
            clean_mask = mask
            break
    if clean_mask is None:
        raise RuntimeError(f"未找到 obj_id={obj_id} 的预测掩码。")

    clean_mask_np = (clean_mask.squeeze() > mask_threshold).detach().cpu().numpy().astype(bool)
    clean_iou, clean_dice = eval_masks_numpy(clean_mask_np, gt_mask_np)

    return {
        "sequence": sequence,
        "clean_iou": float(clean_iou),
        "clean_dice": float(clean_dice),
        "frame_path": frame_path.as_posix(),
        "mask_path": mask_path.as_posix(),
    }


def main() -> None:
    args = parse_args()
    ensure_prerequisites()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_sequence = "bear"
    predictor = build_sam2_video_predictor(CONFIG_PATH.name, WEIGHT_PATH.as_posix())
    helper = SAM2ForwardHelper(predictor, device=device)

    train_samples, skipped_train = load_samples_safe(
        sequences=[train_sequence],
        frame_token=args.frame_token,
        mask_subdir=args.mask_subdir,
        device=device,
        input_size=args.input_size,
        keep_aspect=args.keep_aspect_ratio,
        gt_label=args.gt_label,
    )
    if not train_samples:
        raise RuntimeError(f"未能加载 {train_sequence} 的训练样本：{skipped_train}")

    trainer = UniversalPatchTrainer(
        helper=helper,
        samples=train_samples,
        epsilon=args.epsilon,
        loss_type=args.loss_type,
        mask_threshold=args.mask_threshold,
    )
    patch, history = trainer.train(
        attack="fgsm",
        steps=1,
        step_size=args.step_size,
        random_start=False,
    )
    train_summary = trainer.evaluate(patch, train_samples, split="train")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    patch_dir = ensure_dir(args.patch_dir)
    patch_tensor_path = patch_dir / f"bear_fgsm_patch_{timestamp}.pt"
    torch.save(patch.detach().cpu(), patch_tensor_path)

    patch_image_path = patch_dir / f"bear_fgsm_patch_{timestamp}.png"
    save_perturbation_image(patch.squeeze(0), patch_image_path)

    patch_norms = compute_perturbation_norms(patch)

    try:
        train_clean_baseline = compute_clean_baseline(
            predictor=predictor,
            sequence=train_sequence,
            frame_token=args.frame_token,
            obj_id=args.obj_id,
            mask_subdir=args.mask_subdir,
            gt_label=args.gt_label,
            mask_threshold=args.mask_threshold,
            device=device,
        )
    except Exception as exc:
        train_clean_baseline = {"sequence": train_sequence, "error": str(exc)}

    if args.test_sequences:
        test_sequences = to_sequence_list(args.test_sequences)
    else:
        test_sequences = discover_sequences(exclude=[train_sequence])

    eval_samples, skipped_eval = load_samples_safe(
        sequences=test_sequences,
        frame_token=args.frame_token,
        mask_subdir=args.mask_subdir,
        device=device,
        input_size=args.input_size,
        keep_aspect=args.keep_aspect_ratio,
        gt_label=args.gt_label,
    )
    if not eval_samples:
        raise RuntimeError("未能加载任何测试序列，请检查数据集路径或参数设置。")

    eval_trainer = UniversalPatchTrainer(
        helper=helper,
        samples=eval_samples,
        epsilon=args.epsilon,
        loss_type=args.loss_type,
        mask_threshold=args.mask_threshold,
    )
    eval_summary = eval_trainer.evaluate(patch, eval_samples, split="test")

    eval_sequences_loaded = sorted({sample.sequence for sample in eval_samples})

    test_clean_baseline: List[Dict[str, object]] = []
    for seq in eval_sequences_loaded:
        try:
            baseline = compute_clean_baseline(
                predictor=predictor,
                sequence=seq,
                frame_token=args.frame_token,
                obj_id=args.obj_id,
                mask_subdir=args.mask_subdir,
                gt_label=args.gt_label,
                mask_threshold=args.mask_threshold,
                device=device,
            )
            test_clean_baseline.append(baseline)
        except Exception as exc:
            skipped_eval.append({"sequence": seq, "reason": f"baseline_failed: {exc}"})

    ensure_dir(args.output.parent)
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attack": "fgsm",
        "train_sequence": train_sequence,
        "frame_token": args.frame_token,
        "mask_subdir": args.mask_subdir,
        "gt_label": args.gt_label,
        "obj_id": args.obj_id,
        "epsilon": args.epsilon,
        "step_size": args.step_size,
        "input_size": args.input_size,
        "keep_aspect_ratio": args.keep_aspect_ratio,
        "mask_threshold": args.mask_threshold,
        "loss_type": args.loss_type,
        "device": str(device),
        "patch_tensor_path": patch_tensor_path.as_posix(),
        "patch_image_path": patch_image_path.as_posix(),
        "patch_norms": patch_norms,
        "train_metrics_surrogate": summary_to_dict(train_summary),
        "train_clean_official": train_clean_baseline,
        "test_metrics_surrogate": summary_to_dict(eval_summary),
        "test_clean_official": test_clean_baseline,
        "skipped_sequences": skipped_train + skipped_eval,
        "history": history_to_list(history),
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    agg = eval_summary.aggregate
    if agg:
        print(
            f"[INFO] 测试序列平均 IoU {agg.mean_adv_iou:.4f} (Δ {agg.mean_delta_iou:.4f})，"
            f"平均 Dice {agg.mean_adv_dice:.4f} (Δ {agg.mean_delta_dice:.4f})"
        )
    print(f"[INFO] 实验结果已保存到 {args.output}")


if __name__ == "__main__":
    main()
