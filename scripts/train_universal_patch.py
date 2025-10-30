"""
批量训练 SAM2 通用对抗补丁的命令行脚本。

支持攻击类型：FGSM / PGD / BIM / C&W，
并在训练/验证序列上输出 IoU / Dice 等指标及可视化文件。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image

if __package__ is None:  # pragma: no cover - 兼容直接执行
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from sam2.build_sam import build_sam2_video_predictor

from scripts.wheels.attacks import SAM2ForwardHelper
from scripts.wheels.trainer import (
    AggregateMetrics,
    SampleEvaluation,
    StepRecord,
    UAPSample,
    UniversalPatchTrainer,
    load_uap_samples,
    match_sample,
)
from scripts.wheels.utils import (
    AttackConfig,
    AttackLogger,
    AttackSummary,
    compute_perturbation_norms,
    ensure_dir,
    mask_probs_to_numpy,
    restore_image_tensor,
    save_perturbation_image,
    save_rgb_tensor,
)

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "sam2_hiera_s.yaml"
WEIGHT_PATH = PROJECT_ROOT / "weights" / "sam2_hiera_small.pt"
DATA_ROOT = PROJECT_ROOT / "data" / "DAVIS"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
LOG_ROOT = PROJECT_ROOT / "logs"

os.environ.setdefault("SAM2_CONFIG_DIR", str(CONFIG_PATH.parent))


def parse_sequence_list(raw: str) -> List[str]:
    """将逗号分隔的序列字符串解析为有序去重列表。"""
    if not raw:
        return []
    items: List[str] = []
    seen = set()
    for part in raw.split(","):
        name = part.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        items.append(name)
    return items


def safe_int(token: str) -> int:
    try:
        return int(token)
    except ValueError:
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练通用对抗补丁 (UAP)。")
    parser.add_argument("--attack", choices=("fgsm", "pgd", "bim", "cw"), required=True, help="选择攻击类型。")
    parser.add_argument("--train-sequences", type=str, required=True, help="用于训练补丁的 DAVIS 序列列表，逗号分隔。")
    parser.add_argument("--val-sequences", type=str, default="", help="用于验证补丁的序列列表，逗号分隔。")
    parser.add_argument("--frame-token", type=str, default="00000", help="首帧编号，默认 00000。")
    parser.add_argument(
        "--mask-subdir",
        type=str,
        default="Annotations_unsupervised",
        help="掩码所在子目录（默认无监督标签）。",
    )
    parser.add_argument("--gt-label", type=int, default=None, help="若为多标签掩码，可指定目标标签。")
    parser.add_argument("--obj-id", type=int, default=1, help="记录用的对象编号。")
    parser.add_argument("--epsilon", type=float, default=0.03, help="L_inf 扰动半径。")
    parser.add_argument("--step-size", type=float, default=0.01, help="单步更新步长。")
    parser.add_argument("--steps", type=int, default=40, help="PGD/BIM/C&W 的迭代步数。")
    parser.add_argument("--random-start", action="store_true", help="PGD 是否使用随机初始补丁。")
    parser.add_argument("--input-size", type=int, default=1024, help="图像统一缩放至该尺寸。")
    parser.add_argument("--keep-aspect-ratio", action="store_true", help="保持宽高比缩放并补零。")
    parser.add_argument("--loss-type", choices=("dice", "bce"), default="dice", help="补丁训练使用的损失函数。")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="概率转二值掩码的阈值。")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备（cuda 或 cpu）。")
    parser.add_argument("--cw-confidence", type=float, default=0.0, help="C&W 置信度常数。")
    parser.add_argument("--cw-binary-steps", type=int, default=5, help="C&W 二分搜索步数。")
    parser.add_argument("--cw-lr", type=float, default=0.01, help="C&W 优化步长（Adam 学习率）。")
    parser.add_argument("--visual-limit", type=int, default=5, help="每个数据集保存的可视化样本上限；设置为 -1 表示全部。")
    parser.add_argument("--no-visuals", action="store_true", help="不生成图像和掩码可视化。")
    return parser.parse_args()


def ensure_prerequisites() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"未找到配置文件：{CONFIG_PATH}")
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"未找到权重文件：{WEIGHT_PATH}")
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"未找到 DAVIS 数据目录：{DATA_ROOT}")


def write_metrics_csv(path: Path, metrics: Sequence[SampleEvaluation]) -> None:
    ensure_dir(path.parent)
    headers = [
        "split",
        "sequence",
        "frame_token",
        "clean_iou",
        "clean_dice",
        "adv_iou",
        "adv_dice",
        "delta_iou",
        "delta_dice",
        "frame_path",
        "mask_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for item in metrics:
            writer.writerow(
                [
                    item.split,
                    item.sequence,
                    item.frame_token,
                    f"{item.clean_iou:.6f}",
                    f"{item.clean_dice:.6f}",
                    f"{item.adv_iou:.6f}",
                    f"{item.adv_dice:.6f}",
                    f"{item.delta_iou:.6f}",
                    f"{item.delta_dice:.6f}",
                    item.frame_path,
                    item.mask_path,
                ]
            )


def history_to_list(history: Sequence[StepRecord]) -> List[Dict[str, object]]:
    return [
        {"step": record.step, "loss": record.loss, "per_sample_losses": record.per_sample_losses}
        for record in history
    ]


def summary_to_dict(summary: Optional[AggregateMetrics]) -> Optional[Dict[str, object]]:
    return summary.to_dict() if summary else None


def save_training_history(path: Path, history: Sequence[StepRecord]) -> None:
    ensure_dir(path.parent)
    payload = history_to_list(history)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_aggregate_json(path: Path, aggregates: Sequence[AggregateMetrics]) -> None:
    ensure_dir(path.parent)
    payload = [item.to_dict() for item in aggregates]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_visuals(
    trainer: UniversalPatchTrainer,
    patch: torch.Tensor,
    samples: Sequence[UAPSample],
    output_dir: Path,
    split: str,
    limit: Optional[int],
) -> None:
    ensure_dir(output_dir)
    selected = samples if limit is None or limit < 0 else samples[:limit]
    for sample in selected:
        clean = sample.resized_image.unsqueeze(0)
        adv = torch.clamp(clean + patch, 0.0, 1.0)
        clean_out = trainer.helper.forward(clean, prompt_mask=sample.resized_mask)
        adv_out = trainer.helper.forward(adv, prompt_mask=sample.resized_mask)

        clean_mask = mask_probs_to_numpy(clean_out.probs, sample.resize_info, sample.orig_hw, trainer.mask_threshold)
        adv_mask = mask_probs_to_numpy(adv_out.probs, sample.resize_info, sample.orig_hw, trainer.mask_threshold)

        clean_img = restore_image_tensor(clean, sample.resize_info, sample.orig_hw).squeeze(0).detach()
        adv_img = restore_image_tensor(adv, sample.resize_info, sample.orig_hw).squeeze(0).detach()

        base = f"{sample.sequence}_{sample.frame_token}_{split}"
        save_rgb_tensor(clean_img, output_dir / f"{base}_clean.png")
        save_rgb_tensor(adv_img, output_dir / f"{base}_adv.png")
        Image.fromarray((clean_mask.astype(float) * 255).astype("uint8")).save(output_dir / f"{base}_clean_mask.png")
        Image.fromarray((adv_mask.astype(float) * 255).astype("uint8")).save(output_dir / f"{base}_adv_mask.png")


def log_per_sample_summaries(
    logger: AttackLogger,
    attack_name: str,
    patch: torch.Tensor,
    samples: Sequence[UAPSample],
    evaluations: Sequence[SampleEvaluation],
    obj_id: int,
    gt_label: Optional[int],
) -> None:
    eval_map = {(item.sequence, item.frame_token): item for item in evaluations}
    patch_norms = compute_perturbation_norms(patch)
    for sample in samples:
        key = (sample.sequence, sample.frame_token)
        if key not in eval_map:
            continue
        metrics = eval_map[key]
        summary = AttackSummary(
            attack_name=attack_name,
            sequence=sample.sequence,
            frame_idx=safe_int(sample.frame_token),
            obj_id=obj_id,
            gt_label=gt_label,
            clean_iou=metrics.clean_iou,
            clean_dice=metrics.clean_dice,
            adv_iou=metrics.adv_iou,
            adv_dice=metrics.adv_dice,
            perturbation_norm=patch_norms,
        )
        logger.save_summary(
            summary,
            extra={
                "frame_path": sample.frame_path.as_posix(),
                "mask_path": sample.mask_path.as_posix(),
            },
        )


def main() -> None:
    args = parse_args()
    ensure_prerequisites()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_sequences = parse_sequence_list(args.train_sequences)
    val_sequences = parse_sequence_list(args.val_sequences)

    predictor = build_sam2_video_predictor(CONFIG_PATH.name, WEIGHT_PATH.as_posix())
    helper = SAM2ForwardHelper(predictor, device=device)

    train_samples = load_uap_samples(
        sequences=train_sequences,
        frame_token=args.frame_token,
        data_root=DATA_ROOT,
        mask_subdir=args.mask_subdir,
        device=device,
        input_size=args.input_size,
        keep_aspect_ratio=args.keep_aspect_ratio,
        gt_label=args.gt_label,
    )
    if not train_samples:
        raise RuntimeError("训练序列为空，无法继续。")

    val_samples = load_uap_samples(
        sequences=val_sequences,
        frame_token=args.frame_token,
        data_root=DATA_ROOT,
        mask_subdir=args.mask_subdir,
        device=device,
        input_size=args.input_size,
        keep_aspect_ratio=args.keep_aspect_ratio,
        gt_label=args.gt_label,
    )

    trainer = UniversalPatchTrainer(
        helper=helper,
        samples=train_samples,
        epsilon=args.epsilon,
        loss_type=args.loss_type,
        mask_threshold=args.mask_threshold,
    )

    patch, history = trainer.train(
        steps=args.steps,
        step_size=args.step_size,
        random_start=args.random_start,
        attack_type=args.attack,
        cw_confidence=args.cw_confidence,
        cw_binary_steps=args.cw_binary_steps,
        cw_lr=args.cw_lr,
    )

    train_eval = trainer.evaluate(patch, train_samples, split="train")
    val_eval = trainer.evaluate(patch, val_samples, split="val") if val_samples else None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    attack_output_root = ensure_dir(OUTPUT_ROOT / "uap" / args.attack)
    visual_root = attack_output_root / "visuals"
    log_dir = ensure_dir(LOG_ROOT / "uap" / args.attack)

    patch_image_path = attack_output_root / f"{args.attack}_patch.png"
    save_perturbation_image(patch.squeeze(0), patch_image_path)

    logger = AttackLogger(log_dir)
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
    logger.save_config(config)
    logger.save_tensor(patch.squeeze(0), name=f"{args.attack}_uap_patch")

    history_path = log_dir / f"{args.attack}_history_{timestamp}.json"
    save_training_history(history_path, history)

    train_csv = log_dir / f"{args.attack}_train_metrics_{timestamp}.csv"
    write_metrics_csv(train_csv, train_eval.samples)
    if val_eval and val_eval.samples:
        val_csv = log_dir / f"{args.attack}_val_metrics_{timestamp}.csv"
        write_metrics_csv(val_csv, val_eval.samples)

    aggregates: List[AggregateMetrics] = []
    if train_eval.aggregate:
        aggregates.append(train_eval.aggregate)
    if val_eval and val_eval.aggregate:
        aggregates.append(val_eval.aggregate)
    if aggregates:
        aggregate_path = log_dir / f"{args.attack}_aggregate_{timestamp}.json"
        save_aggregate_json(aggregate_path, aggregates)

    all_samples = list(train_samples) + list(val_samples)
    all_evals: List[SampleEvaluation] = list(train_eval.samples)
    if val_eval:
        all_evals.extend(val_eval.samples)
    log_per_sample_summaries(
        logger=logger,
        attack_name=args.attack,
        patch=patch,
        samples=all_samples,
        evaluations=all_evals,
        obj_id=args.obj_id,
        gt_label=args.gt_label,
    )

    if not args.no_visuals:
        limit = None if args.visual_limit < 0 else args.visual_limit
        save_visuals(trainer, patch, train_samples, visual_root / "train", "train", limit)
        if val_samples:
            save_visuals(trainer, patch, val_samples, visual_root / "val", "val", limit)

    print(f"[INFO] 通用补丁训练完成，输出目录：{attack_output_root}")
    if aggregates:
        for agg in aggregates:
            print(
                f"[INFO] {agg.split} 平均 IoU: {agg.mean_adv_iou:.4f} (Δ {agg.mean_delta_iou:.4f}), "
                f"平均 Dice: {agg.mean_adv_dice:.4f} (Δ {agg.mean_delta_dice:.4f})"
            )


if __name__ == "__main__":
    main()
