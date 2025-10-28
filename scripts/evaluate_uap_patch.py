"""
评估已训练好的通用对抗补丁 (UAP)，并将结果汇总保存到单个文件中。

使用示例：
    python3 scripts/evaluate_uap_patch.py \
        --attack pgd \
        --patch-path logs/uap/pgd/pgd_uap_patch_20240601-120000.pt \
        --sequences bear,boat \
        --frame-token 00000 \
        --mask-subdir Annotations_unsupervised \
        --output logs/uap/pgd/eval_summary.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

import os

from sam2.build_sam import build_sam2_video_predictor

from scripts.wheels.attacks import SAM2ForwardHelper
from scripts.wheels.trainer import EvaluationSummary, UniversalPatchTrainer, load_uap_samples
from scripts.wheels.utils import compute_perturbation_norms, ensure_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "sam2_hiera_s.yaml"
WEIGHT_PATH = PROJECT_ROOT / "weights" / "sam2_hiera_small.pt"
DATA_ROOT = PROJECT_ROOT / "data" / "DAVIS"

# 确保 SAM2 可以在本地路径下正确寻找配置文件
os.environ.setdefault("SAM2_CONFIG_DIR", str(CONFIG_PATH.parent))


def parse_sequence_list(raw: str) -> List[str]:
    items: List[str] = []
    if not raw:
        return items
    seen = set()
    for part in raw.split(","):
        name = part.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        items.append(name)
    return items


def ensure_prerequisites() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"未找到配置文件：{CONFIG_PATH}")
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"未找到权重文件：{WEIGHT_PATH}")
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"未找到 DAVIS 数据目录：{DATA_ROOT}")


def load_patch_tensor(path: Path, device: torch.device) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"未找到补丁文件：{path}")
    patch = torch.load(path, map_location=device)
    if patch.ndim == 3:
        patch = patch.unsqueeze(0)
    if patch.ndim != 4 or patch.size(0) != 1 or patch.size(1) != 3:
        raise ValueError("补丁张量应为形状 (1, 3, H, W) 或 (3, H, W)。")
    return patch.float().to(device)


def evaluation_entry(
    attack: str,
    patch_path: Path,
    sequences: List[str],
    frame_token: str,
    metrics: EvaluationSummary,
    patch_stats: Dict[str, float],
    epsilon: float,
) -> Dict[str, object]:
    per_samples: List[Dict[str, object]] = [
        {
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
        for item in metrics.samples
    ]
    aggregate: Optional[Dict[str, object]] = (
        metrics.aggregate.to_dict() if metrics.aggregate is not None else None
    )
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attack": attack,
        "patch_path": patch_path.as_posix(),
        "epsilon": float(epsilon),
        "sequences": sequences,
        "frame_token": frame_token,
        "patch_norms": patch_stats,
        "aggregate": aggregate,
        "samples": per_samples,
    }


def append_results(output_path: Path, entry: Dict[str, object], overwrite: bool) -> None:
    ensure_dir(output_path.parent)
    payload: List[Dict[str, object]] = []
    if output_path.exists() and not overwrite:
        with output_path.open("r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
                if isinstance(existing, list):
                    payload = existing
                else:
                    payload = [existing]
            except json.JSONDecodeError:
                payload = []
    payload.append(entry)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估通用对抗补丁 (UAP) 在指定序列上的效果。")
    parser.add_argument("--attack", type=str, required=True, help="补丁对应的攻击名称。")
    parser.add_argument("--patch-path", type=Path, required=True, help="保存的补丁 .pt 文件路径。")
    parser.add_argument("--sequences", type=str, required=True, help="逗号分隔的 DAVIS 序列列表。")
    parser.add_argument("--frame-token", type=str, default="00000", help="评估的帧编号。")
    parser.add_argument(
        "--mask-subdir",
        type=str,
        default="Annotations_unsupervised",
        help="掩码所在子目录名称。",
    )
    parser.add_argument("--gt-label", type=int, default=None, help="多标签掩码下的目标标签。")
    parser.add_argument("--input-size", type=int, default=1024, help="补丁生成时使用的输入尺寸。")
    parser.add_argument("--keep-aspect-ratio", action="store_true", help="是否保持宽高比缩放。")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="概率图转掩码的阈值。")
    parser.add_argument("--loss-type", choices=("dice", "bce"), default="dice", help="与训练时一致的损失类型。")
    parser.add_argument("--epsilon", type=float, default=None, help="补丁的裁剪半径，不提供则自动推断。")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备。")
    parser.add_argument("--output", type=Path, required=True, help="结果保存路径（JSON）。")
    parser.add_argument("--overwrite", action="store_true", help="若文件存在则覆盖，而非追加。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_prerequisites()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    sequences = parse_sequence_list(args.sequences)
    if not sequences:
        raise ValueError("序列列表为空，无法评估补丁。")

    patch = load_patch_tensor(args.patch_path, device=device)
    patch_norms = compute_perturbation_norms(patch)

    epsilon = float(args.epsilon if args.epsilon is not None else float(patch.abs().max().item()))

    predictor = build_sam2_video_predictor(CONFIG_PATH.name, WEIGHT_PATH.as_posix())
    helper = SAM2ForwardHelper(predictor, device=device)

    samples = load_uap_samples(
        sequences=sequences,
        frame_token=args.frame_token,
        data_root=DATA_ROOT,
        mask_subdir=args.mask_subdir,
        device=device,
        input_size=args.input_size,
        keep_aspect_ratio=args.keep_aspect_ratio,
        gt_label=args.gt_label,
    )
    if not samples:
        raise RuntimeError("未成功加载任何评估样本，请检查序列名称是否正确。")

    trainer = UniversalPatchTrainer(
        helper=helper,
        samples=samples,
        epsilon=epsilon,
        loss_type=args.loss_type,
        mask_threshold=args.mask_threshold,
    )

    target_size = trainer.samples[0].resize_info.target_size
    if patch.size(-1) != target_size or patch.size(-2) != target_size:
        raise ValueError(
            f"补丁尺寸 {tuple(patch.shape[-2:])} 与输入尺寸 {target_size} 不一致，请确认训练与评估设置相同。"
        )

    metrics = trainer.evaluate(patch, samples, split="eval")

    entry = evaluation_entry(
        attack=args.attack,
        patch_path=args.patch_path,
        sequences=sequences,
        frame_token=args.frame_token,
        metrics=metrics,
        patch_stats=patch_norms,
        epsilon=epsilon,
    )
    append_results(args.output, entry, overwrite=args.overwrite)

    summary = metrics.aggregate
    if summary:
        print(
            f"[INFO] {args.attack} 补丁评估完成，平均 IoU {summary.mean_adv_iou:.4f} "
            f"(Δ {summary.mean_delta_iou:.4f})，平均 Dice {summary.mean_adv_dice:.4f} "
            f"(Δ {summary.mean_delta_dice:.4f})。"
        )
    else:
        print("[INFO] 评估完成，未生成聚合指标。")


if __name__ == "__main__":
    main()
