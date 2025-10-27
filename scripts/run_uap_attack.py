"""
针对 SAM2 实现通用扰动（UAP）攻击的命令行脚本。

当前版本仅支持对首帧进行攻击，后续可扩展为逐帧或序列级别。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import os
import torch

if __package__ is None:  # pragma: no cover - 兼容直接以 python 执行脚本
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from sam2.build_sam import build_sam2_video_predictor

from scripts.sam2_attack_utils import (
    AttackConfig,
    AttackLogger,
    AttackSummary,
    BestWorstTracker,
    compute_perturbation_norms,
    eval_masks_numpy,
    load_mask_tensor,
    load_rgb_tensor,
    mask_to_binary,
    mask_probs_to_numpy,
    resize_image_tensor,
    resize_mask_tensor,
    restore_image_tensor,
    save_rgb_tensor,
    save_perturbation_image,
)
from scripts.uap_attacks import (
    FGSMAttack,
    PGDAttack,
    BIMAttack,
    CarliniWagnerAttack,
    SAM2ForwardHelper,
)

# 项目路径与数据目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "sam2_hiera_s.yaml"
WEIGHT_PATH = PROJECT_ROOT / "weights" / "sam2_hiera_small.pt"
DATA_ROOT = PROJECT_ROOT / "data" / "DAVIS"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
LOG_ROOT = PROJECT_ROOT / "logs"

os.environ.setdefault("SAM2_CONFIG_DIR", str(CONFIG_PATH.parent))


def parse_args() -> argparse.Namespace:
    """解析命令行参数，集中管理默认值。"""
    parser = argparse.ArgumentParser(description="对 SAM2 施加通用扰动 (UAP) 攻击。")
    parser.add_argument("--sequence", type=str, required=True, help="DAVIS 序列名称，例如 bear。")
    parser.add_argument("--frame-token", type=str, default="00000", help="要攻击的帧编号，默认首帧 00000。")
    parser.add_argument(
        "--gt-label",
        type=int,
        default=None,
        help="首帧掩码中的目标标签，默认为 None 表示掩码 > 0 均为前景。",
    )
    parser.add_argument("--obj-id", type=int, default=1, help="记录用的对象编号。")
    parser.add_argument(
        "--attack",
        type=str,
        choices=("fgsm", "pgd", "bim", "cw"),
        required=True,
        help="选择攻击类型。",
    )
    parser.add_argument("--epsilon", type=float, default=0.03, help="L_inf 扰动半径。")
    parser.add_argument("--step-size", type=float, default=0.01, help="每步更新步长。")
    parser.add_argument("--steps", type=int, default=40, help="迭代步数，对于 FGSM 可保持默认。")
    parser.add_argument("--random-start", action="store_true", help="PGD 是否随机初始化。")
    parser.add_argument("--input-size", type=int, default=1024, help="输入统一缩放到的分辨率。")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="概率图转二值掩码的阈值。")
    parser.add_argument("--cw-confidence", type=float, default=0.0, help="C&W 置信度超参数。")
    parser.add_argument("--cw-binary-steps", type=int, default=5, help="C&W 内部二分搜索次数。")
    parser.add_argument("--cw-lr", type=float, default=0.01, help="C&W 优化器学习率。")
    parser.add_argument("--device", type=str, default="cuda", help="指定使用的设备，可为 cuda 或 cpu。")
    return parser.parse_args()


def ensure_prerequisites() -> None:
    """检查配置、权重与数据路径是否存在。"""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"未找到 SAM2 配置文件：{CONFIG_PATH}")
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"未找到 SAM2 权重文件：{WEIGHT_PATH}")
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"未找到 DAVIS 数据目录：{DATA_ROOT}")


def find_frame_path(rgb_dir: Path, frame_token: str) -> Path:
    """在常见后缀中查找帧文件。"""
    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
        candidate = rgb_dir / f"{frame_token}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"序列 {rgb_dir.parent.name} 的帧 {frame_token} 文件不存在。")


def build_attack(args: argparse.Namespace) -> Tuple[object, AttackConfig]:
    """根据参数实例化攻击器，并返回配置记录。"""
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
        raise FileNotFoundError(f"RGB 帧目录不存在：{rgb_dir}")
    ann_dir = DATA_ROOT / "Annotations_unsupervised" / "480p" / sequence
    if not ann_dir.exists():
        raise FileNotFoundError(f"掩码目录不存在：{ann_dir}")

    frame_path = find_frame_path(rgb_dir, frame_token)
    mask_path = ann_dir / f"{frame_token}.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"未找到首帧掩码：{mask_path}")

    # 载入图像与掩码，并按官方预处理缩放 + 填充
    image_tensor = load_rgb_tensor(frame_path, device=device)
    origin_hw = tuple(image_tensor.shape[-2:])
    resized_image, resize_info = resize_image_tensor(image_tensor, args.input_size)

    raw_mask = load_mask_tensor(mask_path, device=device)
    binary_mask = mask_to_binary(raw_mask, label=args.gt_label)
    resized_mask = resize_mask_tensor(binary_mask, resize_info).to(device)

    gt_mask_numpy = binary_mask.detach().cpu().numpy().astype(bool)

    # 构建预测器与前向辅助
    predictor = build_sam2_video_predictor(CONFIG_PATH.name, WEIGHT_PATH.as_posix())
    helper = SAM2ForwardHelper(predictor, device=device)

    clean_input = resized_image.unsqueeze(0)
    attack, attack_config = build_attack(args)

    # 初始化日志目录与输出路径
    attack_dir = OUTPUT_ROOT / sequence / args.attack
    attack_dir.mkdir(parents=True, exist_ok=True)
    log_dir = LOG_ROOT / sequence / "attacks" / args.attack
    logger = AttackLogger(log_dir)
    tracker = BestWorstTracker(
        record_path=log_dir / "best_worst.json",
        best_dir=attack_dir / "best_cases",
        worst_dir=attack_dir / "worst_cases",
    )
    logger.save_config(attack_config)

    # 计算干净样本的预测，用于基线指标
    clean_output = helper.forward(clean_input, prompt_mask=resized_mask)
    clean_probs = clean_output.probs
    clean_mask_np = mask_probs_to_numpy(clean_probs, resize_info, origin_hw, args.mask_threshold)
    clean_iou, clean_dice = eval_masks_numpy(clean_mask_np, gt_mask_numpy)

    # 执行攻击生成通用扰动
    perturbation = attack.generate(
        helper=helper,
        clean_image=clean_input,
        target_mask=resized_mask,
        random_start=args.random_start,
    )
    adv_input = (clean_input + perturbation).clamp(0.0, 1.0)

    adv_output = helper.forward(adv_input, prompt_mask=resized_mask)
    adv_probs = adv_output.probs
    adv_mask_np = mask_probs_to_numpy(adv_probs, resize_info, origin_hw, args.mask_threshold)
    adv_iou, adv_dice = eval_masks_numpy(adv_mask_np, gt_mask_numpy)

    # 计算扰动范数并写入日志
    perturbation_norms = compute_perturbation_norms(perturbation)
    delta_iou = clean_iou - adv_iou

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
    logger.save_summary(summary, extra={"frame_path": str(frame_path), "mask_path": str(mask_path)})
    logger.save_tensor(perturbation.squeeze(0), name=f"{sequence}_{args.attack}_uap")

    artifacts = {
        "clean": clean_image_path,
        "adv": adv_image_path,
        "perturbation": perturbation_image_path,
    }
    tracker_update = tracker.update(summary, artifacts, attack_name=args.attack)
    if tracker_update.get("skipped"):
        print(
            f"[INFO] clean_iou={clean_iou:.6f} 低于 BestWorstTracker 阈值 {tracker.min_clean_iou:.2f}，"
            "跳过最佳/最差案例更新。"
        )
    else:
        if tracker_update["best"]:
            print(
                "[INFO] 当前结果刷新最佳攻击案例 "
                f"(clean_iou={clean_iou:.6f}, adv_iou={adv_iou:.6f}, ΔIoU={delta_iou:.6f})，"
                f"已保存至 {tracker.best_root / args.attack}"
            )
        if tracker_update["worst"]:
            print(
                "[INFO] 当前结果刷新最差攻击案例 "
                f"(clean_iou={clean_iou:.6f}, adv_iou={adv_iou:.6f}, ΔIoU={delta_iou:.6f})，"
                f"已保存至 {tracker.worst_root / args.attack}"
            )

    mask_dir = attack_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    clean_mask_img = (clean_mask_np.astype(float) * 255).astype("uint8")
    adv_mask_img = (adv_mask_np.astype(float) * 255).astype("uint8")
    gt_mask_img = (gt_mask_numpy.astype(float) * 255).astype("uint8")

    from PIL import Image

    Image.fromarray(clean_mask_img).save(mask_dir / f"{frame_token}_clean.png")
    Image.fromarray(adv_mask_img).save(mask_dir / f"{frame_token}_adv.png")
    Image.fromarray(gt_mask_img).save(mask_dir / f"{frame_token}_gt.png")

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

    print("[INFO] 攻击完成，核心指标如下：")
    for key, value in report.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
