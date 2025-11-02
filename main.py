#!/usr/bin/env python3
"""
SAM2 评估主入口（本地执行）。

功能：
- 读取预测与 GT 掩码，计算 IoU / Dice / Precision / Recall。
- 将逐帧指标导出为 CSV，汇总信息可选输出为 JSON。
- 可选输出可视化结果，方便人工核验。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.data_loader import (
    SequenceSpec,
    find_frame_path,
    list_frame_tokens,
    resolve_sequence_paths,
)
from src.evaluator import evaluate_sequence, write_report_csv
from src.visualizer import overlay_mask, save_overlay, stack_overlays


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 SAM2 预测结果。")
    parser.add_argument("--dataset", required=True, help="数据集名称，如 davis/mose/vos。")
    parser.add_argument("--sequence", required=True, help="序列名称。")
    parser.add_argument("--resolution", default="480p", help="序列分辨率（用于 DAVIS 等数据集）。")
    parser.add_argument("--split", help="适用于 YouTube-VOS 等含 split 的数据集。")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="数据集根目录。")
    parser.add_argument("--pred-dir", type=Path, help="预测掩码目录，缺省时基于 results/comparisons 推断。")
    parser.add_argument("--images-dir", type=Path, help="原始图像目录，可显式指定。")
    parser.add_argument("--gt-dir", type=Path, help="GT 掩码目录，可显式指定。")
    parser.add_argument("--frame-tokens", help="逗号分隔或文本文件路径，指定参与评估的帧。")
    parser.add_argument("--tag", help="可选的实验标签，用于结果命名。")
    parser.add_argument("--results-root", type=Path, default=Path("results"), help="评估结果输出根目录。")
    parser.add_argument("--report-csv", type=Path, help="自定义指标 CSV 输出路径。")
    parser.add_argument("--summary-json", type=Path, help="可选的汇总 JSON 输出路径。")
    parser.add_argument("--visualize-dir", type=Path, help="若指定则生成可视化图像。")
    parser.add_argument("--visualize-count", type=int, default=0, help="限制可视化帧数，0 表示全部。")
    parser.add_argument("--threshold", type=int, default=128, help="掩码二值化阈值。")
    return parser.parse_args()


def _load_frame_tokens(value: str | None, pred_dir: Path) -> list[str]:
    if value is None:
        return list_frame_tokens(pred_dir)
    value = value.strip()
    candidate = Path(value)
    if candidate.exists():
        return [line.strip() for line in candidate.read_text().splitlines() if line.strip()]
    return [token.strip() for token in value.split(",") if token.strip()]


def _load_mask(path: Path, threshold: int) -> np.ndarray:
    array = np.array(Image.open(path).convert("L"))
    return (array >= threshold).astype(np.uint8)


def _default_pred_dir(results_root: Path, dataset: str, sequence: str, tag: str | None) -> Path:
    label = tag or "default"
    return results_root / "comparisons" / dataset / sequence / label


def _default_visual_dir(results_root: Path, dataset: str, sequence: str, tag: str | None) -> Path:
    label = tag or "default"
    return results_root / "visualizations" / dataset / sequence / label


def _default_report_path(results_root: Path, dataset: str, sequence: str, tag: str | None) -> Path:
    label = tag or "default"
    return results_root / "metrics" / f"{dataset}_{sequence}_{label}.csv"


def main() -> None:
    args = _parse_args()

    spec = SequenceSpec(
        dataset=args.dataset,
        sequence=args.sequence,
        resolution=args.resolution,
        split=args.split,
    )

    sequence_paths = None
    if args.images_dir is None or args.gt_dir is None:
        try:
            sequence_paths = resolve_sequence_paths(spec, data_root=args.data_root, create=False)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "无法根据默认布局推断图像或 GT 掩码目录，请同时使用 --images-dir 与 --gt-dir 指定。"
            ) from exc

    pred_dir = (args.pred_dir or _default_pred_dir(args.results_root, spec.dataset, spec.sequence, args.tag)).resolve()
    if not pred_dir.exists():
        raise FileNotFoundError(f"预测掩码目录不存在：{pred_dir}")

    if args.images_dir is not None:
        images_dir = args.images_dir.resolve()
    else:
        images_dir = sequence_paths.rgb_dir if sequence_paths else None
    if images_dir is None:
        raise RuntimeError("无法推断原始图像目录，请通过 --images-dir 指定。")
    if not images_dir.exists():
        raise FileNotFoundError(f"原始图像目录不存在：{images_dir}")

    if args.gt_dir is not None:
        gt_dir = args.gt_dir.resolve()
    else:
        gt_dir = sequence_paths.mask_dir if sequence_paths else None
    if gt_dir is None:
        raise RuntimeError("无法推断 GT 掩码目录，请通过 --gt-dir 指定。")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT 掩码目录不存在：{gt_dir}")

    frame_tokens = _load_frame_tokens(args.frame_tokens, pred_dir)
    if not frame_tokens:
        raise RuntimeError(f"目录 {pred_dir} 中未找到任何帧。")

    predictions: list[np.ndarray] = []
    ground_truth: list[np.ndarray] = []
    for token in frame_tokens:
        pred_path = find_frame_path(pred_dir, token)
        gt_path = find_frame_path(gt_dir, token)
        predictions.append(_load_mask(pred_path, args.threshold))
        ground_truth.append(_load_mask(gt_path, args.threshold))

    report = evaluate_sequence(
        dataset=spec.dataset,
        sequence=spec.sequence,
        frame_tokens=frame_tokens,
        predictions=predictions,
        ground_truth=ground_truth,
    )

    report_csv = args.report_csv or _default_report_path(args.results_root, spec.dataset, spec.sequence, args.tag)
    write_report_csv(report, report_csv)

    summary = report.to_summary_row()
    summary["tag"] = args.tag
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    visualize_dir = args.visualize_dir or _default_visual_dir(args.results_root, spec.dataset, spec.sequence, args.tag)
    if visualize_dir is not None:
        visualize_dir.mkdir(parents=True, exist_ok=True)
        limit = args.visualize_count if args.visualize_count > 0 else len(frame_tokens)
        for token, pred_mask, gt_mask in zip(frame_tokens[:limit], predictions[:limit], ground_truth[:limit]):
            try:
                frame_path = find_frame_path(images_dir, token)
            except FileNotFoundError:
                # 本地可能尚未同步原始帧，跳过可视化
                continue
            frame = Image.open(frame_path).convert("RGB")
            pred_overlay = overlay_mask(frame, pred_mask, color=(0, 255, 0), alpha=0.4)
            gt_overlay = overlay_mask(frame, gt_mask, color=(255, 0, 0), alpha=0.4)
            combined = stack_overlays([pred_overlay, gt_overlay])
            save_overlay(combined, visualize_dir / f"{token}.png")


if __name__ == "__main__":
    main()
