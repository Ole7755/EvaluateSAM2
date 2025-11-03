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
from typing import Optional

import numpy as np
from PIL import Image

from src.data_loader import SequenceSpec, find_frame_path, list_frame_tokens, list_sequences_for_spec, resolve_sequence_paths
from src.evaluator import evaluate_sequence, write_report_csv
from src.model_loader import load_image_predictor
from src.prompt_generator import build_prompt_bundle
from src.visualizer import overlay_mask, save_overlay, stack_overlays


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 SAM2 预测结果。")
    parser.add_argument("--dataset", required=True, help="数据集名称，如 davis/mose/vos。")
    parser.add_argument("--sequence", help="单个序列名称。")
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="一次指定多个序列名称（用空格分隔）。",
    )
    parser.add_argument(
        "--sequence-list",
        type=Path,
        help="包含序列名称的文本文件，每行一个序列。",
    )
    parser.add_argument(
        "--all-sequences",
        action="store_true",
        help="自动遍历数据集下可用的全部序列。",
    )
    parser.add_argument("--resolution", default="480p", help="序列分辨率（用于 DAVIS 等数据集）。")
    parser.add_argument("--split", help="适用于 YouTube-VOS 等含 split 的数据集。")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="数据集根目录。")
    parser.add_argument("--images-dir", type=Path, help="原始图像目录，可显式指定。")
    parser.add_argument("--gt-dir", type=Path, help="GT 掩码目录，可显式指定。")
    parser.add_argument("--frame-tokens", help="逗号分隔或文本文件路径，指定参与评估的帧。")
    parser.add_argument("--tag", help="可选的实验标签，用于结果命名。")
    parser.add_argument("--results-root", type=Path, default=Path("results"), help="评估结果输出根目录。")
    parser.add_argument("--pred-output-dir", type=Path, help="可选的预测掩码输出目录。")
    parser.add_argument(
        "--save-pred-masks",
        action="store_true",
        help="保存 SAM2 预测的掩码（默认为仅计算指标、不写入掩码）。",
    )
    parser.add_argument("--report-csv", type=Path, help="自定义指标 CSV 输出路径。")
    parser.add_argument("--summary-json", type=Path, help="可选的汇总 JSON 输出路径。")
    parser.add_argument("--visualize-dir", type=Path, help="若指定则生成可视化图像。")
    parser.add_argument("--visualize-count", type=int, default=0, help="限制可视化帧数，0 表示全部。")
    parser.add_argument("--threshold", type=int, default=128, help="GT 掩码二值化阈值。")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="SAM2 输出掩码阈值。")
    parser.add_argument(
        "--prompt-type",
        choices=["point", "box", "point_box"],
        default="point_box",
        help="生成提示的类型，支持单点、单框或点+框组合。",
    )
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        choices=["point", "box", "point_box"],
        help="一次运行内评估多种提示类型，提供后将覆盖 --prompt-type。",
    )
    parser.add_argument(
        "--background-points",
        type=int,
        default=0,
        help="采样背景提示点数量（仅在使用点提示时生效）。",
    )
    parser.add_argument(
        "--multimask-output",
        action="store_true",
        help="启用 SAM2 的多掩码输出（返回三张掩码并选择评分最高者）。",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子，用于背景点采样。")
    parser.add_argument("--sam2-config", type=Path, required=True, help="SAM2 配置文件路径。")
    parser.add_argument("--checkpoint", type=Path, required=True, help="SAM2 权重文件路径。")
    parser.add_argument("--device", default="cuda", help="运行设备，例如 cuda 或 cpu。")
    return parser.parse_args()


def _load_frame_tokens(
    value: str | None,
    images_dir: Path,
    gt_dir: Path,
) -> list[str]:
    if value is not None:
        value = value.strip()
        candidate = Path(value)
        if candidate.exists():
            return [line.strip() for line in candidate.read_text().splitlines() if line.strip()]
        return [token.strip() for token in value.split(",") if token.strip()]

    image_tokens = set(list_frame_tokens(images_dir))
    gt_tokens = set(list_frame_tokens(gt_dir))
    tokens = sorted(image_tokens & gt_tokens)
    if not tokens:
        missing_info = {
            "images_dir": images_dir.as_posix(),
            "gt_dir": gt_dir.as_posix(),
        }
        raise RuntimeError(f"未找到可评估的帧，请确认目录结构。信息：{json.dumps(missing_info, ensure_ascii=False)}")
    missing_in_images = sorted(gt_tokens - image_tokens)
    missing_in_gt = sorted(image_tokens - gt_tokens)
    if missing_in_images:
        print(f"[WARN] 以下帧在 GT 中存在但图像缺失：{', '.join(missing_in_images[:10])}...")
    if missing_in_gt:
        print(f"[WARN] 以下帧在图像目录中存在但 GT 缺失：{', '.join(missing_in_gt[:10])}...")
    return tokens


def _prepare_prompts(
    gt_mask: np.ndarray,
    *,
    prompt_type: str,
    background_points: int,
    rng: np.random.Generator,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    include_box = prompt_type in {"box", "point_box"}
    include_points = prompt_type in {"point", "point_box"}
    bundle = build_prompt_bundle(
        gt_mask,
        include_points=include_points,
        include_box=include_box,
        background_points=background_points if include_points else 0,
        rng=rng,
    )
    points = bundle.points if include_points else None
    labels = bundle.point_labels if include_points else None
    box = bundle.boxes[0] if (include_box and bundle.boxes is not None) else None
    return points, labels, box


def _save_pred_mask(mask: np.ndarray, output_dir: Path, token: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img.save(output_dir / f"{token}.png")
    

def _load_mask(path: Path, threshold: int) -> np.ndarray:
    image = Image.open(path)
    array = np.array(image)
    if array.ndim == 3:
        array = array[..., 0]
    if array.max() <= 1:
        mask = array > 0
    else:
        mask = array > 0 if array.max() <= threshold else array >= threshold
    return mask.astype(np.uint8)


def _default_pred_dir(
    results_root: Path,
    dataset: str,
    sequence: str,
    tag: str | None,
    prompt_type: str | None = None,
) -> Path:
    label = tag or "default"
    path = results_root / "comparisons" / dataset / sequence / label
    if prompt_type:
        path = path / prompt_type
    return path


def _default_visual_dir(
    results_root: Path,
    dataset: str,
    sequence: str,
    tag: str | None,
    prompt_type: str | None = None,
) -> Path:
    label = tag or "default"
    path = results_root / "visualizations" / dataset / sequence / label
    if prompt_type:
        path = path / prompt_type
    return path


def _default_report_path(
    results_root: Path,
    dataset: str,
    sequence: str,
    tag: str | None,
    prompt_type: str | None = None,
) -> Path:
    label = tag or "default"
    stem = f"{dataset}_{sequence}_{label}"
    if prompt_type:
        stem = f"{stem}_{prompt_type}"
    return results_root / "metrics" / f"{stem}.csv"


def _load_sequence_list_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"序列列表文件不存在：{path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _deduplicate_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _resolve_sequence_names(args: argparse.Namespace) -> list[str]:
    collected: list[str] = []

    if args.sequence:
        for token in args.sequence.split(","):
            token = token.strip()
            if token:
                collected.append(token)

    if args.sequences:
        for item in args.sequences:
            for token in item.split(","):
                token = token.strip()
                if token:
                    collected.append(token)

    if args.sequence_list:
        collected.extend(_load_sequence_list_file(args.sequence_list))

    if args.all_sequences:
        probe_spec = SequenceSpec(
            dataset=args.dataset,
            sequence="__SEQUENCE_PLACEHOLDER__",
            resolution=args.resolution,
            split=args.split,
        )
        auto_sequences = list_sequences_for_spec(probe_spec, data_root=args.data_root)
        if not auto_sequences:
            raise RuntimeError("未在指定数据集中找到任何序列，请确认目录结构或提供 --sequence-list。")
        collected.extend(auto_sequences)

    sequences = _deduplicate_preserve_order([item for item in collected if item])
    if not sequences:
        raise RuntimeError("请至少通过 --sequence / --sequences / --sequence-list / --all-sequences 提供一个序列。")
    return sequences


def _evaluate_sequence_with_prompts(
    *,
    spec: SequenceSpec,
    args: argparse.Namespace,
    predictor,
    prompt_types: list[str],
    multi_prompt: bool,
    sam2_config: Path,
    checkpoint: Path,
    results_root: Path,
    override_images_dir: Path | None,
    override_gt_dir: Path | None,
    multiple_sequences: bool,
) -> list[dict[str, object]]:
    sequence_paths = None
    images_dir = override_images_dir
    gt_dir = override_gt_dir

    if images_dir is None or gt_dir is None:
        try:
            sequence_paths = resolve_sequence_paths(spec, data_root=args.data_root, create=False)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"无法根据默认布局推断序列 {spec.sequence} 的图像或 GT 掩码目录，请提供 --images-dir 与 --gt-dir。"
            ) from exc

    images_dir = images_dir or (sequence_paths.rgb_dir if sequence_paths else None)
    if images_dir is None:
        raise RuntimeError(f"序列 {spec.sequence} 无法确定原始图像目录，请通过 --images-dir 指定。")
    images_dir = images_dir.resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"序列 {spec.sequence} 的原始图像目录不存在：{images_dir}")

    gt_dir = gt_dir or (sequence_paths.mask_dir if sequence_paths else None)
    if gt_dir is None:
        raise RuntimeError(f"序列 {spec.sequence} 无法确定 GT 掩码目录，请通过 --gt-dir 指定。")
    gt_dir = gt_dir.resolve()
    if not gt_dir.exists():
        raise FileNotFoundError(f"序列 {spec.sequence} 的 GT 掩码目录不存在：{gt_dir}")

    frame_tokens = _load_frame_tokens(args.frame_tokens, images_dir, gt_dir)

    frames_data: list[dict[str, object]] = []
    for token in frame_tokens:
        image_path = find_frame_path(images_dir, token)
        gt_path = find_frame_path(gt_dir, token)
        with Image.open(image_path) as frame:
            image_np = np.array(frame.convert("RGB"))
        gt_mask = _load_mask(gt_path, args.threshold)
        frames_data.append(
            {
                "token": token,
                "image_np": image_np,
                "gt_mask": gt_mask,
            }
        )

    ground_truth_masks = [item["gt_mask"] for item in frames_data]

    if len(prompt_types) == 1:
        rng_map = {prompt_types[0]: np.random.default_rng(args.seed)}
    else:
        seed_sequence = np.random.SeedSequence(args.seed)
        child_sequences = seed_sequence.spawn(len(prompt_types))
        rng_map = {
            prompt_type: np.random.default_rng(child_seq)
            for prompt_type, child_seq in zip(prompt_types, child_sequences)
        }

    summaries: list[dict[str, object]] = []

    for prompt_type in prompt_types:
        rng = rng_map[prompt_type]
        predictions: list[np.ndarray] = []
        predicted_scores: list[float] = []

        pred_output_dir: Path | None = None
        if args.save_pred_masks:
            if args.pred_output_dir:
                base_dir = args.pred_output_dir.resolve()
                if multiple_sequences:
                    base_dir = base_dir / spec.sequence
                pred_output_dir = base_dir / prompt_type if multi_prompt else base_dir
            else:
                pred_output_dir = _default_pred_dir(
                    results_root,
                    spec.dataset,
                    spec.sequence,
                    args.tag,
                    prompt_type if multi_prompt else None,
                ).resolve()
            pred_output_dir.mkdir(parents=True, exist_ok=True)

        for data in frames_data:
            token = data["token"]
            image_np = data["image_np"]
            gt_mask = data["gt_mask"]

            predictor.set_image(image_np)

            try:
                points, labels, box = _prepare_prompts(
                    gt_mask.astype(bool),
                    prompt_type=prompt_type,
                    background_points=args.background_points,
                    rng=rng,
                )
            except ValueError as exc:
                print(f"[WARN] 序列 {spec.sequence} 帧 {token}（{prompt_type}）无法生成提示：{exc}，使用空预测掩码。")
                empty_mask = np.zeros_like(gt_mask, dtype=np.uint8)
                predictions.append(empty_mask)
                predicted_scores.append(0.0)
                if pred_output_dir is not None:
                    _save_pred_mask(empty_mask, pred_output_dir, token)
                continue

            masks, iou_predictions, _ = predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=args.multimask_output,
            )
            best_idx = int(np.argmax(iou_predictions))
            selected_mask = masks[best_idx]
            binary_mask = (selected_mask >= args.mask_threshold).astype(np.uint8)

            predictions.append(binary_mask)
            predicted_scores.append(float(iou_predictions[best_idx]))

            if pred_output_dir is not None:
                _save_pred_mask(binary_mask, pred_output_dir, token)

        report = evaluate_sequence(
            dataset=spec.dataset,
            sequence=spec.sequence,
            frame_tokens=frame_tokens,
            predictions=predictions,
            ground_truth=ground_truth_masks,
            predicted_scores=predicted_scores,
        )

        if args.report_csv:
            base_csv = args.report_csv.resolve()
            stem = base_csv.stem
            if multiple_sequences:
                stem = f"{stem}_{spec.sequence}"
            if multi_prompt:
                stem = f"{stem}_{prompt_type}"
            report_csv = base_csv.with_name(f"{stem}{base_csv.suffix}")
        else:
            report_csv = _default_report_path(
                results_root,
                spec.dataset,
                spec.sequence,
                args.tag,
                prompt_type if multi_prompt else None,
            ).resolve()
        write_report_csv(report, report_csv)

        summary = report.to_summary_row()
        summary["tag"] = args.tag
        summary["prompt_type"] = prompt_type
        summary["background_points"] = args.background_points
        summary["multimask_output"] = args.multimask_output
        summary["mask_threshold"] = args.mask_threshold
        summary["sam2_config"] = sam2_config.as_posix()
        summary["checkpoint"] = checkpoint.as_posix()
        summary["device"] = args.device
        summary["images_dir"] = images_dir.as_posix()
        summary["gt_dir"] = gt_dir.as_posix()
        summary["save_pred_masks"] = bool(pred_output_dir)
        if pred_output_dir is not None:
            summary["pred_masks_dir"] = pred_output_dir.as_posix()
        summary["report_csv"] = report_csv.as_posix()
        summary["prompt_seed"] = args.seed

        if args.visualize_dir:
            base_visual_dir = args.visualize_dir.resolve()
            if multiple_sequences:
                base_visual_dir = base_visual_dir / spec.sequence
            visualize_dir = base_visual_dir / prompt_type if multi_prompt else base_visual_dir
        else:
            visualize_dir = _default_visual_dir(
                results_root,
                spec.dataset,
                spec.sequence,
                args.tag,
                prompt_type if multi_prompt else None,
            ).resolve()

        if visualize_dir is not None:
            visualize_dir.mkdir(parents=True, exist_ok=True)
            limit = args.visualize_count if args.visualize_count > 0 else len(frame_tokens)
            for token, pred_mask, gt_mask in zip(
                frame_tokens[:limit],
                predictions[:limit],
                ground_truth_masks[:limit],
            ):
                try:
                    frame_path = find_frame_path(images_dir, token)
                except FileNotFoundError:
                    continue
                frame = Image.open(frame_path).convert("RGB")
                pred_overlay = overlay_mask(frame, pred_mask, color=(0, 255, 0), alpha=0.4)
                gt_overlay = overlay_mask(frame, gt_mask, color=(255, 0, 0), alpha=0.4)
                combined = stack_overlays([pred_overlay, gt_overlay])
                save_overlay(combined, visualize_dir / f"{token}.png")
            summary["visualize_dir"] = visualize_dir.as_posix()

        summaries.append(summary)

    return summaries


def main() -> None:
    args = _parse_args()

    sequence_names = _resolve_sequence_names(args)
    multiple_sequences = len(sequence_names) > 1

    if multiple_sequences and (args.images_dir or args.gt_dir):
        raise RuntimeError("多序列评估暂不支持统一的 --images-dir / --gt-dir 参数，请使用默认目录布局或逐序列运行。")

    sam2_config = args.sam2_config.resolve()
    checkpoint = args.checkpoint.resolve()
    if not sam2_config.exists():
        raise FileNotFoundError(f"SAM2 配置文件不存在：{sam2_config}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM2 权重文件不存在：{checkpoint}")

    predictor = load_image_predictor(
        config_path=sam2_config,
        checkpoint_path=checkpoint,
        device=args.device,
        mask_threshold=args.mask_threshold,
    )

    prompt_types = list(dict.fromkeys(args.prompt_types or [args.prompt_type]))
    multi_prompt = len(prompt_types) > 1

    results_root = args.results_root.resolve()
    summary_json_path = args.summary_json.resolve() if args.summary_json else None
    override_images_dir = args.images_dir.resolve() if args.images_dir else None
    override_gt_dir = args.gt_dir.resolve() if args.gt_dir else None

    all_summaries: list[dict[str, object]] = []
    for sequence_name in sequence_names:
        spec = SequenceSpec(
            dataset=args.dataset,
            sequence=sequence_name,
            resolution=args.resolution,
            split=args.split,
        )
        seq_summaries = _evaluate_sequence_with_prompts(
            spec=spec,
            args=args,
            predictor=predictor,
            prompt_types=prompt_types,
            multi_prompt=multi_prompt,
            sam2_config=sam2_config,
            checkpoint=checkpoint,
            results_root=results_root,
            override_images_dir=override_images_dir,
            override_gt_dir=override_gt_dir,
            multiple_sequences=multiple_sequences,
        )
        all_summaries.extend(seq_summaries)

    summary_payload: list[dict[str, object]] | dict[str, object]
    if len(all_summaries) == 1:
        summary_payload = all_summaries[0]
    else:
        summary_payload = all_summaries

    if summary_json_path is not None:
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)
        summary_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
