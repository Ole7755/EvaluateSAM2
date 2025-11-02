"""
评估指标计算与结果汇总。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "FrameMetrics",
    "SequenceReport",
    "compute_iou",
    "compute_dice",
    "compute_precision_recall",
    "evaluate_sequence",
    "write_report_csv",
]


def _ensure_bool(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.bool_:
        return mask.astype(bool)
    return mask


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    IoU = |A ∩ B| / |A ∪ B|
    """
    pred_b = _ensure_bool(pred)
    gt_b = _ensure_bool(gt)
    intersection = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Dice = 2|A ∩ B| / (|A| + |B|)
    """
    pred_b = _ensure_bool(pred)
    gt_b = _ensure_bool(gt)
    intersection = np.logical_and(pred_b, gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return 1.0 if intersection == 0 else 0.0
    return float((2.0 * intersection) / denom)


def compute_precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred_b = _ensure_bool(pred)
    gt_b = _ensure_bool(gt)
    tp = np.logical_and(pred_b, gt_b).sum()
    fp = np.logical_and(pred_b, np.logical_not(gt_b)).sum()
    fn = np.logical_and(np.logical_not(pred_b), gt_b).sum()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    return precision, recall


@dataclass(slots=True)
class FrameMetrics:
    frame_token: str
    iou: float
    dice: float
    precision: float
    recall: float
    predicted_iou: float | None = None


@dataclass(slots=True)
class SequenceReport:
    dataset: str
    sequence: str
    frames: list[FrameMetrics] = field(default_factory=list)

    def add(self, metrics: FrameMetrics) -> None:
        self.frames.append(metrics)

    @property
    def iou_mean(self) -> float:
        return float(np.mean([item.iou for item in self.frames])) if self.frames else 0.0

    @property
    def dice_mean(self) -> float:
        return float(np.mean([item.dice for item in self.frames])) if self.frames else 0.0

    @property
    def predicted_iou_mean(self) -> float | None:
        scores = [item.predicted_iou for item in self.frames if item.predicted_iou is not None]
        if not scores:
            return None
        return float(np.mean(scores))

    def to_summary_row(self) -> dict[str, object]:
        summary = {
            "dataset": self.dataset,
            "sequence": self.sequence,
            "frames": len(self.frames),
            "iou_mean": round(self.iou_mean, 4),
            "dice_mean": round(self.dice_mean, 4),
        }
        if self.predicted_iou_mean is not None:
            summary["predicted_iou_mean"] = round(self.predicted_iou_mean, 4)
        return summary

    def iter_rows(self) -> Iterable[dict[str, object]]:
        for item in self.frames:
            row = {
                "dataset": self.dataset,
                "sequence": self.sequence,
                "frame": item.frame_token,
                "iou": round(item.iou, 6),
                "dice": round(item.dice, 6),
                "precision": round(item.precision, 6),
                "recall": round(item.recall, 6),
            }
            if item.predicted_iou is not None:
                row["predicted_iou"] = round(item.predicted_iou, 6)
            yield row


def evaluate_sequence(
    *,
    dataset: str,
    sequence: str,
    frame_tokens: Sequence[str],
    predictions: Sequence[np.ndarray],
    ground_truth: Sequence[np.ndarray],
    predicted_scores: Sequence[float] | None = None,
) -> SequenceReport:
    if len(predictions) != len(ground_truth):
        raise ValueError("预测与 GT 掩码数量不一致。")
    if len(frame_tokens) != len(predictions):
        raise ValueError("帧 token 数量与掩码数量不一致。")
    if predicted_scores is not None and len(predicted_scores) != len(predictions):
        raise ValueError("预测分数数量与掩码数量不一致。")

    report = SequenceReport(dataset=dataset, sequence=sequence)
    for idx, (token, pred, gt) in enumerate(zip(frame_tokens, predictions, ground_truth)):
        iou = compute_iou(pred, gt)
        dice = compute_dice(pred, gt)
        precision, recall = compute_precision_recall(pred, gt)
        score = None
        if predicted_scores is not None:
            score = float(predicted_scores[idx])
        report.add(
            FrameMetrics(
                frame_token=token,
                iou=iou,
                dice=dice,
                precision=precision,
                recall=recall,
                predicted_iou=score,
            )
        )
    return report


def write_report_csv(report: SequenceReport, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fp:
        fieldnames = ["dataset", "sequence", "frame", "iou", "dice", "precision", "recall", "predicted_iou"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.iter_rows():
            if "predicted_iou" not in row:
                row["predicted_iou"] = ""
            writer.writerow(row)
