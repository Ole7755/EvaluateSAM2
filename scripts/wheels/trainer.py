"""
训练与评估相关“轮子”模块：复用通用补丁训练器及其数据结构。
"""

from __future__ import annotations

from ..uap_patch_trainer import (
    AggregateMetrics,
    EvaluationSummary,
    SampleEvaluation,
    StepRecord,
    UAPSample,
    UniversalPatchTrainer,
    load_uap_samples,
    match_sample,
)

__all__ = [
    "UniversalPatchTrainer",
    "UAPSample",
    "load_uap_samples",
    "match_sample",
    "AggregateMetrics",
    "EvaluationSummary",
    "SampleEvaluation",
    "StepRecord",
]
