"""
管线类“轮子”模块：复用统一的 bear UAP 训练/评估流程。
"""

from __future__ import annotations

from ..run_bear_uap_pipeline import main as run_bear_uap_experiment

__all__ = ["run_bear_uap_experiment"]
