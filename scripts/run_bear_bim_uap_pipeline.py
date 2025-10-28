"""
在 bear 序列上使用 BIM 训练通用补丁，并在其它序列上评估。

复用通用管线脚本 `run_bear_uap_pipeline.py`。
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:  # pragma: no cover - 兼容直接执行
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.run_bear_uap_pipeline import main as run_pipeline  # noqa: E402


def main() -> None:
    argv = ["--attack", "bim", *sys.argv[1:]]
    run_pipeline(argv)


if __name__ == "__main__":
    main()

