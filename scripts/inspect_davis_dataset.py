"""
简单浏览 DAVIS 数据集的辅助脚本，帮助快速了解可用序列与标注情况。

示例：
    python -m scripts.inspect_davis_dataset --resolution 480p
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "DAVIS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 DAVIS 数据集基础信息。")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="DAVIS 数据所在根目录，默认指向 data/DAVIS。",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="480p",
        help="帧分辨率子目录，例如 480p、Full-Resolution 等。",
    )
    parser.add_argument(
        "--mask-subdir",
        type=str,
        default="Annotations_unsupervised",
        help="掩码目录名称，可切换为 Annotations。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅统计前 N 个序列，主要用于快速调试。",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="若指定，则将统计结果保存为 JSON 文件。",
    )
    parser.add_argument(
        "--show-mask-stats",
        action="store_true",
        help="额外统计首帧掩码的标签及占比。",
    )
    return parser.parse_args()


def list_sequences(rgb_root: Path) -> List[Path]:
    if not rgb_root.exists():
        raise FileNotFoundError(f"RGB 帧目录不存在：{rgb_root}")
    return sorted(path for path in rgb_root.iterdir() if path.is_dir())


def read_image_size(image_path: Path) -> Optional[tuple[int, int]]:
    try:
        with Image.open(image_path) as img:
            return img.size[1], img.size[0]  # 返回 (H, W)
    except Exception:
        return None


def collect_mask_stats(mask_path: Path) -> Optional[Dict[str, List[Dict[str, float]]]]:
    if not mask_path.exists():
        return None
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    values, counts = np.unique(mask, return_counts=True)
    total = mask.size
    stats = [
        {
            "label": int(v),
            "pixels": int(c),
            "ratio": float(c / total),
        }
        for v, c in zip(values, counts)
    ]
    return {"labels": stats}


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    rgb_root = data_root / "JPEGImages" / args.resolution
    mask_root = data_root / args.mask_subdir / args.resolution

    sequences = list_sequences(rgb_root)
    if args.limit is not None:
        sequences = sequences[: args.limit]

    summary = []

    for seq_path in sequences:
        frames = sorted(
            f
            for f in seq_path.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        num_frames = len(frames)
        height_width = read_image_size(frames[0]) if frames else None

        mask_dir = mask_root / seq_path.name
        mask_frames = (
            sorted(
                f
                for f in mask_dir.iterdir()
                if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            if mask_dir.exists()
            else []
        )

        mask_stats = None
        if args.show_mask_stats and mask_frames:
            mask_stats = collect_mask_stats(mask_frames[0])

        record = {
            "sequence": seq_path.name,
            "num_frames": num_frames,
            "frame_size": height_width,
            "has_masks": mask_dir.exists(),
            "num_masks": len(mask_frames),
        }
        if mask_stats is not None:
            record["mask_stats"] = mask_stats
        summary.append(record)

    # 控制台输出
    if not summary:
        print("[WARN] 未在指定路径下找到任何序列，请检查数据目录。")
    else:
        print(f"[INFO] 数据目录：{data_root}")
        print(f"[INFO] 共统计序列数：{len(summary)}")
        for record in summary:
            line = (
                f"- {record['sequence']}: frames={record['num_frames']}, "
                f"size={record['frame_size']}, masks={record['num_masks']}"
            )
            print(line)
            if args.show_mask_stats and "mask_stats" in record:
                labels = record["mask_stats"]["labels"]
                label_str = ", ".join(
                    f"{item['label']}({item['ratio']:.2%})"
                    for item in labels
                    if item["label"] != 0
                )
                if label_str:
                    print(f"    前景标签占比：{label_str}")

    # 可选 JSON 输出
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 已写入 JSON：{args.output_json}")


if __name__ == "__main__":
    main()
