"""
数据集加载与路径解析工具。

本地 macOS 仅负责生成评估脚本，因此这里的函数以“构建路径描述”和
“检查目录布局”为主，不直接访问远程 GPU 环境上的数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

__all__ = [
    "DATASET_ALIASES",
    "DEFAULT_LAYOUT",
    "SequenceSpec",
    "SequencePaths",
    "resolve_dataset_root",
    "resolve_sequence_paths",
    "list_sequences",
    "list_frame_tokens",
    "find_frame_path",
    "normalize_object_ids",
    "normalize_masks",
]

# 数据集名称别名，便于命令行快速输入。
DATASET_ALIASES: dict[str, tuple[str, ...]] = {
    "davis": ("davis", "DAVIS"),
    "mose": ("mose", "MOSE"),
    "vos": ("vos", "youtube-vos", "ytvos", "YouTubeVOS"),
}

# 默认的目录模板，可通过 SequenceSpec 覆盖。
DEFAULT_LAYOUT: dict[str, dict[str, str | None]] = {
    "davis": {
        "rgb": "davis/JPEGImages/{resolution}/{sequence}",
        "mask": "davis/Annotations/{resolution}/{sequence}",
    },
    "mose": {
        "rgb": "mose/JPEGImages/{sequence}",
        "mask": "mose/Annotations/{sequence}",
    },
    "vos": {
        # YouTube-VOS 通常按 split/sequence 组织
        "rgb": "vos/{split}/JPEGImages/{sequence}",
        "mask": "vos/{split}/Annotations/{sequence}",
    },
}

_FRAME_EXTENSIONS = (".jpg", ".png", ".jpeg", ".JPG", ".PNG")


@dataclass(frozen=True)
class SequenceSpec:
    """
    描述一次评估所需的序列信息。
    """

    dataset: str
    sequence: str
    resolution: str | None = "480p"
    split: str | None = None
    rgb_layout: str | None = None
    mask_layout: str | None = None


@dataclass(frozen=True)
class SequencePaths:
    """
    将 SequenceSpec 解析为具体的目录描述。
    """

    dataset: str
    dataset_root: Path
    sequence: str
    rgb_dir: Path
    mask_dir: Path | None

    def ensure_local(self) -> None:
        """
        在本地创建必要的目录层级，便于挂载远程软链接。
        """
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        if self.mask_dir is not None:
            self.mask_dir.mkdir(parents=True, exist_ok=True)


def _normalize_dataset_name(dataset: str) -> str:
    lowered = dataset.strip()
    for canonical, aliases in DATASET_ALIASES.items():
        if lowered == canonical or lowered in aliases:
            return canonical
    raise ValueError(f"未知数据集名称：{dataset}")


def resolve_dataset_root(
    dataset: str,
    data_root: Path | str = Path("data"),
    create: bool = True,
) -> Path:
    """
    根据数据集名称返回其在项目中的根目录。

    create=True 时会自动创建目录骨架，方便挂载远程数据。
    """
    canonical = _normalize_dataset_name(dataset)
    root = Path(data_root).resolve() / canonical
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _format_layout(spec: SequenceSpec, kind: str) -> str | None:
    layout = spec.rgb_layout if kind == "rgb" else spec.mask_layout
    if layout is not None:
        template = layout
    else:
        canonical = _normalize_dataset_name(spec.dataset)
        template = DEFAULT_LAYOUT.get(canonical, {}).get(kind)
    if template is None:
        return None
    resolution = spec.resolution or ""
    split = spec.split or ""
    return template.format(resolution=resolution, split=split, sequence=spec.sequence)


def resolve_sequence_paths(
    spec: SequenceSpec,
    data_root: Path | str = Path("data"),
    create: bool = False,
) -> SequencePaths:
    """
    将 SequenceSpec 转换为具体的 RGB/掩码目录路径。

    若 create=True，会在本地创建对应目录，方便随后挂载远程软链接。
    """
    data_root = Path(data_root)
    dataset_root = resolve_dataset_root(spec.dataset, data_root=data_root, create=create)

    rgb_layout = _format_layout(spec, "rgb")
    if rgb_layout is None:
        raise ValueError(f"{spec.dataset} 缺少 RGB 布局模板，请在 SequenceSpec 中指定 rgb_layout。")
    mask_layout = _format_layout(spec, "mask")

    rgb_dir = (data_root / rgb_layout).resolve()
    mask_dir = (data_root / mask_layout).resolve() if mask_layout is not None else None

    paths = SequencePaths(
        dataset=_normalize_dataset_name(spec.dataset),
        dataset_root=dataset_root,
        sequence=spec.sequence,
        rgb_dir=rgb_dir,
        mask_dir=mask_dir,
    )
    if create:
        paths.ensure_local()
    return paths


def list_sequences(dataset: str, data_root: Path | str = Path("data")) -> list[str]:
    """
    列出数据集中已存在的序列名称（按目录名粗略统计）。
    """
    dataset_root = resolve_dataset_root(dataset, data_root, create=False)
    if not dataset_root.exists():
        return []
    return sorted(
        entry.name
        for entry in dataset_root.iterdir()
        if entry.is_dir()
    )


def list_frame_tokens(rgb_dir: Path) -> list[str]:
    """
    根据图像文件名列出所有帧 token（不包含后缀）。
    """
    tokens: set[str] = set()
    if not rgb_dir.exists():
        return []
    for path in rgb_dir.iterdir():
        if path.is_file():
            for ext in _FRAME_EXTENSIONS:
                if path.name.endswith(ext):
                    tokens.add(path.name[:-len(ext)])
                    break
    return sorted(tokens)


def find_frame_path(rgb_dir: Path, frame_token: str) -> Path:
    """
    在给定目录下查找匹配的帧文件。
    """
    for ext in _FRAME_EXTENSIONS:
        candidate = rgb_dir / f"{frame_token}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{rgb_dir} 中未找到帧 {frame_token}")


def normalize_object_ids(object_ids: Iterable) -> list[int]:
    """
    将模型输出的对象 ID 统一转换为 Python int 列表。
    """
    if isinstance(object_ids, torch.Tensor):
        if object_ids.ndim == 0:
            return [int(object_ids.item())]
        return [int(item) for item in object_ids.detach().cpu().tolist()]
    return [int(item) for item in object_ids]


def normalize_masks(masks: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """
    将掩码结果统一转换为 torch.Tensor 列表，方便后续批处理。
    """
    if isinstance(masks, torch.Tensor):
        if masks.ndim == 2:
            return [masks]
        return [masks[idx] for idx in range(masks.shape[0])]
    return [torch.as_tensor(mask) for mask in masks]
