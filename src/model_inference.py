"""
SAM2 推理封装：本地负责编排命令行与路径，实际执行在远程 Linux。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .data_loader import SequencePaths

__all__ = [
    "SAM2InferenceConfig",
    "RemoteCommand",
    "SAM2InferenceRunner",
]


@dataclass(slots=True)
class SAM2InferenceConfig:
    """
    描述一次 SAM2 推理任务的核心配置。
    """

    checkpoint: Path
    config: Path
    device: str = "cuda"
    python_bin: str = "python3"
    remote_workspace: Path | None = None
    extra_args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None

    def to_remote_path(self, path: Path) -> str:
        """
        将本地同步路径映射到远程路径。

        若远程目录布局一致，可直接返回 str(path)。
        """
        if self.remote_workspace is None:
            return str(path)
        local_root = Path.cwd()
        try:
            rel = path.resolve().relative_to(local_root)
        except ValueError:
            # 若路径不在项目内，直接返回原始绝对路径
            return str(path)
        return str((self.remote_workspace / rel).as_posix())


@dataclass(slots=True)
class RemoteCommand:
    """
    封装一次远程执行所需的信息，便于提交到 SSH / 任务队列。
    """

    argv: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)

    def format_shell(self) -> str:
        """
        以 shell 字符串形式返回命令，方便复制到终端。
        """
        env_prefix = " ".join(f"{key}={value}" for key, value in self.env.items())
        cmd = " ".join(self.argv)
        return f"{env_prefix} {cmd}".strip()


class SAM2InferenceRunner:
    """
    构造远程推理命令的辅助类。
    """

    def __init__(self, config: SAM2InferenceConfig):
        self.config = config

    def build_command(
        self,
        entry_script: Path | str,
        sequence: SequencePaths,
        *,
        output_dir: Path,
        tag: str | None = None,
        additional_args: Iterable[str] = (),
    ) -> RemoteCommand:
        """
        生成远程执行命令。默认假设 entry_script 位于同步后的工作区中。
        """
        entry = Path(entry_script)
        entry_remote = self.config.to_remote_path(entry)
        images_remote = self.config.to_remote_path(sequence.rgb_dir)
        gt_remote = self.config.to_remote_path(sequence.mask_dir) if sequence.mask_dir else None
        output_remote = self.config.to_remote_path(output_dir)

        argv: list[str] = [
            self.config.python_bin,
            entry_remote,
            "--dataset",
            sequence.dataset,
            "--sequence",
            sequence.sequence,
            "--images-dir",
            images_remote,
            "--checkpoint",
            self.config.to_remote_path(self.config.checkpoint),
            "--sam2-config",
            self.config.to_remote_path(self.config.config),
            "--output-dir",
            output_remote,
            "--device",
            self.config.device,
        ]

        if gt_remote is not None:
            argv.extend(["--gt-dir", gt_remote])
        if tag:
            argv.extend(["--tag", tag])
        argv.extend(additional_args)
        argv.extend(self.config.extra_args)

        env = dict(self.config.env) if self.config.env is not None else {}
        return RemoteCommand(argv=tuple(argv), env=env)

    def describe(
        self,
        entry_script: Path | str,
        sequence: SequencePaths,
        *,
        output_dir: Path,
        tag: str | None = None,
        additional_args: Iterable[str] = (),
    ) -> str:
        """
        便于日志记录的字符串表示。
        """
        command = self.build_command(
            entry_script=entry_script,
            sequence=sequence,
            output_dir=output_dir,
            tag=tag,
            additional_args=additional_args,
        )
        return command.format_shell()
