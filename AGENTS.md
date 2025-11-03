项目目标
------
- 聚焦评估 SAM2 模型在不同视频数据集上的分割表现，核心任务是比较 SAM2 输出的掩码与 ground truth 的差异。
- 本地（macOS）负责维护评估代码、整理配置与结果；所有推理在远程 Linux GPU 环境完成。

评估范围
------
- 关注已有模型或补丁的性能对比，不在本仓库内开展攻击/训练任务。
- 支持对单序列与多序列的掩码预测结果计算核心指标（IoU、Dice 等），对齐预测与 GT 的对应关系。
- 统一记录推理输出、指标摘要与可视化，便于快速回溯与横向对比。

目录指引
------
- `sam2/`：上游 SAM2 官方源码（以子模块或镜像形式同步），其 `checkpoints/` 目录存放模型权重。
- `configs/`：评估所需的 SAM2 配置模板，保持与远程环境一致。
- `data/`：数据入口目录，按数据集划分，每个数据集下采用 `annotations/` 与 `images/` 的镜像结构，必要时通过软链接指向远程数据。
- `src/`：评估核心模块（数据加载、prompt 生成、推理编排、指标计算与可视化）。
- `results/`：评估输出目录，统一保存 `metrics/`、`visualizations/` 与 `comparisons/` 等产物。
- `main.py`：评估入口脚本，支持命令行指定数据集、序列、实验标签，可通过 `--images-dir` / `--gt-dir` 显式传入路径，并基于 GT 掩码自动生成点/框提示调用 SAM2 进行推理。

核心脚本
------
- `main.py`：对每帧图像调用 SAM2 生成分割掩码，与 GT 比对后输出 CSV / JSON / 可视化结果。
- `src/data_loader.py`：解析多数据集目录布局，管理序列与帧路径。
- `src/prompt_generator.py`：依据 GT 掩码生成 SAM2 所需 prompt（点/框/掩码）。
- `src/model_loader.py`：加载 SAM2 图像预测器，便于在评估脚本中直接调用。
- `src/model_inference.py`：构建远程 Linux 环境上的 SAM2 推理命令。
- `src/evaluator.py`：实现 IoU / Dice / Precision / Recall 计算与结果汇总。
- `src/visualizer.py`：叠加掩码并输出对比图像，辅助人工质检。

评估流程约定
------
- 远程 Linux 需预先配置好 SAM2 环境、权重与数据路径；本地脚本负责生成命令与同步结果。
- 评估输出（可视化、指标等）统一写入 `results/` 目录。
- 若新增依赖或配置，需同步更新 `requirements.txt` 与 `configs/`，并在远程环境完成安装。
- 当数据目录布局不符合默认模板时，请在运行脚本时通过 `--images-dir` 与 `--gt-dir` 指定实际路径，并确保提供正确的 `--sam2-config` 与 `--checkpoint`（默认存放于 `sam2/checkpoints/`）。
- 编写新功能时优先复用 `src/` 下模块，保持数据加载、指令构建和指标计算的一致性。
- `main.py` 支持通过 `--sequences` / `--sequence-list` / `--all-sequences` 批量评估多个序列，运行多序列时需遵循默认目录布局（不再单独传入 `--images-dir` / `--gt-dir`）。

常见核对项
------
- 预测掩码或原始帧缺失时，首先检查远程路径及本地软链接是否生效。
- 汇总 CSV / JSON 前确认字段是否包含序列名、帧数、IoU、Dice 等关键信息。
- 对比不同配置或实验标签时，保持相同的阈值与扰动幅度记录，确保可复现性。
- 若模型加载失败，请确认已执行 `pip install -e sam2` 并传入 `sam2/checkpoints/` 下存在的权重文件与对应配置。
