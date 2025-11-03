# SAM2 Evaluation Toolkit

本仓库专注于 **评估** Segment Anything Model 2 (SAM2) 在多数据集场景下的表现。macOS 端只负责代码重构、配置管理与结果整理；所有推理任务均在远程 Linux GPU 环境完成。

## 目录结构

```
project/
├── sam2/                   # SAM2 官方源码镜像或子模块
├── configs/                # SAM2 配置模板
├── data/
│   ├── davis/
│   ├── mose/
│   └── vos/
├── src/                    # 评估核心模块
├── results/
│   ├── visualizations/     # 可视化输出
│   ├── metrics/            # 指标 CSV/JSON
│   └── comparisons/        # 对比图表
└── main.py                 # 评估入口脚本
```

## 快速开始

1. **同步依赖**
   ```bash
   pip install -r requirements.txt
   ```
2. **挂载数据与权重**
   - 在 `data/` 下为 `davis/`、`mose/`、`vos/` 创建软链接，指向远程 Linux 上的实际数据（或直接在运行时使用 `--images-dir` / `--gt-dir` 指定路径）。
   - SAM2 权重统一放在 `sam2/checkpoints/`，运行脚本时通过 `--sam2-config` 与 `--checkpoint` 指定对应文件。
   - 若尚未安装 SAM2 Python 包，请执行 `pip install -e sam2`（需在仓库根目录下运行）。
3. **准备评估标签（可选）**  
   评估时可通过 `--tag`（或其他自定义标识）记录实验信息，便于结果对比。

## 运行评估

`main.py` 会读取预测与 GT 掩码，计算 IoU / Dice 等指标，同时生成可选可视化。

```bash
python3 main.py \
  --dataset davis \
  --sequence bear \
  --resolution 480p \
  --tag exp1 \
  --images-dir data/davis/DAVIS/JPEGImages/480p/bear \
  --gt-dir data/davis/DAVIS/Annotations_unsupervised/480p/bear \
  --sam2-config configs/sam2.1_hiera_small.yaml \
  --checkpoint sam2/checkpoints/sam2.1_hiera_small.pt \
  --summary-json results/metrics/bear_exp1_summary.json \
  --visualize-count 10 \
  --save-pred-masks \
  --device cuda:0
```

- 默认使用 GT 掩码生成点+框提示（`--prompt-type point_box`），可通过 `--prompt-type` 切换为 `point` 或 `box`。
- `--background-points` 用于在点提示模式下随机采样背景点数量；`--seed` 控制采样复现。
- 若需启用 SAM2 的多掩码输出以选择评分最高的一张，可添加 `--multimask-output`。
- `--mask-threshold` 控制预测掩码二值化阈值（默认 0.5）。
- 若需一次评估多个序列，可使用 `--sequences seq1 seq2`、`--sequence-list list.txt` 或 `--all-sequences`。批量模式下会为每个序列生成独立的指标、可视化与掩码文件夹（不支持同时指定 `--images-dir` / `--gt-dir`）。

- 结果 CSV 默认位于 `results/metrics/<dataset>_<sequence>_<tag>.csv`。
- 可视化输出位于 `results/visualizations/<dataset>/<sequence>/<tag>/`。
- 若未提供 `--images-dir` 或 `--gt-dir`，脚本会尝试根据 `data/<dataset>/` 的默认布局推断路径；当目录结构不同步时请显式传参。
- 加上 `--save-pred-masks` 可将预测掩码写入 `results/comparisons/<dataset>/<sequence>/<tag>/`。
- `--images-dir` 与 `--gt-dir` 建议直接指向具体序列的帧/掩码目录，例如 `data/davis/DAVIS/JPEGImages/480p/bear`。 

## 远程推理与命令生成

`src/model_inference.py` 提供 `SAM2InferenceRunner`，用于根据权重、配置和序列信息构建远程执行命令：

```python
from pathlib import Path
from src.data_loader import SequenceSpec, resolve_sequence_paths
from src.model_inference import SAM2InferenceConfig, SAM2InferenceRunner

spec = SequenceSpec(dataset="davis", sequence="bear", resolution="480p")
paths = resolve_sequence_paths(spec)
cfg = SAM2InferenceConfig(
    checkpoint=Path("sam2/checkpoints/sam2.1_hiera_small.pt"),
    config=Path("configs/sam2.1_hiera_small.yaml"),
    remote_workspace=Path("/remote/workspaces/AttackSAM2"),
)
runner = SAM2InferenceRunner(cfg)
command = runner.describe("main.py", paths, output_dir=Path("results/comparisons/bear/exp1"))
print(command)
```

生成的命令可直接在远程终端运行，或提交到集群调度器。

## 结果整理建议

- 指标：统一保存到 `results/metrics/`，可额外输出 JSON 摘要以便自动化汇总。
- 可视化：将预测与 GT 叠加图保存到 `results/visualizations/`，对比图表放在 `results/comparisons/`。
- 预测：若需长期保留推理掩码，可在 `results/comparisons/` 下建立子目录或单独的存储路径。

## 依赖

核心依赖包含 `torch`、`numpy`、`Pillow`、`matplotlib` 等。若在本地新增依赖，请记得更新 `requirements.txt` 并在远程环境重新安装。
