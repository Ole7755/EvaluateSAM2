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
├── models/
│   └── sam2_weights/       # 权重占位说明
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
   - 在远程环境中准备 SAM2 权重，保持与 `models/sam2_weights/` 约定的命名一致。
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
  --pred-dir /path/to/predictions \
  --images-dir /path/to/images \
  --gt-dir /path/to/gt_masks \
  --summary-json results/metrics/bear_exp1_summary.json \
  --visualize-count 10
```

- 结果 CSV 默认位于 `results/metrics/<dataset>_<sequence>_<tag>.csv`。
- 可视化输出位于 `results/visualizations/<dataset>/<sequence>/<tag>/`。
- 若 `--pred-dir` / `--gt-dir` 省略，则基于数据/结果目录的约定自动推断。
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
    checkpoint=Path("models/sam2_weights/sam2_hiera_small.pt"),
    config=Path("configs/sam2_hiera_s.yaml"),
    remote_workspace=Path("/remote/workspaces/AttackSAM2"),
)
runner = SAM2InferenceRunner(cfg)
command = runner.describe("main.py", paths, output_dir=Path("results/visualizations"))
print(command)
```

生成的命令可直接在远程终端运行，或提交到集群调度器。

## 结果整理建议

- 指标：统一保存到 `results/metrics/`，可额外输出 JSON 摘要以便自动化汇总。
- 可视化：将预测与 GT 叠加图保存到 `results/visualizations/`，对比图表放在 `results/comparisons/`。
- 预测：若需长期保留推理掩码，可在 `results/comparisons/` 下建立子目录或单独的存储路径。

## 依赖

核心依赖包含 `torch`、`numpy`、`Pillow`、`matplotlib` 等。若在本地新增依赖，请记得更新 `requirements.txt` 并在远程环境重新安装。
