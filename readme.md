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
   - 准备待评估数据集的图像目录与 GT 目录，可放置在任意位置；运行脚本时需显式通过 `--images-dir` / `--gt-dir`（单序列）或 `--images-root` / `--gt-root`（批量）传入。
   - SAM2 权重统一放在 `sam2/checkpoints/`，运行脚本时通过 `--sam2-config` 与 `--checkpoint` 指定对应文件。
   - 若尚未安装 SAM2 Python 包，请执行 `pip install -e sam2`（需在仓库根目录下运行）。
3. **准备评估标签（可选）**  
   评估时可通过 `--tag`（或其他自定义标识）记录实验信息，便于结果对比。

## 运行评估

`main.py` 会读取预测与 GT 掩码，计算 IoU / Dice / Precision / Recall，并按需输出 CSV、JSON 及可视化结果。无论单序列还是批量评估，都需要显式传入图像与 GT 的根目录。

### 数据目录要求
- **单序列**：`--images-dir` 与 `--gt-dir` 指向同一序列的帧目录与掩码目录，目录内的文件名（不含后缀）需一致。
- **批量序列**：`--images-root` 与 `--gt-root` 指向包含多个序列子目录的根目录，子目录名称需与序列名一致。批量模式会在根目录下自动拼接各序列路径。

### 单序列示例
```bash
python3 main.py \
  --sequence bear \
  --resolution 480p \
  --dataset-label davis \
  --images-dir /mnt/data/davis/JPEGImages/480p/bear \
  --gt-dir /mnt/data/davis/Annotations/480p/bear \
  --sam2-config configs/sam2.1_hiera_small.yaml \
  --checkpoint sam2/checkpoints/sam2.1_hiera_small.pt \
  --tag exp1 \
  --summary-json results/metrics/bear_exp1_summary.json \
  --visualize-count 10 \
  --save-pred-masks \
  --device cuda:0
```

### 批量评估多个序列
```bash
python3 main.py \
  --all-sequences \
  --resolution 480p \
  --dataset-label davis \
  --images-root /mnt/data/davis/JPEGImages/480p \
  --gt-root /mnt/data/davis/Annotations/480p \
  --prompt-types point box point_box \
  --background-points 16 \
  --sam2-config configs/sam2.1_hiera_small.yaml \
  --checkpoint sam2/checkpoints/sam2.1_hiera_small.pt \
  --results-root results \
  --save-pred-masks \
  --summary-json results/metrics/davis_full_summary.json
```
> 也可以用 `--sequences seq1 seq2` 手动列出序列，或通过 `--sequence-list list.txt` 读取序列清单（文件每行一个序列名）。

### 常用参数速览
- `--prompt-type` / `--prompt-types`：控制使用点、框或点+框提示；提供 `--prompt-types` 时会依次执行多种提示并输出独立结果。
- `--dataset-label`：自定义结果目录与汇总中显示的数据集标签，未指定时默认使用 `default`。
- `--background-points`：点提示时从背景随机采样负点数量；与 `--seed` 配合可复现采样。
- `--multimask-output`：启用 SAM2 的多掩码输出（默认关闭，会在返回的三个掩码中选取得分最高的一个）。
- `--mask-threshold`：将概率掩码阈值化为二值图的阈值（默认 0.5）。
- `--batch-size`：控制每批一起推理的帧数，>1 时会启用批量推理以更好地利用 GPU（需根据显存自行调整）。
- `--save-pred-masks`：开启后把预测掩码保存到 `results/comparisons/<datasetlabel>/<sequence>/<tag>/<prompt>`。
- `--visualize-dir` / `--visualize-count`：自定义可视化输出目录以及每个序列生成的叠加图数量。
- `--report-csv` / `--summary-json`：分别控制指标 CSV 与汇总 JSON 的输出路径；多序列或多提示时脚本会自动附加序列名/提示类型后缀。

所有命令都会在终端打印最终指标汇总，同时将详细数据写入 `results/metrics/`（CSV）与可选的 `summary-json` 文件中，方便后续比对。

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
