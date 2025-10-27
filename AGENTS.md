项目概述
------
- 本仓库用于本地编写与管理脚本，实验 SAM2 在 DAVIS 视频目标分割上的流程，为后续针对 SAM2 的对抗鲁棒性研究做准备。
- 所有模型加载、推理和评估运行均在远程 Linux 环境完成；本地仅负责代码编辑与同步。

目录结构
------
- `configs/`：SAM2 配置文件（现有 `sam2_hiera_s.yaml`）。
- `data/`：数据集与素材（例如 `data/DAVIS/...`，可通过软链接指向远程同步目录）。
- `logs/`：记录评估结果与实验日志，如 `logs/<sequence>/metrics.csv`。
- `outputs/`：模型推理输出，按序列划分 `outputs/<sequence>/`。
- `scripts/`：全部可执行脚本。
- `weights/`：模型权重（如 `weights/sam2_hiera_small.pt`）。
- 根目录保留项目说明文件（`readme.md`、`requirements.txt` 等）。

脚本说明
------
- `scripts/attack_bear.py`：最小化示例，演示如何用首帧掩码初始化 `bear` 序列并把预测写入 `outputs/bear/`。
- `scripts/segment_video_with_first_mask.py`：批量拆分 DAVIS 首帧标签，调用官方 `SAM2VideoPredictor` 在整段视频上传播掩码，适合生成基线结果。
- `scripts/evaluate_sam2_metrics.py`：比较预测与真值掩码，逐帧打印 IoU/Dice，并可选写入 CSV 风格的日志文件。
- `scripts/inspect_davis_dataset.py`：汇总 DAVIS 序列的分辨率、帧数和标签占比，辅助挑选基线可靠的序列。
- `scripts/run_uap_attack.py`：首帧像素级对抗攻击入口，支持 FGSM / PGD / BIM / C&W，可同时记录干净基线和攻击后指标。
- `scripts/run_attacks_batch.sh`：批量跑完整个序列列表和四种攻击的 Bash 脚本，内置 metrics 跳过机制。
- `scripts/sam2tutorial.py`：单张图片的交互式示例。

脚本用法
------
- `scripts/segment_video_with_first_mask.py`  
  ```
  python3 scripts/segment_video_with_first_mask.py \
    --sequence bear \
    --resolution 480p \
    --mask-subdir Annotations \
    --frame-token 00000 \
    --verbose
  ```
  关键参数：`--sequence` 指定 DAVIS 序列；`--mask-subdir` 挑选标签来源（监督/无监督）；`--max-objects` 可截断对象数量；`--output-root` 默认为 `outputs/`。

- `scripts/evaluate_sam2_metrics.py`  
  ```
  python3 scripts/evaluate_sam2_metrics.py \
    --pred-dir outputs/bear \
    --gt-dir data/DAVIS/Annotations/480p/bear \
    --obj-id 1 \
    --gt-label 1 \
    --output logs/bear/metrics.csv
  ```
  若预测目录按 `*_id{obj}.png` 命名，可通过 `--obj-id` 选定对象；多标签真值可设定 `--gt-label`。

- `scripts/run_uap_attack.py`  
  ```
  python3 scripts/run_uap_attack.py \
    --sequence bear \
    --frame-token 00000 \
    --gt-label 1 \
    --obj-id 1 \
    --attack pgd \
    --epsilon 0.03 \
    --step-size 0.01 \
    --steps 40 \
    --random-start \
    --input-size 1024 \
    --mask-threshold 0.5 \
    --device cuda
  ```
  默认先用标准 `SAM2VideoPredictor` 计算干净基线，再执行攻击。`--attack` 支持 `fgsm/pgd/bim/cw`；PGD/BIM 可加 `--random-start`；C&W 需额外配置 `--cw-lr`、`--cw-confidence`、`--cw-binary-steps`；如需等比例缩放后 padding，可启用 `--keep-aspect-ratio`。

- `scripts/run_attacks_batch.sh`  
  ```
  ./scripts/run_attacks_batch.sh
  ```
  Bash 脚本会遍历内置序列与四种攻击；若 `outputs/<sequence>/<attack>/<frame>_<attack>_metrics.json` 已存在则自动跳过。可在脚本顶部的 `GT_LABELS` 字典中为多标签序列指定首帧标签。

- `scripts/attack_bear.py`  
  ```
  python3 scripts/attack_bear.py
  ```
  读取 `bear` 序列的首帧掩码，示例化 `add_new_mask` 接口并把预测写到 `outputs/bear/`。

- `scripts/inspect_davis_dataset.py`  
  ```
  python3 scripts/inspect_davis_dataset.py \
    --sequence bear \
    --resolution 480p \
    --mask-subdir Annotations
  ```
  打印指定序列的帧数、分辨率、各标签占比，可省略 `--sequence` 以遍历所有序列。

- `scripts/sam2tutorial.py`  
  ```
  python3 scripts/sam2tutorial.py --image path/to/image.jpg
  ```
  单图示例，用于确认环境和权重加载是否正常。

运行约定
------
- 运行脚本前，需要在远程环境准备好相应的数据集目录、模型配置与权重文件，路径默认指向 `configs/`、`weights/`、`data/`。
- 推理输出统一写入 `outputs/`，日志统一写入 `logs/`，如需自定义路径请修改脚本参数。
- 本地不进行网络下载或模型推理；若需新增依赖，请更新 `requirements.txt` 并在远程环境安装。

当前研究重点
----------
- 深入评估 SAM2 在多目标与单目标视频分割中的鲁棒性表现。
- 设计并实现针对 SAM2 的对抗攻击实验，比较不同攻击策略对分割质量（mIoU、Dice 等）的影响。
- 建立标准化的评估流程，便于在不同提示、攻击强度、数据扰动下快速对比结果。

进展记录
------
- 2025-10-24：完成 `walking` 序列双目标分割的排障，梳理多点提示策略与输出存储逻辑。
- 2025-10-25：整理仓库结构，统一脚本路径依赖；编写基于首帧掩码的单/多目标脚本；搭建 mIoU、Dice 评估工具。
- 2025-10-26：修复 `run_uap_attack` 前向封装，复用 `predictor.forward_image` 获取 FPN 投影特征，解决高分辨率特征通道错配；`bear` 序列 FGSM 攻击基线跑通并产出日志。
- 2025-10-26：扩展攻击脚本，自动追踪并缓存最佳 / 最差攻击案例的原图、对抗样本与扰动可视化，便于快速复盘；仅在基线分割足够好（clean IoU ≥ 0.5）时参与排名，焦点聚集在攻击导致显著退化的样本。
- 2025-10-26：新增 `inspect_davis_dataset.py` 工具脚本，汇总序列帧数、分辨率及首帧掩码标签占比，辅助筛选基线表现良好的 DAVIS 序列。
- 2025-10-26：为 `segment_video_with_first_mask.py` 增加命令行参数，便于针对任意序列/分辨率传播掩码并批量生成基线结果。

下一步计划
------


问题记录
------
- 2025-10-26：通过调用预测器自带的 `forward_image` 获取投影后的 FPN 特征并补充高分辨率分支输入，问题已解决。
