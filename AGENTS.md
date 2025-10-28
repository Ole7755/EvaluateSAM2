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
- `scripts/train_universal_patch.py`：多序列联合训练通用补丁的入口脚本，支持 FGSM / PGD / BIM / C&W，输出日志与可视化。

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
- 运行脚本前，请在远程环境准备好数据集、配置与权重（默认路径指向 `configs/`、`weights/`、`data/`）。
- 推理输出统一写入 `outputs/`，日志统一写入 `logs/`，如需自定义路径请修改脚本参数。
- 本地仓库仅负责代码编辑与同步，不直接执行大规模推理；新增依赖需更新 `requirements.txt` 后在远程环境安装。
- **代码复用优先**：实现新功能时应尽量复用现有脚本、工具函数与模块，避免重复造轮子。

未来工作
------
- **通用扰动训练**：搭建跨序列 UAP 优化流程（多序列、首帧联合训练），跟踪平均与最坏 IoU。
- **多类型攻击**：探索提示级扰动、混合攻击或多帧攻击手段，分析 propagate 阶段的脆弱点。
- **自动化评估**：批量汇总 `logs/` 与 `outputs/`，生成 per-attack/per-sequence 报告（首帧 + 全视频指标）。
- **传播研究**：评估首帧扰动对后续帧的影响，必要时设计跨帧 UAP 以破坏长期跟踪。

近期计划 —— UAP 通用对抗补丁
------
- **数据准备**：挑选 2~3 个 DAVIS 序列构成训练集，并预留若干序列做验证；统一缓存首帧图像、掩码及尺寸信息，减少重复预处理的开销。
- **训练框架**：实现 `UniversalPatchTrainer`，维护一个可学习的补丁张量，对每个样本应用补丁后通过 `SAM2ForwardHelper` 前向，累积梯度并在每步后做 `[-ε, ε]` 投影与像素裁剪。
- **攻击算法**：基于现有 FGSM、PGD、BIM、C&W 攻击实现补丁更新流程；FGSM 单步聚合梯度，PGD/BIM 多步迭代（PGD 支持随机初始化），C&W 使用 Adam 与二分搜索平衡常数并附加 L∞ 约束。
- **日志与可视化**：在 `outputs/uap/<attack>/` 保存补丁、干净/对抗对比图，在 `logs/uap/<attack>/metrics.csv` 记录每个序列的 clean/adv IoU、Dice、ΔIoU 及扰动范数。
- **评估步骤**：先在训练序列上验证补丁有效，再迁移至验证序列检验跨序列泛化；同时对比 clean baseline，分析不同攻击的性能差异并据此调整超参数。
