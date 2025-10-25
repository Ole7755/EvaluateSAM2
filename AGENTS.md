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
- `scripts/attack_bear.py`：使用首帧掩码初始化单目标视频分割（序列 `bear`），并将结果写入 `outputs/bear/`。
- `scripts/segment_video_with_first_mask.py`：将带实例标签的首帧掩码拆分为多个对象并传播（默认序列 `walking`），结果保存到 `outputs/<sequence>/`。
- `scripts/evaluate_sam2_metrics.py`：对比预测与真值掩码，输出逐帧与平均的 mIoU、Dice，可通过 `--output` 将指标写入日志文件。
- `scripts/sam2tutorial.py`：单张图片的 SAM2 交互示例，输出保存在 `outputs/tutorial/`。

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
- 调研并实现首批对抗攻击方法（例如扰动提示点、像素级对抗扰动），衡量对 SAM2 分割性能的影响。
- 根据实验需要扩展脚本参数化程度（序列选择、对象数量、攻击配置等），确保流程可复用、可批量化。
- 攻击脚本中 clean IoU 与原生流程对齐：去除额外缩放带来的失真或复用 `add_new_mask` + `propagate` 流程，减少基线差异。

问题记录
------
- 2025-??-??：`python -m scripts.run_uap_attack --sequence bear --attack fgsm --epsilon 0.03 --gt-label 1 --device cuda` 运行失败，报错：
  ```
  RuntimeError: The size of tensor a (32) must match the size of tensor b (256) at non-singleton dimension 1
  ```
-  定位在 `sam2/modeling/sam/mask_decoder.py` 的 `predict_masks` 函数。正在排查高分辨率特征与解码器输入对齐问题。
-  2025-10-26：通过调用预测器自带的 `forward_image` 获取投影后的 FPN 特征并补充高分辨率分支输入，问题已解决。
