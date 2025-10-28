项目概述
------
- 本仓库用于本地编写与管理脚本，以便在远程 Linux 环境上实验 SAM2 模型在 DAVIS 数据集上的对抗攻击与评估流程。
- 所有推理、训练与评估需在远程 GPU 环境执行；本地仅负责代码维护与同步。

目录结构
------
- `configs/`：SAM2 配置文件。
- `data/`：DAVIS 数据集（可通过软链接指向远程同步目录）。
- `logs/`：实验日志与指标输出。
- `outputs/`：模型推理与补丁可视化结果。
- `scripts/`：主要脚本与可复用“轮子”模块。
  - `scripts/wheels/`：集中存放攻击、训练、数据处理等可复用组件。
- `weights/`：SAM2 模型权重文件。

核心脚本
------
- `scripts/run_bear_uap_pipeline.py`：统一的 UAP 训练与评估管线，支持 `fgsm/pgd/bim/cw`。
- `scripts/run_bear_fgsm_uap_pipeline.py` / `run_bear_pgd_uap_pipeline.py` / `run_bear_bim_uap_pipeline.py` / `run_bear_cw_uap_pipeline.py`：针对单一攻击的便捷入口，内部复用通用管线。
- `scripts/run_uap_attack.py`：单序列像素级攻击脚本，可生成干净与对抗指标。
- `scripts/train_universal_patch.py`：多序列联合训练通用补丁。
- `scripts/evaluate_uap_patch.py`：加载已有补丁并在指定序列上批量评估。
- `scripts/evaluate_sam2_metrics.py`：对预测掩码目录进行 IoU/Dice 评估。

使用示例
------
- 在 bear 上训练 PGD 补丁并评估其它序列：
  ```
  python3 scripts/run_bear_pgd_uap_pipeline.py \
    --output logs/uap/pgd_bear/pipeline_eps10over255.json \
    --gt-label 1 \
    --obj-id 1 \
    --epsilon 0.0392156862745098 \
    --step-size 0.0392156862745098 \
    --steps 40 \
    --random-start \
    --device cuda
  ```
- 评估已有补丁：
  ```
  python3 scripts/evaluate_uap_patch.py \
    --attack pgd \
    --patch-path logs/uap/pgd/pgd_uap_patch_xxxx.pt \
    --sequences bear,boat \
    --frame-token 00000 \
    --output logs/uap/pgd/eval_summary.json
  ```

运行约定
------
- 运行脚本前需确保远程环境已准备好配置、权重与数据集。
- 推理输出统一写入 `outputs/`，指标与日志统一写入 `logs/`。
- 本地仓库只做代码编辑与同步，新增依赖需更新 `requirements.txt` 后在远程安装。
- **代码复用优先**：新增功能请优先从 `scripts/wheels/` 导入现有组件，避免重复造轮子。
- **扰动幅度约定**：通用补丁或像素级攻击的默认 L∞ 扰动上限为 `10/255`（约 0.0392156863），除非实验需要特别对比。

备注
------
- 如果脚本遇到数据缺失或权限错误，请优先检查远程数据路径与软链接是否正确。
- 所有结果文件请保存在当前仓库的 `logs/` 与 `outputs/` 目录，便于共享与复现。
