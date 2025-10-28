# 项目说明

本项目围绕 Segment Anything Model (SAM2) 的鲁棒性展开研究，目标是系统评估其在视频目标分割场景下的稳定性，并探索多种对抗攻击策略，以支持向模式识别（Pattern Recognition, PR）期刊投稿的实验工作。仓库主要用于本地编写脚本、整理实验流程，并与远程运行环境同步。

# 2025-10-28 通用补丁 (10/255) 评估汇总

使用 `10/255 (≈0.0392)` 的 L∞ 约束，我们在 `bear` 首帧上训练了四种攻击的通用补丁，并在其余 89 个 DAVIS 480p 序列上评估迁移效果。结果记录在：

```
logs/uap/<attack>_bear/pipeline_eps10over255.json
```

| 攻击 | 样本数 | 平均干净 IoU | 平均攻击后 IoU | ΔIoU | 平均干净 Dice | 平均攻击后 Dice | ΔDice | 最高残余序列 (IoU) | 完全失效序列 |
| ---- | ---- | ------------- | --------------- | ---- | --------------- | ----------------- | ------ | -------------------- | -------------- |
| FGSM | 89   | 0.960         | 0.060           | 0.900 | 0.975           | 0.105             | 0.870 | kid-football (0.400) | tuk-tuk (0.000) |
| PGD  | 89   | 0.960         | 0.058           | 0.902 | 0.975           | 0.102             | 0.873 | kid-football (0.412) | tuk-tuk (0.000) |
| BIM  | 89   | 0.960         | 0.055           | 0.905 | 0.975           | 0.096             | 0.878 | kid-football (0.456) | tuk-tuk (0.000) |
| C&W  | 89   | 0.960         | 0.087           | 0.873 | 0.975           | 0.142             | 0.833 | bus (0.634)          | tuk-tuk (0.000) |

- 四种补丁均显著降低了迁移序列的首帧 IoU，BIM / PGD 在 L∞ = 10/255 下平均 ΔIoU 超过 0.90。
- `kid-football` 在 FGSM/PGD/BIM 攻击下仍保留 0.40+ 的首帧 IoU，可作为稳健案例；C&W 则在 `bus` 序列上留下最高残余 (0.634)。
- `tuk-tuk` 对所有补丁完全失效 (adv IoU = 0)，说明该序列对首帧扰动尤为敏感，适合进一步分析传播阶段的劣化。
- 每份 JSON 中包含补丁范数、训练/测试明细与历史损失，可直接复现输出结果。

# 下载 SAM2 源码

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

# 下载 SAM2 权重与配置

```bash
wget https://huggingface.co/facebook/sam2-hiera-small/resolve/main/sam2_hiera_small.pt
wget https://huggingface.co/facebook/sam2-hiera-small/resolve/main/sam2_hiera_s.yaml
```
