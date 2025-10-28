# 项目说明

本项目围绕 Segment Anything Model (SAM2) 的鲁棒性展开研究，目标是系统评估其在视频目标分割场景下的稳定性，并探索多种对抗攻击策略，以支持向模式识别（Pattern Recognition, PR）期刊投稿的实验工作。仓库主要用于本地编写脚本、整理实验流程，并与远程运行环境同步。

# 2025-10-27 攻击实验汇总

昨日本地同步了远程节点在 DAVIS 480p 序列上跑完的首帧攻击实验，覆盖 48 个序列和 4 种单帧扰动（FGSM / PGD / BIM / C&W）。汇总数据来自 `logs/<sequence>/attacks/<attack>/*.json`。

| 攻击 | 样本数 | 平均干净 IoU | 平均攻击后 IoU | 平均下降 | adv IoU = 0 序列数 |
| ---- | ---- | ------------- | --------------- | -------- | ------------------- |
| fgsm | 48   | 0.973         | 0.056           | 0.916    | 4                   |
| pgd  | 47   | 0.972         | 0.008           | 0.965    | 26                  |
| bim  | 47   | 0.972         | 0.020           | 0.952    | 19                  |
| cw   | 47   | 0.972         | 0.079           | 0.893    | 3                   |

- `bmx-bumps` 是四种攻击的共同薄弱样本，攻击后 IoU 均降至 0。
- `pgd` 和 `bim` 在 26/19 个序列上完全击破首帧分割（adv IoU = 0），需要重点关注传播阶段的劣化。
- `kid-football`（C&W）和 `koala`（BIM）保持了目前观察到的最高攻击后 IoU（≈0.5），可作为稳健序列的对比案例。
- `snowboard` 仅完成了 FGSM 攻击，其他三种攻击仍需补齐。

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
