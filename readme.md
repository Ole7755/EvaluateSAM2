# 项目说明

本项目围绕 Segment Anything Model (SAM2) 的鲁棒性展开研究，目标是系统评估其在视频目标分割场景下的稳定性，并探索多种对抗攻击策略，以支持向模式识别（Pattern Recognition, PR）期刊投稿的实验工作。仓库主要用于本地编写脚本、整理实验流程，并与远程运行环境同步。

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
