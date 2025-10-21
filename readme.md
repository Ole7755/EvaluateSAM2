#下载sam：
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
#下载权重：
wget https://huggingface.co/facebook/sam2-hiera-small/resolve/main/sam2_hiera_small.pt
wget https://huggingface.co/facebook/sam2-hiera-small/resolve/main/sam2_hiera_s.yaml
