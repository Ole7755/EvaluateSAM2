# scripts

集中存放所有可执行脚本，例如分割流程（`segment_video_with_first_mask.py`）、单目标实验（`attack_bear.py`）和评估工具（`evaluate_sam2_metrics.py`）。运行脚本前请确认所需配置、权重和数据路径均指向项目根目录下的对应文件夹。

## 多目标首帧传播

- `segment_video_with_first_mask.py`：从指定序列的首帧掩码初始化对象，并传播整段视频的预测掩码。支持通过参数选择分辨率、掩码目录及对象数量：
  ```bash
  python -m scripts.segment_video_with_first_mask \
    --sequence koala \
    --resolution 480p \
    --mask-subdir Annotations_unsupervised \
    --max-objects 3 \
    --verbose
  ```
  输出默认保存在 `outputs/<sequence>/`，掩码命名格式为 `frame_id_id<number>.png`。

## 新增：通用扰动攻击脚本

- `run_uap_attack.py`：在首帧上对 SAM2 施加 FGSM / PGD / BIM / C&W 攻击，生成通用扰动（UAP），并保存攻前攻后预测、扰动张量及指标日志。  
  - 推荐以模块方式运行，自动解析包内相对导入：  
    ```bash
    python -m scripts.run_uap_attack \
      --sequence bear \
      --attack pgd \
      --epsilon 0.03 \
      --step-size 0.01 \
      --steps 40 \
      --gt-label 1
    ```
  - 日志写入 `logs/<sequence>/attacks/<attack>/`，扰动和可视化结果存放在 `outputs/<sequence>/<attack>/`。

## 新增：数据集信息脚本

- `inspect_davis_dataset.py`：遍历 `data/DAVIS` 下的序列，统计帧数、分辨率，以及可选的首帧掩码标签占比，用于快速挑选基线效果较好的样本。  
  - 默认读取 `JPEGImages/<resolution>` 与 `Annotations_unsupervised/<resolution>`，可通过参数覆盖：  
    ```bash
    python -m scripts.inspect_davis_dataset --resolution 480p --show-mask-stats
    ``` 
  - 支持 `--output-json path/to/summary.json` 将统计信息保存为文件。
