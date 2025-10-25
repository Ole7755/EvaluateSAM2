# scripts

集中存放所有可执行脚本，例如分割流程（`segment_video_with_first_mask.py`）、单目标实验（`attack_bear.py`）和评估工具（`evaluate_sam2_metrics.py`）。运行脚本前请确认所需配置、权重和数据路径均指向项目根目录下的对应文件夹。

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
