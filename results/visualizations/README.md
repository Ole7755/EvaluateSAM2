# visualizations

用于保存评估阶段生成的可视化图像，包括预测掩码与 GT 掩码的叠加图。默认结构：

```
results/visualizations/
  └── <dataset>/<sequence>/<tag>/<frame>.png
```

可通过 `--visualize-count` 控制输出帧数。若未同步原始图像，脚本会跳过可视化但仍计算指标。
