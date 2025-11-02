# comparisons

用于存放对比图表与汇总可视化，例如不同实验设置或不同权重的指标曲线、示例帧拼接结果等。若运行 `main.py` 时加上 `--save-pred-masks`，预测掩码也会保存在此目录的 `<dataset>/<sequence>/<tag>/` 子路径下。

建议根据需要创建分组，例如：
```
results/comparisons/
├── davis/
│   ├── exp1/
│   │   └── 00000.png
│   └── summary_plots.png
└── vos/
    └── baseline_vs_variant.pdf
```
