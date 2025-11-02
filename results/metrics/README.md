# metrics

逐帧与汇总指标输出目录。`main.py` 默认在此写入 `<dataset>_<sequence>_<tag>.csv`，字段包含 IoU、Dice、Precision、Recall 以及 SAM2 预测的置信度（`predicted_iou`）。

推荐约定：
- 若存在多次实验，可在文件名前加入时间戳或超参数缩写，例如 `20250105_expA.csv`。
- 对于批量评估，可在上层追加子目录（如 `metrics/davis/experimentA/`），保持数据集与实验标签的清晰层级。
- 若需要 JSON 摘要，请在命令行通过 `--summary-json` 指向 `results/metrics/summary/` 等自定义位置。 
