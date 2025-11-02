"""
项目评估核心逻辑所在的包。

模块划分：
- data_loader：数据集路径解析与帧序列管理。
- prompt_generator：从 GT 掩码构造 SAM2 所需提示。
- model_inference：SAM2 远程推理封装。
- evaluator：预测指标计算与整合。
- visualizer：可视化工具。
"""

__all__ = [
    "data_loader",
    "prompt_generator",
    "model_inference",
    "evaluator",
    "visualizer",
]
