# src

评估流程的核心模块。结构概览：

- `data_loader.py`：数据集布局解析与帧检索工具。
- `prompt_generator.py`：从 GT 掩码生成 SAM2 prompt 组合。
- `model_inference.py`：构建远程推理命令，封装环境变量与路径映射。
- `evaluator.py`：指标计算与结果汇总。
- `visualizer.py`：生成叠加可视化图。

引导：
- 模块间尽量保持纯函数或轻量类，便于在本地进行单元测试。
- 若需新增功能，请在此目录下扩展相应模块，并在 `main.py` 中引入。
- 远程执行相关的重型逻辑应放在 Linux 环境运行，本地只负责配置与命令拼装。 
