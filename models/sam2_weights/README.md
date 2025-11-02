# sam2_weights

SAM2 权重说明目录。本地仅存放 README 与占位结构，实际权重文件位于远程 Linux GPU 环境并通过同步工具挂载。

推荐命名示例：
- `sam2_hiera_tiny.pt`
- `sam2_hiera_small.pt`
- `sam2_hiera_base.pt`
- `sam2_hiera_large.pt`

更新权重后请同步修改远程配置，并在 `SAM2InferenceConfig` 中指向正确的文件路径。 
