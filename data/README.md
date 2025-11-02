# data

数据入口根目录，覆盖多种视频/图像分割数据集。通用目录约定：

```
data/
└── dataset_name/
    ├── annotations/   # Ground truth masks，与原图同名
    │   ├── video1/
    │   │   ├── 00000.png
    │   │   └── 00001.png
    │   └── video2/
    │       └── 00000.png
    └── images/        # 原始图像或视频帧
        ├── video1/
        │   ├── 00000.jpg
        │   └── 00001.jpg
        └── video2/
            └── 00000.jpg
```

- 所有帧或图像文件命名须在 `annotations/` 与 `images/` 之间保持一致，确保可一一匹配。
- 连续帧（视频序列）按子目录存放；单张图像可直接置于 `images/` 根目录。
- 若需要扩展额外层级（如 train/val split），请保持 `annotations/` 与 `images/` 的相对结构一致，并在 `SequenceSpec` 中通过 `rgb_layout` / `mask_layout` 自定义。
- 本地 macOS 可通过软链接指向远程 Linux 的实际存储路径，保持一致的目录树。 
