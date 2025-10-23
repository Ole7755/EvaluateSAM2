项目目的与范围
--------------
- 本仓库用于本地编写代码，学习与实验 SAM2 在 DAVIS 上的视频目标分割流程；完成学习与验证后，将基于 SAM2 开展对抗攻击实验。
- 所有运行（执行脚本、加载模型、保存权重/结果）均在远程 Linux 上完成；本地仅进行代码编写与同步。

协作与约定（供智能体参考）
--------------------------
- 未经用户明确指令，不要擅自写代码或改动文件；有不确定时先在对话中给出思路与片段。
- 优先复用官方 SAM2 API/示例，避免重复造轮子。
- 改动应最小化，聚焦当前目标；避免无关重构。
- 本地不进行网络下载；模型/数据的获取在远程环境按用户指示进行。

环境与文件
----------
- 远程环境：Python 3.10（Linux）。
- 依赖：使用仓库内 `requirements.txt`，远程已安装 `sam2`。
- 模型文件位于仓库根目录：
  - 权重：`sam2_hiera_small.pt`
  - 配置：`sam2_hiera_s.yaml`

DAVIS 数据布局
---------------
- 数据集根相对路径：`DAVIS/`
- 帧：`DAVIS/JPEGImages/480p/<sequence>/<frame>.jpg`
- 掩码（真值）：`DAVIS/Annotations/480p/<sequence>/<frame>.png`
- 学习示例序列：`bear`（480p）。

SAM2 典型使用流程（视频）
-------------------------
1) 构建视频预测器（使用根目录下配置与权重）：
   - `predictor = build_sam2_video_predictor("sam2_hiera_s.yaml", "sam2_hiera_small.pt")`
2) 使用 DAVIS 的 JPEG 目录初始化视频状态：
   - `state = predictor.init_state("DAVIS/JPEGImages/480p/<sequence>")`
   - 传入目录字符串即可，预测器内部会按字典序读取 `.jpg` 帧。
3) 在某一帧添加提示（像素坐标，故 `normalize_coords=False`）：
   - `predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, points=..., labels=..., normalize_coords=False)`
   - 点格式：`[[x, y], ...]`；标签：`1`=前景，`0`=背景。
4) 传播至整段视频并保存掩码：
   - `for frame_idx, object_ids, masks in predictor.propagate_in_video(state):`
     - 单目标：直接取 `masks[0]` 保存。
     - 保存前转换：`(mask > 0.5).to(torch.uint8).cpu().numpy() * 255`。

注意事项与常见坑
----------------
- 坐标规范：
  - 传像素坐标时用 `normalize_coords=False`；若设为 True，需要按宽高归一化到 [0,1]。
- 对象 ID 与位置索引：
  - `object_ids` 是对象 ID；`masks` 按“位置”堆叠。单目标时 `masks[0]` 就是该对象掩码，即使 `obj_id` 是 1。
- 背景点选择：
  - 简单做法是在前景外部选择任意背景像素，并确保坐标不越界。
- 掩码保存：
  - 确保输出目录存在（如 `outputs/<sequence>`），并将掩码转换为 `uint8` 的 0/255 后再保存 PNG。

目录与产物
----------
- 按当前约定，示例/实验脚本先放在仓库根目录。
- 远程运行时将输出保存到 `outputs/<sequence>/`。

后续计划
--------
- 在学习 SAM2 的视频分割流程后，基于 DAVIS 序列实现并评测针对 SAM2 的对抗攻击实验；具体模块/脚本在用户确认方案后再新增。
