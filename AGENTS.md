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

进展记录（排障日志）
------------------
- 2025-10-24：多目标（两人）分割学习与排障（序列：`walking`，DAVIS 2017，480p）。
  - 使用同一流程：`build_sam2_video_predictor` → `init_state("DAVIS/JPEGImages/480p/<seq>")` → 第0帧为多对象添加提示 → `propagate_in_video`。
  - 提示点（像素坐标，`normalize_coords=False`）
    - obj_id=1（近景右侧）：`[[560,235],[600,320],[740,365],[540,165]]` 前景；`[[820,260],[820,440],[650,100]]` 背景。
    - obj_id=2（中左远景）：`[[365,250],[370,320],[355,200]]` 前景；`[[150,240],[220,440],[260,260]]` 背景。
  - 代码修正：
    - 修复保存循环作用域，将阈值化与保存放到对象循环内部，确保每帧按 `obj_id` 单独保存掩码（此前只保存最后一个对象）。
    - 添加 `add_new_points_or_box` 后的面积调试输出（>0.5 前景像素数）。
  - 现象：
    - 首帧分割质量差，概率图将路面大面积作为前景；二值图接近全黑。
    - 一度出现仅 `id1` 正常、`id2` 失败；之后两者均失败。
  - 假设与排查方向：
    - 坐标使用差异：可能存在 `[x,y]` 与 `[y,x]` 顺序不符，或误将像素坐标按归一化处理（`normalize_coords=True`）。
    - 提示歧义：`id2` 与 `id1` 相近且遮挡，需为 `id2` 添加“强负样本”（在 `id1` 掩码上采样负点），或增加一个紧框提示。
  - 已给出的调试脚本思路（本地不改文件，远程可运行）：
    - 方案A：用首帧 GT 自动生成稳健前景/背景点（质心+四极值+边缘背景），即时检查首帧面积并保存 `00000_id*_prob.png`、`00000_id*.png`。
    - 方案B：网格排查 4 种组合并保存首帧结果：`xy/yx` × `pixel/norm`，目录：`output/<seq>_debug/<variant>/`，打印各对象面积，选择有效组合。
    - 方案C：先添加 `id1`，从其首帧掩码中采样若干负点，作为 `id2` 的强负约束；或给 `id2` 加紧框（如 `[300,150,420,420]`）。
  - 输出目录：当前调试脚本使用 `output/<seq>` 与 `output/<seq>_debug/`；后续与本文件约定对齐为 `outputs/<sequence>/`。
  - 明日待办：
    - 运行方案B，确定正确的坐标顺序与是否归一化设置；若无有效组合，转方案C。
    - 确认后产出最小化多目标脚本（仅像素坐标、`normalize_coords=False`），统一输出到 `outputs/<seq>/`。
    - 评估首帧与全视频质量，记录 IoU 与失败案例，再决定是否引入框提示或更多负点。
- 2025-10-25：基于首帧掩码完成 `walking` 双目标初始化与传播。
  - 新增脚本：`segment_video_with_first_mask.py`，从 `DAVIS/Annotations_unsupervised/480p/walking/00000.png` 读取带实例标签的掩码。
  - 自动拆分首帧掩码，按标签映射为 `obj_id=1/2`，逐个调用 `add_new_mask`（缺失时回退为采样点）。
  - 首帧与传播结果保存到 `output/walking/00000_id{obj}.png`，验证两名行人可被分别追踪。
  - 若掩码存在更多标签，默认截取前两个实例；后续可拓展为参数化序列/对象选择。
