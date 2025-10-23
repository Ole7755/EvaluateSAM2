import torch
from sam2.build_sam import build_sam2_video_predictor
from pathlib import Path
import numpy as np
from PIL import Image

    
check_point = "sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
predictor = build_sam2_video_predictor(model_cfg,check_point)

seq = "bear"
rgb_dir = Path("DAVIS/JPEGImages/480p/bear")
ann_dir = Path("DAVIS/Annotations/480p/bear")

# 转成“POSIX 风格”的字符串路径，也就是用正斜杠 / 分隔的形式
state = predictor.init_state(rgb_dir.as_posix())

first_mask = np.array(Image.open(ann_dir / "00000.png"))

# np.nonzero(a)返回数组 a 中“非零元素”的索引位置。
# 对二维数组，它返回两个一维数组：(行索引数组, 列索引数组)。

# ys 是所有前景像素的行坐标（y 坐标）
# xs 是所有前景像素的列坐标（x 坐标）
ys,xs = np.nonzero(first_mask)

# fg_point 是一个形状为 (1, 2) 的点坐标数组，表示前景提示点
fg_point = np.array([[xs.mean(), ys.mean()]], dtype=np.float32)

bg_x = max(int(xs.min()) - 10, 0)
bg_y = max(int(ys.min()) - 10, 0)
bg_point = np.array([[bg_x, bg_y]], dtype=np.float32)

points = np.concatenate([fg_point,bg_point],axis=0)
labels = np.array([1,0],dtype=np.int32)

"""
frame_idx: 当前即时分割所对应的帧索引（int，通常等于你传入的那个 frame_idx）。
  - object_ids: 本帧里已被追踪/存在结果的对象 ID 列表或张量（按位置与 masks 对齐）。注意这是“对象 ID”，不是张量索引；位置 k 与 masks[k] 一一对应。
  - masks: 当前帧上各对象的分割结果，按 object_ids 的顺序堆叠。常见为 torch.Tensor，形状约为 [K, H, W]（单目标时也可能直接是 [H, W] 或 [1, H, W]），值为 0/1 或概率。保存时需要
    先 .cpu() 再转 numpy，并做阈值/类型转换。

"""

# 在指定帧上为某个对象添加交互式提示（正/负点或框），并立刻返回该帧的分割结果。
frame_idx,object_ids,masks = predictor.add_new_points_or_box(
    state,
    frame_idx = 0,
    obj_id = 1,
    points = points,
    labels = labels,
    clear_old_points = True,
    normalize_coords = False
)
out_dir = Path("outputs/bear")
out_dir.mkdir(parents=True, exist_ok=True)

# 把你已添加的提示（点/框/已有 mask）从某一帧“传播”到整段视频，顺序地为每一帧生成该对象的分割结果
for frame_idx, object_ids,masks in predictor.propagate_in_video(state):

    obj_id = int(object_ids[0]) if hasattr(object_ids, "__len__") else int(object_ids)
    mask_t = masks[0]
    # 转为 uint8 的 0/255 图像再保存
    mask_u8 = (mask_t > 0.5).to(torch.uint8).cpu().numpy() * 255
    save_path = out_dir / f"{frame_idx:05d}.png"
    Image.fromarray(mask_u8).save(save_path)