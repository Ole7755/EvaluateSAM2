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
ys,xs = np.nonzero(first_mask)
fg_point = np.array([[xs.mean(), ys.mean()]], dtype=np.float32)
bg_x = max(int(xs.min()) - 10, 0)
bg_y = max(int(ys.min()) - 10, 0)
bg_point = np.array([[bg_x, bg_y]], dtype=np.float32)

points = np.concatenate([fg_point,bg_point],axis=0)
labels = np.array([1,0],dtype=np.int32)

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


for frame_idx, object_ids,masks in predictor.propagate_in_video(state):

    obj_id = int(object_ids[0]) if hasattr(object_ids, "__len__") else int(object_ids)
    mask_t = masks[0]

    if mask_t.ndim == 3 and mask_t.shape[0] == 1:
          mask_t = mask_t[0]
    # 转为 uint8 的 0/255 图像再保存
    mask_u8 = (mask_t > 0.5).to(torch.uint8).cpu().numpy() * 255
    save_path = out_dir / f"{frame_idx:05d}.png"
    Image.fromarray(mask_u8).save(save_path)