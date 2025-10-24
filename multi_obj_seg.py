import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

def as_id_list(object_ids):
    if isinstance(object_ids, (list, tuple)):
        return [int(x) for x in object_ids]
    if hasattr(object_ids, "tolist"):
        return [int(x) for x in object_ids.tolist()]
    return [int(object_ids)]

def sample_points_from_mask(mask_bool, n=20, seed=0):
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(ys.size, size=min(n, ys.size), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)

def main():
    cfg, ckpt = "sam2_hiera_s.yaml", "sam2_hiera_small.pt"
    seq = sys.argv[1] if len(sys.argv) > 1 else "walking"

    rgb_dir = Path(f"DAVIS/JPEGImages/480p/{seq}")
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

    predictor = build_sam2_video_predictor(cfg, ckpt)
    state = predictor.init_state(rgb_dir.as_posix())

    # 手动点（像素坐标）
    points_id1 = np.array(
        [[560,235],[600,320],[740,365],[540,165],[820,260],[820,440],[650,100]],
        dtype=np.float32)
    labels_id1 = np.array([1,1,1,1,0,0,0], dtype=np.int64)

    points_id2_fg = np.array(
        [[365,250],[370,320],[355,200]], dtype=np.float32)
    points_id2_bg = np.array(
        [[150,240],[220,440],[260,260]], dtype=np.float32)  # 背景边缘
    labels_id2_base = np.array([1,1,1, 0,0,0], dtype=np.int64)

    # 1) 添加 id1
    _, _, _ = predictor.add_new_points_or_box(
        state, frame_idx=0, obj_id=1,
        points=points_id1, labels=labels_id1,
        clear_old_points=False, normalize_coords=False)

    # 2) 取首帧当前结果，得到 id1 掩码，采样若干点作为 id2 的“强负样本”
    fidx, oids, masks = predictor.add_new_points_or_box(
        state, frame_idx=0, obj_id=1,
        points=np.zeros((0,2), np.float32), labels=np.zeros((0,), np.int64),
        clear_old_points=False, normalize_coords=False)
    ids = as_id_list(oids)
    k1 = ids.index(1) if 1 in ids else 0
    m1 = masks[k1]
    if m1.ndim == 3: m1 = m1[0]
    id1_mask_bool = (m1 > 0.5).cpu().numpy().astype(bool)
    id1_as_neg_for_id2 = sample_points_from_mask(id1_mask_bool, n=32, seed=42)
    labels_id2_neg_from_id1 = np.zeros((id1_as_neg_for_id2.shape[0],), dtype=np.int64)

    # 3) 组合 id2 的正负点并添加
    points_id2 = np.concatenate([points_id2_fg, points_id2_bg, id1_as_neg_for_id2], axis=0)
    labels_id2 = np.concatenate([labels_id2_base, labels_id2_neg_from_id1], axis=0)
    _, _, _ = predictor.add_new_points_or_box(
        state, frame_idx=0, obj_id=2,
        points=points_id2, labels=labels_id2,
        clear_old_points=False, normalize_coords=False)

    # 4) 传播并保存
    out_dir = Path(f"output/{seq}"); out_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ids = as_id_list(object_ids)
        for k, oid in enumerate(ids):
            mask_t = masks[k]
            if mask_t.ndim == 3: mask_t = mask_t[0]
            mask_u8 = (mask_t > 0.5).to(torch.uint8).cpu().numpy() * 255
            Image.fromarray(mask_u8).save(out_dir / f"{frame_idx:05d}_id{oid}.png")

if __name__ == "__main__":
    main()
