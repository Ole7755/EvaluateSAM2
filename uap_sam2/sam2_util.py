import copy
import json
import os
import random
import re
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from time import time
import sys
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor
from tqdm import trange, tqdm

from attack_setting import seed_everything, SamForwarder
from deformer import Deformer
from sam2.build_sam import build_sam2_video_predictor
from tps import sparse_image_warp, avg_batch_sparse_image_warp_by_filter, generate_max_filter

from datasets import Dataset_SA_V, Dataset_DAVIS, Dataset_YOUTUBE, Dataset_segtrack, generate_random_point, Dataset_GOT, \
    Dataset_MOSE
from typing import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
torch.cuda.set_device(1)
Data  = Union[np.ndarray, Tensor]
DATA_ROOT = Path("/HARD-DRIVE/ZZQ/songyufei/data/sav_test/JPEGImages_24fps")
DATA_ROOT1 = Path("/HARD-DRIVE/ZZQ/songyufei/data/DAVIS/JPEGImages/480p")
DATA_ROOT2 = Path("/HARD-DRIVE/ZZQ/songyufei/data/SegTrackv2/JPEGImages")
DATA_ROOT3 = Path("/HARD-DRIVE/ZZQ/songyufei/data/YOUTUBE/train/JPEGImages")
DATA_ROOT4 = Path("/HARD-DRIVE/ZZQ/songyufei/data/got10k/train")
DATA_ROOT5 = Path("/HARD-DRIVE/ZZQ/songyufei/data/MOSE/train/JPEGImages")
def choose_dataset(args = None):
    print(f">> choose dataset from {args.train_dataset}")
    if args.train_dataset == 'SA-V':
        video_dirs = [video_dir for video_dir in DATA_ROOT.iterdir() if video_dir.is_dir()]
        video_dirs.sort(key=lambda x: x.name, reverse=True)
        num_samples = len(video_dirs) if args.limit_img == -1 else min(max(args.limit_img, 0), len(video_dirs))
        video_dirs = random.sample(video_dirs, num_samples)
        video_sample_ids = {}
        start_frames = {}  # 保存每个视频的第一个帧索引
        json_path = Path("/HARD-DRIVE/ZZQ/songyufei/data/sav_test/Annotations_6fps")  # GT 根目录
        for video_dir in video_dirs:
            if video_dir.is_dir():  # 只处理子目录
                frames = [f"{video_dir.name}/{fp.stem}" for fp in video_dir.iterdir() if fp.is_file()]
                frames.sort()
                if frames:  # 处理第一个帧的索引
                    first_frame_name = os.path.basename(frames[0].split('/')[-1])
                    try:
                        start_frame_idx = int(first_frame_name.lstrip('0') or '0')  # 处理前导零情况
                        start_frames[video_dir.name] = start_frame_idx
                    except ValueError as e:
                        print(f"Error processing first frame of {video_dir.name}: {e}")
                        start_frames[video_dir.name] = None  # 或者设置为其他默认值
                if args.limit_frames > 0 and len(frames) >= args.limit_frames:
                    #step = max(1, len(frames) // args.limit_frames)
                    step = 4
                    selected_frames = frames[::step][:args.limit_frames]
                else:
                    print(f"Warning: Video {video_dir.name} has less than {args.limit_frames} frames. All frames will be used.")
                    selected_frames = frames
                selected_frames = selected_frames[::-1]  # 倒序
                # **过滤掉没有对应 Ground Truth 的 sample_id**
                valid_frames = []
                for sample_id in selected_frames:
                    gt_fore_id = sample_id.split("/")[0]  # 例如 "sav_042483"
                    gt_back_id = sample_id.split("/")[-1]  # 例如 "00176"
                    gt_path = json_path / gt_fore_id / "000" / f"{gt_back_id}.png"
                    if gt_path.exists():  # 只有 GT 存在时才保留
                        valid_frames.append(sample_id)
                video_sample_ids[video_dir.name] = valid_frames
        sample_ids = [item for sublist in video_sample_ids.values() for item in sublist]
        custom_dataset = Dataset_SA_V(sample_ids, DATA_ROOT, json_path, args=args, start_frames=start_frames)

    elif args.train_dataset == 'DAVIS':

        video_dirs = [video_dir for video_dir in DATA_ROOT1.iterdir() if video_dir.is_dir()]
        video_dirs.sort(key=lambda x: x.name, reverse=True)
        num_samples = len(video_dirs) if args.limit_img == -1 else min(max(args.limit_img, 0), len(video_dirs))
        video_dirs = random.sample(video_dirs, num_samples)
        video_sample_ids = {}
        start_frames = {}  # 用于保存每个视频的第一个帧索引
        for video_dir in video_dirs:
            if video_dir.is_dir():  # 确保我们只处理子目录
                frames = [f"{video_dir.name}/{fp.stem}" for fp in video_dir.iterdir() if fp.is_file()]
                frames.sort()
                if frames:  # 如果有可用帧
                    first_frame_name = os.path.basename(frames[0].split('/')[-1])
                    try:
                        start_frame_idx = int(first_frame_name.lstrip('0') or '0')  # 处理可能的全零情况
                        start_frames[video_dir.name] = start_frame_idx
                    except ValueError as e:
                        print(f"Error processing first frame of {video_dir.name}: {e}")
                        start_frames[video_dir.name] = None  # 或者设置为其他默认值
                if args.limit_frames > 0 and len(frames) >= args.limit_frames:
                    step = max(1, len(frames) // args.limit_frames)
                    selected_frames = frames[::step][:args.limit_frames]
                else:
                    print(f"Warning: Video {video_dir.name} has less than {args.limit_frames} frames. All frames will be used.")
                    selected_frames = frames
                selected_frames = selected_frames[::-1]  # 倒序
                video_sample_ids[video_dir.name] = selected_frames
        sample_ids = [item for sublist in video_sample_ids.values() for item in sublist]
        json_path = "/HARD-DRIVE/ZZQ/songyufei/data/DAVIS/Annotations/480p"
        custom_dataset = Dataset_DAVIS(sample_ids, DATA_ROOT1, json_path, args=args, start_frames=start_frames)

    elif args.train_dataset == 'YOUTUBE':

        video_dirs = [video_dir for video_dir in DATA_ROOT3.iterdir() if video_dir.is_dir()]
        video_dirs.sort(key=lambda x: x.name, reverse=True)
        num_samples = len(video_dirs) if args.limit_img == -1 else min(max(args.limit_img, 0), len(video_dirs))
        video_dirs = random.sample(video_dirs, num_samples)

        video_sample_ids = {}
        start_frames = {}  # 用于保存每个视频的第一个帧索引
        for video_dir in video_dirs:
            if video_dir.is_dir():  # 确保我们只处理子目录
                frames = [f"{video_dir.name}/{fp.stem}" for fp in video_dir.iterdir() if fp.is_file()]
                frames.sort()
                #frames.reverse()
                if frames:  # 如果有可用帧
                    first_frame_name = os.path.basename(frames[0].split('/')[-1])
                    try:
                        start_frame_idx = int(first_frame_name.lstrip('0') or '0')  # 处理可能的全零情况
                        start_frames[video_dir.name] = start_frame_idx
                    except ValueError as e:
                        print(f"Error processing first frame of {video_dir.name}: {e}")
                        start_frames[video_dir.name] = None  # 或者设置为其他默认值
                if args.limit_frames > 0 and len(frames) >= args.limit_frames:
                    step = max(1, len(frames) // args.limit_frames)
                    selected_frames = frames[::step][:args.limit_frames]
                else:
                    print(f"Warning: Video {video_dir.name} has less than {args.limit_frames} frames. All frames will be used.")
                    selected_frames = frames
                selected_frames = selected_frames[::-1]  # 倒序
                video_sample_ids[video_dir.name] = selected_frames
        sample_ids = [item for sublist in video_sample_ids.values() for item in sublist]
        json_path = "/HARD-DRIVE/ZZQ/songyufei/data/YOUTUBE/train/Annotations"
        custom_dataset = Dataset_YOUTUBE(sample_ids, DATA_ROOT3, json_path, args=args, start_frames=start_frames)

    elif args.train_dataset == 'segtrack':

        video_dirs = [video_dir for video_dir in DATA_ROOT2.iterdir() if video_dir.is_dir()]
        video_dirs.sort(key=lambda x: x.name, reverse=True)
        num_samples = len(video_dirs) if args.limit_img == -1 else min(max(args.limit_img, 0), len(video_dirs))
        video_dirs = random.sample(video_dirs, num_samples)
        # video_dirs = video_dirs[:args.limit_img]
        # random.shuffle(video_dirs)
        video_sample_ids = {}
        start_frames = {}  # 用于保存每个视频的第一个帧索引
        for video_dir in video_dirs:
            if video_dir.is_dir():  # 确保我们只处理子目录
                frames = [f"{video_dir.name}/{fp.stem}" for fp in video_dir.iterdir() if fp.is_file()]
                frames.sort()
                if frames:  # 如果有可用帧
                    first_frame_name = frames[0]
                    start_frame_idx = get_frame_index_from_segtrack(first_frame_name)
                    start_frames[video_dir.name] = start_frame_idx
                if args.limit_frames > 0 and len(frames) >= args.limit_frames:
                    step = max(1, len(frames) // args.limit_frames)
                    selected_frames = frames[::step][:args.limit_frames]
                else:
                    print(f"Warning: Video {video_dir.name} has less than {args.limit_frames} frames. All frames will be used.")
                    selected_frames = frames
                selected_frames = selected_frames[::-1]  # 倒序
                video_sample_ids[video_dir.name] = selected_frames
        sample_ids = [item for sublist in video_sample_ids.values() for item in sublist]
        json_path = "/HARD-DRIVE/ZZQ/songyufei/data/SegTrackv2/GroundTruth"
        custom_dataset = Dataset_segtrack(sample_ids, DATA_ROOT2, json_path, args=args, start_frames=start_frames)

    elif args.train_dataset == 'GOT':
        video_dirs = [video_dir for video_dir in DATA_ROOT4.iterdir() if video_dir.is_dir()]
        video_dirs.sort(key=lambda x: x.name, reverse=True)
        num_samples = len(video_dirs) if args.limit_img == -1 else min(max(args.limit_img, 0), len(video_dirs))
        video_dirs = random.sample(video_dirs, num_samples)

        video_sample_ids = {}
        start_frames = {}  # 用于保存每个视频的第一个帧索引
        for video_dir in video_dirs:
            if video_dir.is_dir():  # 确保我们只处理子目录
                frames = [f"{video_dir.name}/{fp.stem}" for fp in video_dir.iterdir() if fp.is_file()]
                frames.sort()
                # frames.reverse()
                if frames:  # 如果有可用帧
                    first_frame_name = os.path.basename(frames[0].split('/')[-1])
                    try:
                        start_frame_idx = int(first_frame_name.lstrip('0') or '0')  # 处理可能的全零情况
                        start_frames[video_dir.name] = start_frame_idx
                    except ValueError as e:
                        print(f"Error processing first frame of {video_dir.name}: {e}")
                        start_frames[video_dir.name] = None  # 或者设置为其他默认值
                if args.limit_frames > 0 and len(frames) >= args.limit_frames:
                    step = max(1, len(frames) // args.limit_frames)
                    selected_frames = frames[::step][:args.limit_frames]
                else:
                    print(f"Warning: Video {video_dir.name} has less than {args.limit_frames} frames. All frames will be used.")
                    selected_frames = frames
                selected_frames = selected_frames[::-1]  # 倒序
                video_sample_ids[video_dir.name] = selected_frames
        sample_ids = [item for sublist in video_sample_ids.values() for item in sublist]
        #json_path = "/HARD-DRIVE/ZZQ/songyufei/data/YOUTUBE/train/Annotations"
        custom_dataset = Dataset_GOT(sample_ids, DATA_ROOT4,  args=args, start_frames=start_frames)
    elif args.train_dataset == 'MOSE':
        video_dirs = [video_dir for video_dir in DATA_ROOT5.iterdir() if video_dir.is_dir()]
        video_dirs.sort(key=lambda x: x.name, reverse=True)
        num_samples = len(video_dirs) if args.limit_img == -1 else min(max(args.limit_img, 0), len(video_dirs))
        video_dirs = random.sample(video_dirs, num_samples)

        video_sample_ids = {}
        start_frames = {}  # 用于保存每个视频的第一个帧索引
        for video_dir in video_dirs:
            if video_dir.is_dir():  # 确保我们只处理子目录
                frames = [f"{video_dir.name}/{fp.stem}" for fp in video_dir.iterdir() if fp.is_file() and '.' not in fp.stem]
                frames.sort()
                # frames.reverse()
                if frames:  # 如果有可用帧
                    first_frame_name = os.path.basename(frames[0].split('/')[-1])
                    try:
                        start_frame_idx = int(first_frame_name.lstrip('0') or '0')  # 处理可能的全零情况
                        start_frames[video_dir.name] = start_frame_idx
                    except ValueError as e:
                        print(f"Error processing first frame of {video_dir.name}: {e}")
                        start_frames[video_dir.name] = None  # 或者设置为其他默认值
                if args.limit_frames > 0 and len(frames) >= args.limit_frames:
                    step = max(1, len(frames) // args.limit_frames)
                    selected_frames = frames[::step][:args.limit_frames]
                else:
                    print(f"Warning: Video {video_dir.name} has less than {args.limit_frames} frames. All frames will be used.")
                    selected_frames = frames
                selected_frames = selected_frames[::-1]  # 倒序
                video_sample_ids[video_dir.name] = selected_frames
        sample_ids = [item for sublist in video_sample_ids.values() for item in sublist]
        json_path = "/HARD-DRIVE/ZZQ/songyufei/data/MOSE/train/Annotations"
        custom_dataset = Dataset_MOSE(sample_ids, DATA_ROOT5,  json_path,args=args, start_frames=start_frames)

    return custom_dataset
def get_video_to_indices(dataset):
    video_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        img_ID = dataset.get_img_id(idx)
        video_name = img_ID.split('/')[0]
        video_to_indices[video_name].append(idx)
    return video_to_indices
def load_model(args,device = "cuda:1"):
    if args.checkpoints == 'sam2-t':
        checkpoint = "/HARD-DRIVE/ZZQ/songyufei/Advsam2/checkpoints/sam2_hiera_tiny.pt"
        model_cfg = "configs/sam2/sam2_hiera_t.yaml"
    elif args.checkpoints == 'sam2-s':
        checkpoint = "/HARD-DRIVE/ZZQ/songyufei/sam2/sam2/checkpoints/sam2_hiera_small.pt"
        model_cfg = "configs/sam2/sam2_hiera_s.yaml"
    elif args.checkpoints == 'sam2-b+':
        checkpoint = "/HARD-DRIVE/ZZQ/songyufei/sam2/sam2/checkpoints/sam2_hiera_base_plus.pt"
        model_cfg = "configs/sam2/sam2_hiera_b+.yaml"
    elif args.checkpoints == 'sam2.1-t':
        checkpoint = "/HARD-DRIVE/ZZQ/songyufei/Advsam2/checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    else:
        raise ValueError(f"Unsupported checkpoint type: {args.checkpoints}")
    sam2 = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    sam2.eval()
    sam_fwder = SamForwarder(sam2).to(device)
    sam_fwder.eval()
    predictor = sam2
    return sam_fwder, predictor
def get_iou(x: np.ndarray, y: np.ndarray) -> float:
  while x.ndim < y.ndim:
    x = np.expand_dims(x, 0)
  while y.ndim < x.ndim:
    y = np.expand_dims(y, 0)
  if x.shape == y.shape:
    denominator = np.sum(np.logical_or(x, y))
    return np.sum(np.logical_and(x, y)) / denominator if denominator != 0 else 0.0
  else:
    min_shape = tuple(min(a, b) for a, b in zip(x.shape, y.shape))
    x = x[tuple(slice(0, s) for s in min_shape)]
    y = y[tuple(slice(0, s) for s in min_shape)]
    denominator = np.sum(np.logical_or(x, y))
    return np.sum(np.logical_and(x, y)) / denominator if denominator != 0 else 0
def get_iou_auto(x:Union[Data, List[Data]], y:Data) -> float:
  if isinstance(x, list):
    iou = max([get_iou(m, y) for m in x])
  else:
    iou = get_iou(x, y)
  return iou
def collate_fn(batch):
    buffer_list, P_list, sample_id,gt ,point = zip(*batch)
    return buffer_list, P_list, sample_id,gt,point
def show_mask(mask, ax, random_color=False):
    # if random_color:
    #     color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    # else:
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])  # 默认颜色
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) #* color.reshape(1, 1, -1)
    ax.imshow(mask_image,cmap='gray')
    #ax.imshow(mask_image)
def show_points(coords, labels, ax, marker_size=375):
  pos_points = coords[labels == 1]
  neg_points = coords[labels == 0]
  ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
             linewidth=1.25)
  ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
             linewidth=1.25)
def show_box(box, ax):
  x0, y0 = box[0], box[1]
  w, h = box[2] - box[0], box[3] - box[1]
  ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
def get_frame_index(img_ID):
    # 获取文件名部分并去除扩展名（如果有）
    base_name = os.path.splitext(os.path.basename(img_ID))[0]

    try:
        # 假设文件名为5位数字（可以调整以适应实际情况）
        if base_name.isdigit() and len(base_name) == 5:
            return int(base_name)
        else:
            raise ValueError(f"Unexpected frame name format: {base_name}")
    except ValueError as e:
        print(f"Error processing {img_ID}: {e}")
        raise
def get_frame_index_seg(img_ID):
    base_name = os.path.basename(img_ID).split('.')[0]

    try:
        # 匹配文件名中的任意位置的数字部分
        match = re.search(r'\d+', base_name)

        if match:
            frame_index_str = match.group(0)
            return int(frame_index_str)
        else:
            raise ValueError(f"No digits found in the file name: {base_name}")
    except ValueError as e:
        print(f"Error processing {img_ID}: {e}")
        return None
def get_frame_index_from_segtrack(filename):

    match = re.search(r'(\d+)(?:\.|$)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    else:
        print(f"Error processing first frame of {filename}: No numeric part found.")
        return None
def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.3):

    # 确保mask是布尔类型，并转换为uint8类型（0或255）
    mask = mask.astype(bool)

    # 创建一个全黑图像作为覆盖层
    overlay = np.zeros_like(image)
    overlay[mask] = color

    # 将覆盖层叠加到原始图像上
    result = cv2.addWeighted(image, 1, overlay, alpha, 0)

    return result
def save_image_only(image, video_name, frame_idx, save_dir):

    video_save_dir = os.path.join(save_dir, video_name)
    os.makedirs(video_save_dir, exist_ok=True)
    file_prefix = f'{frame_idx:04d}'
    filename = f'{file_prefix}.jpg'
    save_path = os.path.join(video_save_dir, filename)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_to_save = image

    cv2.imwrite(save_path, image_to_save)
    print(f"图像已保存: {save_path}")
def save_attack_mask(mask, video_name, frame_idx, save_dir):

    if mask.ndim > 2:
        mask = mask.squeeze()

    video_save_dir = os.path.join(save_dir, video_name)
    os.makedirs(video_save_dir, exist_ok=True)

    file_prefix = f'{frame_idx:04d}'
    filename = f'{file_prefix}_mask.png'

    save_path = os.path.join(video_save_dir, filename)

    mask_image = (mask * 255).astype(np.uint8)

    cv2.imwrite(save_path, mask_image)
    print(f"Mask 已保存为灰度图像: {save_path}")
# def process_videos_defense(video_root_dir, output_dir, mask_gt_dict, start_P_dict, predictor, category,
#                    video_range=None, skipped_frames=None, args=None):
#     os.makedirs(output_dir, exist_ok=True)
#
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(args.seed)
#         torch.cuda.manual_seed_all(args.seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#
#     total_iou = 0
#     iou_count = 0
#     miou_values = []
#
#     if skipped_frames is None:
#         skipped_frames = set()
#
#     print(f"Processing {category} samples...")
#
#     video_names = sorted(os.listdir(video_root_dir))
#     if video_range is not None:
#         start_idx, end_idx = video_range
#         video_names = video_names[start_idx:end_idx]
#
#     for video_name in video_names:
#         video_dir = os.path.join(video_root_dir, video_name)
#         if not os.path.isdir(video_dir):
#             continue
#
#         frame_names = [
#             p for p in os.listdir(video_dir)
#             if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
#         ]
#         frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
#
#         # inference_state = predictor.init_state(video_path=video_dir)
#         # predictor.reset_state(inference_state)
#         try:
#             inference_state = predictor.init_state(video_path=video_dir)
#             predictor.reset_state(inference_state)
#         except RuntimeError as e:
#             print(f"[SKIP] 跳过 {video_name}，因为视频帧加载失败：{e}")
#             continue
#
#         if video_name in start_P_dict:
#             if args.test_prompts == 'pt':
#                 points = start_P_dict[video_name]
#                 points = np.array(points, dtype=np.float32)
#                 print(f"🎯 读取存储的 {video_name} start_P 作为 points: {points}")
#                 labels = np.array([1], np.int32)
#                 ann_obj_id = 1
#                 ann_frame_idx = 0
#                 _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#                     inference_state=inference_state,
#                     frame_idx=ann_frame_idx,
#                     obj_id=ann_obj_id,
#                     points=points,
#                     labels=labels,
#                 )
#             elif args.test_prompts == 'bx':
#                 box = start_P_dict[video_name]
#                 box = np.array(box, dtype=np.float32)
#                 print(f"🎯 读取存储的 {video_name} start_P 作为 box: {box}")
#                 ann_obj_id = 1
#                 ann_frame_idx = 0
#                 _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#                     inference_state=inference_state,
#                     frame_idx=ann_frame_idx,
#                     obj_id=ann_obj_id,
#                     box = box
#                 )
#
#         else:
#             print(f"⚠️ {video_name} 不在 start_P_dict，跳过该帧处理")
#             continue
#
#         video_segments = {}
#         for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#             video_segments[out_frame_idx] = {
#                 out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#                 for i, out_obj_id in enumerate(out_obj_ids)
#             }
#
#         vis_frame_stride = 1
#         for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
#             fig, ax = plt.subplots(figsize=(6, 4))
#             plt.title(f"frame {out_frame_idx}")
#
#             image_path = os.path.join(video_dir, frame_names[out_frame_idx])
#             image = np.array(Image.open(image_path))
#             ax.imshow(image)
#
#             if (video_name, out_frame_idx) in mask_gt_dict:
#                 mask_gt = mask_gt_dict[(video_name, out_frame_idx)]
#             else:
#                 print(f"Warning: No ground truth mask found for {video_name}, frame {out_frame_idx}")
#                 continue
#
#             frame_iou_values = []
#             for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():
#
#                 save_dir = f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/save_mask/{args.train_dataset}_{args.checkpoints}_mask_clean" if category == "clean" else f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/save_mask/{args.train_dataset}_{args.checkpoints}_mask_adv"
#                 os.makedirs(save_dir, exist_ok=True)
#                 save_attack_mask(out_mask, video_name, out_frame_idx, save_dir)
#                 show_mask(out_mask, ax)
#                 if args.test_prompts =='pt':
#                     show_points(points, labels, ax)
#                 elif args.test_prompts == 'bx':
#                     show_box(box,ax)
#
#                 if not np.any(mask_gt > 0):
#                     print(f"⚠️ {video_name}, frame {out_frame_idx}: mask_gt 全为 0，跳过 IoU 计算")
#                     continue
#
#                 iou_img = get_iou_auto(out_mask, mask_gt)
#
#                 if category == "clean" and iou_img < 0.0:
#                     print(f"🚫 Skipping frame {out_frame_idx} of {video_name} due to low IoU ({iou_img:.4f})")
#                     skipped_frames.add((video_name, out_frame_idx))
#                     continue
#
#                 if category == "adversarial" and (video_name, out_frame_idx) in skipped_frames:
#                     print(f"🚫 Skipping adversarial IoU for {video_name}, frame {out_frame_idx}, as clean IoU was < 0.15")
#                     continue
#
#                 total_iou += iou_img
#                 iou_count += 1
#                 frame_iou_values.append(iou_img)
#                 print(f"IoU for {category} {video_name}, frame {out_frame_idx}, object {out_obj_id}: {iou_img:.4f}")
#
#             if frame_iou_values:
#                 miou_values.append(np.mean(frame_iou_values))
#
#             save_prefix = "clean" if category == "clean" else "adv"
#             save_path = os.path.join(output_dir, f"{video_name}_{save_prefix}_frame_{out_frame_idx:04d}.jpg")
#             plt.savefig(save_path)
#             plt.close(fig)
#
#     avg_iou = total_iou / iou_count if iou_count > 0 else 0
#     miou = np.mean(miou_values) if miou_values else 0
#
#     if category == "clean":
#         print(f"Average IoU for clean samples: {avg_iou:.4f}")
#     else:
#         print(f"Average IoU for adversarial samples: {avg_iou:.4f}")
#
#     return miou, iou_count, skipped_frames
# def process_videos_defense(video_root_dir, output_dir, mask_gt_dict, start_P_dict, predictor, category,video_range=None, skipped_frames=None, args=None):
#
#     seed_everything(seed = args.seed)
#     total_iou = 0
#     iou_count = 0
#     miou_values = []
#
#     if skipped_frames is None:
#         skipped_frames = set()
#
#     print(f"Processing {category} samples...")
#
#     video_names = sorted(os.listdir(video_root_dir))
#     if video_range is not None:
#         start_idx, end_idx = video_range
#         video_names = video_names[start_idx:end_idx]
#
#     for video_name in video_names:
#         video_dir = os.path.join(video_root_dir, video_name)
#         if not os.path.isdir(video_dir):
#             continue
#
#         frame_names = [
#             p for p in os.listdir(video_dir)
#             if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
#         ]
#         frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
#
#         inference_state = predictor.init_state(video_path=video_dir)
#         predictor.reset_state(inference_state)
#
#         if video_name in start_P_dict:
#             if args.test_prompts == 'pt':
#                 points = start_P_dict[video_name]
#                 points = np.array(points, dtype=np.float32)
#                 print(f"🎯 读取存储的 {video_name} start_P 作为 points: {points}")
#                 labels = np.array([1], np.int32)
#                 ann_obj_id = 1
#                 ann_frame_idx = 0
#                 _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#                     inference_state=inference_state,
#                     frame_idx=ann_frame_idx,
#                     obj_id=ann_obj_id,
#                     points=points,
#                     labels=labels,
#                 )
#             elif args.test_prompts == 'bx':
#                 box = start_P_dict[video_name]
#                 box = np.array(box, dtype=np.float32)
#                 print(f"🎯 读取存储的 {video_name} start_P 作为 box: {box}")
#                 ann_obj_id = 1
#                 ann_frame_idx = 0
#                 _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#                     inference_state=inference_state,
#                     frame_idx=ann_frame_idx,
#                     obj_id=ann_obj_id,
#                     box = box
#                 )
#
#         else:
#             print(f"⚠️ {video_name} 不在 start_P_dict，跳过该帧处理")
#             continue
#
#         video_segments = {}
#         for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#             video_segments[out_frame_idx] = {
#                 out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#                 for i, out_obj_id in enumerate(out_obj_ids)
#             }
#
#         vis_frame_stride = 1
#         for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
#             fig, ax = plt.subplots(figsize=(6, 4))
#             plt.title(f"frame {out_frame_idx}")
#
#             image_path = os.path.join(video_dir, frame_names[out_frame_idx])
#             image = np.array(Image.open(image_path))
#             ax.imshow(image)
#
#             if (video_name, out_frame_idx) in mask_gt_dict:
#                 mask_gt = mask_gt_dict[(video_name, out_frame_idx)]
#             else:
#                 print(f"Warning: No ground truth mask found for {video_name}, frame {out_frame_idx}")
#                 continue
#
#             frame_iou_values = []
#             for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():
#
#                 if args.save_mask:
#                     save_dir = f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/save_mask/{args.train_dataset}_{args.checkpoints}_{args.attack}_mask_clean" if category == "clean" else f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/save_mask/{args.train_dataset}_{args.checkpoints}_{args.attack}_mask_adv"
#                     os.makedirs(save_dir, exist_ok=True)
#                     save_attack_mask(out_mask, video_name, out_frame_idx, save_dir)
#                     show_mask(out_mask, ax)
#                     if args.test_prompts =='pt':
#                         show_points(points, labels, ax)
#                     elif args.test_prompts == 'bx':
#                         show_box(box,ax)
#
#                 if not np.any(mask_gt > 0):
#                     print(f"⚠️ {video_name}, frame {out_frame_idx}: mask_gt 全为 0，跳过 IoU 计算")
#                     continue
#
#                 iou_img = get_iou_auto(out_mask, mask_gt)
#
#                 if category == "clean" and iou_img < 0.0:
#                     print(f"🚫 Skipping frame {out_frame_idx} of {video_name} due to low IoU ({iou_img:.4f})")
#                     skipped_frames.add((video_name, out_frame_idx))
#                     continue
#
#                 if category == "adversarial" and (video_name, out_frame_idx) in skipped_frames:
#                     print(f"🚫 Skipping adversarial IoU for {video_name}, frame {out_frame_idx}, as clean IoU was < 0.15")
#                     continue
#
#                 total_iou += iou_img
#                 iou_count += 1
#                 frame_iou_values.append(iou_img)
#                 print(f"IoU for {category} {video_name}, frame {out_frame_idx}, object {out_obj_id}: {iou_img:.4f}")
#
#             if frame_iou_values:
#                 miou_values.append(np.mean(frame_iou_values))
#
#             if args.save_img_with_mask:
#                 os.makedirs(output_dir, exist_ok=True)
#                 save_prefix = "clean" if category == "clean" else "adv"
#                 save_path = os.path.join(output_dir, f"{video_name}_{save_prefix}_frame_{out_frame_idx:04d}.jpg")
#                 plt.savefig(save_path)
#                 plt.close(fig)
#     avg_iou = total_iou / iou_count if iou_count > 0 else 0
#     miou = np.mean(miou_values) if miou_values else 0
#
#     if category == "clean":
#         print(f"Average IoU for clean samples: {avg_iou:.4f}")
#     else:
#         print(f"Average IoU for adversarial samples: {avg_iou:.4f}")
#
#     return miou, iou_count, skipped_frames
def process_videos_defense(video_root_dir, output_dir, mask_gt_dict, start_P_dict, predictor, category,
                   video_range=None, skipped_frames=None, args=None):
    os.makedirs(output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    total_iou = 0
    iou_count = 0
    miou_values = []

    if skipped_frames is None:
        skipped_frames = set()

    print(f"Processing {category} samples...")

    video_names = sorted(os.listdir(video_root_dir))
    if video_range is not None:
        start_idx, end_idx = video_range
        video_names = video_names[start_idx:end_idx]

    for video_name in video_names:
        video_dir = os.path.join(video_root_dir, video_name)
        if not os.path.isdir(video_dir):
            continue

        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = predictor.init_state(video_path=video_dir)
        predictor.reset_state(inference_state)

        if video_name in start_P_dict:
            if args.test_prompts == 'pt':
                points = start_P_dict[video_name]
                points = np.array(points, dtype=np.float32)
                print(f"🎯 读取存储的 {video_name} start_P 作为 points: {points}")
                labels = np.array([1], np.int32)
                ann_obj_id = 1
                ann_frame_idx = 0
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )
            elif args.test_prompts == 'bx':
                box = start_P_dict[video_name]
                box = np.array(box, dtype=np.float32)
                print(f"🎯 读取存储的 {video_name} start_P 作为 box: {box}")
                ann_obj_id = 1
                ann_frame_idx = 0
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    box = box
                )


        else:
            print(f"⚠️ {video_name} 不在 start_P_dict，跳过该帧处理")
            continue

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        vis_frame_stride = 1
        for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")

            image_path = os.path.join(video_dir, frame_names[out_frame_idx])
            image = np.array(Image.open(image_path))
            ax.imshow(image)

            if (video_name, out_frame_idx) in mask_gt_dict:
                mask_gt = mask_gt_dict[(video_name, out_frame_idx)]
            else:
                print(f"Warning: No ground truth mask found for {video_name}, frame {out_frame_idx}")
                continue

            frame_iou_values = []
            for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():
                # save_dir = f"/HARD-DRIVE/ZZQ/songyufei/sam2/save_sam2/{args.train_dataset}_{args.checkpoints}_mask_clean" if category == "clean" else f"/HARD-DRIVE/ZZQ/songyufei/sam2/save_sam2/{args.train_dataset}_{args.checkpoints}_mask_adv"
                save_dir = f"/HARD-DRIVE/ZZQ/songyufei/sam2/save_sam2/{args.train_dataset}_{args.checkpoints}_mask_clean" if category == "clean" else f"/HARD-DRIVE/ZZQ/songyufei/sam2/save_sam2/{args.train_dataset}_{args.checkpoints}_mask_adv"
                os.makedirs(save_dir, exist_ok=True)
                #save_attack_mask(out_mask, video_name, out_frame_idx, save_dir, attack=args.attack)
                show_mask(out_mask, ax)
                if args.test_prompts =='pt':
                    show_points(points, labels, ax)
                elif args.test_prompts == 'bx':
                    show_box(box,ax)

                if not np.any(mask_gt > 0):
                    print(f"⚠️ {video_name}, frame {out_frame_idx}: mask_gt 全为 0，跳过 IoU 计算")
                    continue

                iou_img = get_iou_auto(out_mask, mask_gt)

                if category == "clean" and iou_img < 0.0:
                    print(f"🚫 Skipping frame {out_frame_idx} of {video_name} due to low IoU ({iou_img:.4f})")
                    skipped_frames.add((video_name, out_frame_idx))
                    continue

                if category == "adversarial" and (video_name, out_frame_idx) in skipped_frames:
                    print(f"🚫 Skipping adversarial IoU for {video_name}, frame {out_frame_idx}, as clean IoU was < 0.15")
                    #skipped_frames.add((video_name, out_frame_idx))
                    continue

                total_iou += iou_img
                iou_count += 1
                frame_iou_values.append(iou_img)
                print(f"IoU for {category} {video_name}, frame {out_frame_idx}, object {out_obj_id}: {iou_img:.4f}")

            if frame_iou_values:
                miou_values.append(np.mean(frame_iou_values))

            save_prefix = "clean" if category == "clean" else "adv"
            save_path = os.path.join(output_dir, f"{video_name}_{save_prefix}_frame_{out_frame_idx:04d}.jpg")
            plt.savefig(save_path)
            plt.close(fig)

    avg_iou = total_iou / iou_count if iou_count > 0 else 0
    miou = np.mean(miou_values) if miou_values else 0

    if category == "clean":
        print(f"Average IoU for clean samples: {avg_iou:.4f}")
    else:
        print(f"Average IoU for adversarial samples: {avg_iou:.4f}")

    return miou, iou_count, skipped_frames
def process_videos(video_root_dir, output_dir, mask_gt_dict, start_P_dict, predictor, category,video_range=None, skipped_frames=None, args=None):

    seed_everything(seed = args.seed)
    total_iou = 0
    iou_count = 0
    miou_values = []

    if skipped_frames is None:
        skipped_frames = set()

    print(f"Processing {category} samples...")

    video_names = sorted(os.listdir(video_root_dir))
    if video_range is not None:
        start_idx, end_idx = video_range
        video_names = video_names[start_idx:end_idx]

    for video_name in video_names:
        video_dir = os.path.join(video_root_dir, video_name)
        if not os.path.isdir(video_dir):
            continue

        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = predictor.init_state(video_path=video_dir)
        predictor.reset_state(inference_state)

        if video_name in start_P_dict:
            if args.test_prompts == 'pt':
                points = start_P_dict[video_name]
                points = np.array(points, dtype=np.float32)
                print(f"🎯 读取存储的 {video_name} start_P 作为 points: {points}")
                labels = np.array([1], np.int32)
                ann_obj_id = 1
                ann_frame_idx = 0
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )
            elif args.test_prompts == 'bx':
                box = start_P_dict[video_name]
                box = np.array(box, dtype=np.float32)
                print(f"🎯 读取存储的 {video_name} start_P 作为 box: {box}")
                ann_obj_id = 1
                ann_frame_idx = 0
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    box = box
                )

        else:
            print(f"⚠️ {video_name} 不在 start_P_dict，跳过该帧处理")
            continue

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        vis_frame_stride = 1
        for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")

            image_path = os.path.join(video_dir, frame_names[out_frame_idx])
            image = np.array(Image.open(image_path))
            ax.imshow(image)

            if (video_name, out_frame_idx) in mask_gt_dict:
                mask_gt = mask_gt_dict[(video_name, out_frame_idx)]
            else:
                print(f"Warning: No ground truth mask found for {video_name}, frame {out_frame_idx}")
                continue

            frame_iou_values = []
            for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():

                if args.save_mask:
                    save_dir = f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/save_mask/{args.train_dataset}_{args.checkpoints}_{args.attack}_mask_clean" if category == "clean" else f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/save_mask/{args.train_dataset}_{args.checkpoints}_{args.attack}_mask_adv"
                    os.makedirs(save_dir, exist_ok=True)
                    save_attack_mask(out_mask, video_name, out_frame_idx, save_dir)
                    show_mask(out_mask, ax)
                    if args.test_prompts =='pt':
                        show_points(points, labels, ax)
                    elif args.test_prompts == 'bx':
                        show_box(box,ax)

                if not np.any(mask_gt > 0):
                    print(f"⚠️ {video_name}, frame {out_frame_idx}: mask_gt 全为 0，跳过 IoU 计算")
                    continue

                iou_img = get_iou_auto(out_mask, mask_gt)

                if category == "clean" and iou_img < 0.3:
                    print(f"🚫 Skipping frame {out_frame_idx} of {video_name} due to low IoU ({iou_img:.4f})")
                    skipped_frames.add((video_name, out_frame_idx))
                    continue

                if category == "adversarial" and (video_name, out_frame_idx) in skipped_frames:
                    print(f"🚫 Skipping adversarial IoU for {video_name}, frame {out_frame_idx}, as clean IoU was < 0.15")
                    continue

                total_iou += iou_img
                iou_count += 1
                frame_iou_values.append(iou_img)
                print(f"IoU for {category} {video_name}, frame {out_frame_idx}, object {out_obj_id}: {iou_img:.4f}")

            if frame_iou_values:
                miou_values.append(np.mean(frame_iou_values))

            if args.save_img_with_mask:
                os.makedirs(output_dir, exist_ok=True)
                save_prefix = "clean" if category == "clean" else "adv"
                save_path = os.path.join(output_dir, f"{video_name}_{save_prefix}_frame_{out_frame_idx:04d}.jpg")
                plt.savefig(save_path)
                plt.close(fig)
    avg_iou = total_iou / iou_count if iou_count > 0 else 0
    miou = np.mean(miou_values) if miou_values else 0

    if category == "clean":
        print(f"Average IoU for clean samples: {avg_iou:.4f}")
    else:
        print(f"Average IoU for adversarial samples: {avg_iou:.4f}")

    return miou, iou_count, skipped_frames
def process_videos11(video_root_dir, output_dir, mask_gt_dict, start_P_dict, predictor, category,video_range=None, skipped_frames=None, args=None):

    seed_everything(seed = args.seed)
    total_iou = 0
    iou_count = 0
    miou_values = []

    if skipped_frames is None:
        skipped_frames = set()

    print(f"Processing {category} samples...")

    video_names = sorted(os.listdir(video_root_dir))
    if video_range is not None:
        start_idx, end_idx = video_range
        video_names = video_names[start_idx:end_idx]

    for video_name in video_names:
        video_dir = os.path.join(video_root_dir, video_name)
        if not os.path.isdir(video_dir):
            continue

        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = predictor.init_state(video_path=video_dir)
        predictor.reset_state(inference_state)

        if video_name in start_P_dict:
            if args.test_prompts == 'pt':
                points = start_P_dict[video_name]
                points = np.array(points, dtype=np.float32)
                print(f"🎯 读取存储的 {video_name} start_P 作为 points: {points}")
                labels = np.array([1], np.int32)
                ann_obj_id = 1
                ann_frame_idx = 0
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )
            elif args.test_prompts == 'bx':
                box = start_P_dict[video_name]
                box = np.array(box, dtype=np.float32)
                print(f"🎯 读取存储的 {video_name} start_P 作为 box: {box}")
                ann_obj_id = 1
                ann_frame_idx = 0
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    box = box
                )

        else:
            print(f"⚠️ {video_name} 不在 start_P_dict，跳过该帧处理")
            continue

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        vis_frame_stride = 1
        for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")

            image_path = os.path.join(video_dir, frame_names[out_frame_idx])
            image = np.array(Image.open(image_path))
            ax.imshow(image)

            if (video_name, out_frame_idx) in mask_gt_dict:
                mask_gt = mask_gt_dict[(video_name, out_frame_idx)]
            else:
                print(f"Warning: No ground truth mask found for {video_name}, frame {out_frame_idx}")
                continue

            frame_iou_values = []
            for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():

                if args.save_mask:
                    save_dir = f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/save_mask/{args.train_dataset}_{args.checkpoints}_{args.attack}_mask_clean" if category == "clean" else f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/save_mask/{args.train_dataset}_{args.checkpoints}_{args.attack}_mask_adv"
                    os.makedirs(save_dir, exist_ok=True)
                    save_attack_mask(out_mask, video_name, out_frame_idx, save_dir)
                    show_mask(out_mask, ax)
                    if args.test_prompts =='pt':
                        show_points(points, labels, ax)
                    elif args.test_prompts == 'bx':
                        show_box(box,ax)

                if not np.any(mask_gt > 0):
                    print(f"⚠️ {video_name}, frame {out_frame_idx}: mask_gt 全为 0，跳过 IoU 计算")
                    continue

                iou_img = get_iou_auto(out_mask, mask_gt)

                if category == "clean" and iou_img < 0.0:
                    print(f"🚫 Skipping frame {out_frame_idx} of {video_name} due to low IoU ({iou_img:.4f})")
                    skipped_frames.add((video_name, out_frame_idx))
                    continue

                if category == "adversarial" and (video_name, out_frame_idx) in skipped_frames:
                    print(f"🚫 Skipping adversarial IoU for {video_name}, frame {out_frame_idx}, as clean IoU was < 0.15")
                    continue

                total_iou += iou_img
                iou_count += 1
                frame_iou_values.append(iou_img)
                print(f"IoU for {category} {video_name}, frame {out_frame_idx}, object {out_obj_id}: {iou_img:.4f}")

            if frame_iou_values:
                miou_values.append(np.mean(frame_iou_values))

            if args.save_img_with_mask:
                os.makedirs(output_dir, exist_ok=True)
                save_prefix = "clean" if category == "clean" else "adv"
                save_path = os.path.join(output_dir, f"{video_name}_{save_prefix}_frame_{out_frame_idx:04d}.jpg")
                plt.savefig(save_path)
                plt.close(fig)
    avg_iou = total_iou / iou_count if iou_count > 0 else 0
    miou = np.mean(miou_values) if miou_values else 0

    if category == "clean":
        print(f"Average IoU for clean samples: {avg_iou:.4f}")
    else:
        print(f"Average IoU for adversarial samples: {avg_iou:.4f}")

    return miou, iou_count, skipped_frames
def process_videos_test(video_root_dir, output_dir, mask_gt_dict, start_P_dict, predictor, category,
                   video_range=None, skipped_frames=None, args=None):
    #seed_everything(seed=args.seed)
    total_iou = 0
    iou_count = 0

    if skipped_frames is None:
        skipped_frames = set()

    print(f"Processing {category} samples...")

    video_names = sorted(os.listdir(video_root_dir))
    if video_range is not None:
        start_idx, end_idx = video_range
        video_names = video_names[start_idx:end_idx]

    for video_name in video_names:
        video_dir = os.path.join(video_root_dir, video_name)
        if not os.path.isdir(video_dir):
            continue

        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = predictor.init_state(video_path=video_dir)
        predictor.reset_state(inference_state)

        if video_name in start_P_dict:
            if args.test_prompts == 'pt':
                points = start_P_dict[video_name]
                points = np.array(points, dtype=np.float32)
                print(f"🎯 读取存储的 {video_name} start_P 作为 points: {points}")
                labels = np.array([1], np.int32)
                ann_obj_id = 1
                ann_frame_idx = 0
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )
            elif args.test_prompts == 'bx':
                box = start_P_dict[video_name]
                box = np.array(box, dtype=np.float32)
                print(f"🎯 读取存储的 {video_name} start_P 作为 box: {box}")
                ann_obj_id = 1
                ann_frame_idx = 0
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    box=box
                )

        else:
            print(f"⚠️ {video_name} 不在 start_P_dict，跳过该帧处理")
            continue

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        vis_frame_stride = 1
        for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")

            image_path = os.path.join(video_dir, frame_names[out_frame_idx])
            image = np.array(Image.open(image_path))
            ax.imshow(image)

            if (video_name, out_frame_idx) in mask_gt_dict:
                mask_gt = mask_gt_dict[(video_name, out_frame_idx)]
            else:
                print(f"Warning: No ground truth mask found for {video_name}, frame {out_frame_idx}")
                continue

            for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():

                if args.save_mask:
                    save_dir = f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/save_mask/{args.train_dataset}_{args.checkpoints}_mask_clean" if category == "clean" else f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/save_mask/{args.train_dataset}_{args.checkpoints}_mask_adv"
                    os.makedirs(save_dir, exist_ok=True)
                    save_attack_mask(out_mask, video_name, out_frame_idx, save_dir)
                    show_mask(out_mask, ax)
                    if args.test_prompts == 'pt':
                        show_points(points, labels, ax)
                    elif args.test_prompts == 'bx':
                        show_box(box, ax)

                if not np.any(mask_gt > 0):
                    print(f"⚠️ {video_name}, frame {out_frame_idx}: mask_gt 全为 0，跳过 IoU 计算")
                    continue

                iou_img = get_iou_auto(out_mask, mask_gt)

                if category == "clean" and iou_img < 0.3: ####0.3
                    print(f"🚫 Skipping frame {out_frame_idx} of {video_name} due to low IoU ({iou_img:.4f})")
                    skipped_frames.add((video_name, out_frame_idx))
                    continue

                if category == "adversarial" and (video_name, out_frame_idx) in skipped_frames:
                    print(f"🚫 Skipping adversarial IoU for {video_name}, frame {out_frame_idx}, as clean IoU was < 0.15")
                    continue

                total_iou += iou_img
                iou_count += 1
                print(f"IoU for {category} {video_name}, frame {out_frame_idx}, object {out_obj_id}: {iou_img:.4f}")
                print(f"Current iou_count: {iou_count}")

            if args.save_img_with_mask:
                os.makedirs(output_dir, exist_ok=True)
                save_prefix = "clean" if category == "clean" else "adv"
                save_path = os.path.join(output_dir, f"{video_name}_{save_prefix}_frame_{out_frame_idx:04d}.jpg")
                plt.savefig(save_path)
                plt.close(fig)

    avg_iou = total_iou / iou_count if iou_count > 0 else 0

    return avg_iou, iou_count, skipped_frames

def make_print_to_file(path='./'):
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8')  # 使用追加模式打开文件
            self.log_path = os.path.join(path, filename)

        def write(self, message):
            if not isinstance(message, str):
                message = str(message)
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()  # 确保每次写入后都刷新缓冲区

        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def close(self):
            self.log.close()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()

    fileName = datetime.now().strftime('%Y_%m_%d') + '.log'
    logger = Logger(fileName, path=path)
    sys.stdout = logger
    return logger.log_path  # 返回完整的日志文件路径


    # def UAD(self,
    #         clean_cv2_image,
    #         sam_fwder,
    #         denorm,
    #         control_loss_alpha=0.05,
    #         fidelity_loss_alpha=1,
    #         deform_epochs=40,
    #         est_fidelity_iter=4,
    #         warp_size=4,
    #         warp_filter_stride=300,
    #         num_split=6,
    #         src_rand_range=0.0,
    #         dst_rand_range=0.1,
    #         ):
    #     # Hyperparameters
    #     adv_alpha = 1 / 255.0
    #     adv_iters = 10
    #     pixel_mean = [0.485, 0.456, 0.406]
    #     pixel_std = [0.229, 0.224, 0.225]
    #     adv_epsilon = 8 / 255.0
    #
    #     # Initialize Deformer
    #     deformer = Deformer(clean_cv2_image=clean_cv2_image,
    #                         num_split=num_split,
    #                         fix_corner=False,
    #                         src_rand_range=src_rand_range,
    #                         dst_rand_range=dst_rand_range)
    #     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #
    #     # Prepare control points
    #     base_locs = np.array(deformer.source_centers)
    #     fix_idx = 4 if deformer.fix_corner else 0
    #     control_point_size = base_locs.shape[0] - fix_idx
    #     source_locs = np.repeat(base_locs[np.newaxis, :], warp_size, axis=0)
    #     source_locs[:, fix_idx:, 0] = np.minimum(np.maximum(source_locs[:, fix_idx:, 0] + np.random.randn(warp_size,
    #                                                                                                       control_point_size) * deformer.block_h * deformer.src_rand_range,
    #                                                         0), 1023)
    #     source_locs[:, fix_idx:, 1] = np.minimum(np.maximum(source_locs[:, fix_idx:, 1] + np.random.randn(warp_size,
    #                                                                                                       control_point_size) * deformer.block_w * deformer.src_rand_range,
    #                                                         0), 1023)
    #     dest_locs = copy.copy(source_locs)
    #     dest_locs[:, fix_idx:, 0] = np.minimum(np.maximum(source_locs[:, fix_idx:, 0] + np.random.randn(warp_size,
    #                                                                                                     control_point_size) * deformer.block_h * deformer.dst_rand_range,
    #                                                       0), 1023)
    #     dest_locs[:, fix_idx:, 1] = np.minimum(np.maximum(source_locs[:, fix_idx:, 1] + np.random.randn(warp_size,
    #                                                                                                     control_point_size) * deformer.block_w * deformer.dst_rand_range,
    #                                                       0), 1023)
    #
    #     # Convert to tensors
    #     source_locs = torch.Tensor(source_locs).to(device)
    #     dest_locs = torch.Tensor(dest_locs).to(device)
    #
    #     # Optimizer for deformation
    #     deform_optimizer = torch.optim.Adam([source_locs, dest_locs], lr=10)
    #
    #     # Prepare input image
    #     input_image = sam_fwder.transform_image(clean_cv2_image)
    #     input_image = denorm(input_image)
    #     original_input = input_image.detach().clone()
    #     original_features = sam_fwder.get_image_feature(input_image)
    #
    #     L = 256
    #     deform_filter = generate_max_filter(original_input.permute(0, 2, 3, 1), source_locs, dest_locs,
    #                                         filter_stride=warp_filter_stride)
    #     self.cosfn = torch.nn.CosineSimilarity(dim=-1)
    #
    #     # Deformation loop
    #     with trange(deform_epochs, desc='UAD') as pbar:
    #         for deform_epoch in pbar:
    #             deform_optimizer.zero_grad()
    #             source_locs.requires_grad_(True)
    #             dest_locs.requires_grad_(True)
    #
    #             if warp_size == 1:
    #                 warped_input, dense_flows = sparse_image_warp(
    #                     original_input.permute(0, 2, 3, 1),
    #                     source_locs, dest_locs,
    #                     interpolation_order=1,
    #                     regularization_weight=0.0,
    #                     num_boundaries_points=0
    #                 )
    #             elif warp_size > 1:
    #                 warped_input, dense_flows = avg_batch_sparse_image_warp_by_filter(
    #                     original_input.permute(0, 2, 3, 1),
    #                     source_locs, dest_locs,
    #                     interpolation_order=1,
    #                     regularization_weight=0.0,
    #                     num_boundaries_points=0,
    #                     max_filter=deform_filter
    #                 )
    #
    #             deform_loss = deformer.deform_loss(original_input, warped_input.permute(0, 3, 1, 2))
    #             control_loss = deformer.control_loss(source_locs, dest_locs)
    #             deformation_loss = deform_loss + control_loss_alpha * control_loss
    #
    #             deformation_loss.backward(retain_graph=True)  # Ensure retain_graph=True
    #             deform_optimizer.step()
    #
    #             with torch.no_grad():
    #                 source_locs.data = torch.clamp(source_locs.data, 0, 1023)
    #                 dest_locs.data = torch.clamp(dest_locs.data, 0, 1023)
    #
    #             warped_features = sam_fwder.get_image_feature(warped_input.permute(0, 3, 1, 2))
    #
    #             # Adversarial perturbation loop
    #             adv_input = original_input.data
    #             adv_iters = adv_iters if deform_epoch == deform_epochs - 1 else est_fidelity_iter
    #             for adv_iter in range(adv_iters):
    #                 deform_optimizer.zero_grad()
    #                 adv_input.requires_grad_(True)
    #
    #                 fidelity_loss = self.compute_fidelity_loss(
    #                     adv_input, sam_fwder, source_locs, dest_locs, original_features, warped_features,
    #                     fidelity_loss_alpha
    #                 )
    #                 fidelity_loss.backward(retain_graph=True)  # Ensure retain_graph=True
    #
    #                 perturbation = adv_alpha * adv_input.grad.data.sign()
    #                 perturbation = torch.clamp(perturbation, -adv_epsilon, adv_epsilon)
    #
    #             # Update adv_input with perturbation
    #             adv_input = adv_input + perturbation
    #             adv_input = torch.clamp(adv_input, 0, 1)
    #
    #         self.warped_input = warped_input.permute(0, 3, 1, 2)
    #         return perturbation
    #
    # def compute_fidelity_loss(self, adv_input, sam_fwder,source_locs, dest_locs, original_features, warped_features,
    #                           fidelity_loss_alpha):
    #     adv_features = sam_fwder.get_image_feature(adv_input)
    #     L = 256
    #     original_features_flatten = original_features.reshape(L, -1)
    #     adv_features_flatten = adv_features.reshape(L, -1)
    #     warped_features_flatten = warped_features.reshape(L, -1)
    #
    #     fidelity_loss = (- self.cosfn(adv_features_flatten, warped_features_flatten).mean() +
    #                      self.cosfn(adv_features_flatten, original_features_flatten).mean()) * fidelity_loss_alpha
    #     return fidelity_loss
def UAD(clean_cv2_image,
        device,
        sam_fwder,
        benign_img,
        perturbation,
        control_loss_alpha=0.05, fidelity_loss_alpha=1, deform_epochs=40, est_fidelity_iter=4,
        warp_size=4, warp_filter_stride=300, num_split=6, src_rand_range=0.0, dst_rand_range=0.1,
        use_DI=False, DI_noise_std=4.0 / 255, use_MI=False, MI_momentum=1):
    adv_iters = 10
    alpha = 1/255
    eps = 8/255
    deformer = Deformer(clean_cv2_image=clean_cv2_image,
                        num_split=num_split,
                        fix_corner=False,
                        src_rand_range=src_rand_range,
                        dst_rand_range=dst_rand_range)

    base_locs = np.array(deformer.source_centers)
    fix_idx = 4 if deformer.fix_corner else 0
    control_point_size = base_locs.shape[0] - fix_idx
    source_locs = np.repeat(base_locs[np.newaxis, :], warp_size, axis=0)
    source_locs[:, fix_idx:, 0] = np.minimum(np.maximum(source_locs[:, fix_idx:, 0] + np.random.randn(warp_size,
                                                                                                      control_point_size) * deformer.block_h * deformer.src_rand_range,
                                                        0), 1023)
    source_locs[:, fix_idx:, 1] = np.minimum(np.maximum(source_locs[:, fix_idx:, 1] + np.random.randn(warp_size,
                                                                                                      control_point_size) * deformer.block_w * deformer.src_rand_range,
                                                        0), 1023)
    dest_locs = copy.copy(source_locs)
    dest_locs[:, fix_idx:, 0] = np.minimum(np.maximum(source_locs[:, fix_idx:, 0] + np.random.randn(warp_size,
                                                                                                    control_point_size) * deformer.block_h * deformer.dst_rand_range,
                                                      0), 1023)
    dest_locs[:, fix_idx:, 1] = np.minimum(np.maximum(source_locs[:, fix_idx:, 1] + np.random.randn(warp_size,
                                                                                                    control_point_size) * deformer.block_w * deformer.dst_rand_range,
                                                      0), 1023)

    # Convert to tensors
    source_locs = torch.Tensor(source_locs).to(device)
    dest_locs = torch.Tensor(dest_locs).to(device)

    # Optimizer for deformation
    deform_optimizer = torch.optim.Adam([source_locs, dest_locs], lr=10)
    original_features = sam_fwder.get_image_feature(benign_img)
    L = 256

    deform_filter = generate_max_filter(benign_img.permute(0, 2, 3, 1), source_locs, dest_locs, filter_stride=warp_filter_stride)
    cosfn = torch.nn.CosineSimilarity(dim=-1)
    with trange(deform_epochs, desc='UAD') as pbar:
        for deform_epoch in pbar:
            deform_optimizer.zero_grad()
            source_locs.requires_grad_(True)
            dest_locs.requires_grad_(True)
            if warp_size == 1:
                warped_input, dense_flows = sparse_image_warp(
                    benign_img.permute(0, 2, 3, 1),
                    source_locs, dest_locs,
                    interpolation_order=1,
                    regularization_weight=0.0,
                    num_boundaries_points=0
                )
            elif warp_size > 1:
                warped_input, dense_flows = avg_batch_sparse_image_warp_by_filter(
                    benign_img.permute(0, 2, 3, 1),
                    source_locs, dest_locs,
                    interpolation_order=1,
                    regularization_weight=0.0,
                    num_boundaries_points=0,
                    max_filter=deform_filter
                )

            warped_in = warped_input.permute(0, 3, 1, 2)

            # warped_input_np = warped_in.squeeze().permute(1, 2, 0).cpu().numpy()
            # warped_input_np = (warped_input_np - warped_input_np.min()) / (warped_input_np.max() - warped_input_np.min())
            # plt.imsave(f'warped_input_epoch_{deform_epoch}.png', warped_input_np)


            deform_loss = deformer.deform_loss(benign_img, warped_input.permute(0, 3, 1, 2))
            control_loss = deformer.control_loss(source_locs, dest_locs)
            deformation_loss = deform_loss + control_loss_alpha * control_loss
            deformation_loss.backward()
            deform_optimizer.step()
            with torch.no_grad():
                source_locs.data = torch.clamp(source_locs.data, 0, 1023)
                dest_locs.data = torch.clamp(dest_locs.data, 0, 1023)

            warped_features = sam_fwder.get_image_feature(warped_input.permute(0, 3, 1, 2))

            adv_img = benign_img + perturbation

            adv_iters = adv_iters if deform_epoch == deform_epochs - 1 else est_fidelity_iter
            accumulated_grad = torch.zeros_like(adv_img)
            for adv_iter in range(adv_iters):
                if adv_iter == 0:
                    deform_optimizer.zero_grad()

                adv_img.requires_grad_(True)

                adv_features = sam_fwder.get_image_feature(adv_img)

                original_features_flatten = original_features.reshape(L, -1)
                adv_features_flatten = adv_features.reshape(L, -1)
                warped_features_flatten = warped_features.reshape(L, -1)

                fidelity_loss = (- cosfn(adv_features_flatten, warped_features_flatten).mean()
                                 + cosfn(adv_features_flatten, original_features_flatten).mean()) * fidelity_loss_alpha

                retain_graph = adv_iter < adv_iters - 1
                fidelity_loss.backward(inputs=[adv_img, source_locs, dest_locs], retain_graph=retain_graph)
                print(f'fidelity_loss: {fidelity_loss.item()}')

                g = adv_img.grad.data
                delta = g.sign() * alpha
                perturbation = perturbation - delta
                perturbation = torch.clamp(perturbation, -eps, eps)
                perturbation = perturbation.detach()

                adv_img = benign_img + perturbation
                adv_img = torch.clamp(adv_img, 0, 1)
    # final_perturbation = perturbation.cpu().squeeze().permute(1, 2, 0).numpy()
    # # 归一化扰动
    # normalized_perturbation = (final_perturbation - final_perturbation.min()) / (final_perturbation.max() - final_perturbation.min())
    # # 保存扰动图像
    # plt.imsave('final_perturbation.png', normalized_perturbation)
    # print("11111111111111111111")
    # exit()
    return adv_img






