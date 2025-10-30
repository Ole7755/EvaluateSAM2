import csv
import json
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
import sys
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sam2_util import get_frame_index, get_frame_index_seg, get_frame_index_from_segtrack, save_image_only, \
    process_videos, choose_dataset, collate_fn, get_video_to_indices, load_model, process_videos_test
from attack_setting import SamForwarder, make_prompts, seed_everything
def make_print_to_file(path='./'):
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "w", encoding='utf8', )
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            pass
    fileName = datetime.now().strftime('%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
    return fileName

# def run(args, custom_dataset):
#     device = "cuda:1"
#     sam_fwder, predictor = load_model(args, device=device)
#
#     video_to_indices = get_video_to_indices(custom_dataset)
#     total_miou_clean = 0.0
#     total_miou_adv = 0.0
#     video_count = 0
#     printed_videos = set()
#     denorm = lambda x: sam_fwder.denorm_image(x)
#
#     total_frame_count = 0
#     mask_gt_dict_all = {}
#     start_P_dict_all = {}
#     video_result_paths = {}
#     video_result_cleans = {}
#
#     for video_name, indices in video_to_indices.items():
#         print(f"\n{'=' * 40} Processing video: {video_name} {'=' * 40}")
#
#         uap_id = f"{args.train_dataset}_{args.train_prompts}_{args.seed}_{args.attack}_uap"
#         #uap_id = f"{args.train_dataset}_{args.train_prompts}_10_uap_214"
#         uap_path = f"uap_file_all/{uap_id}.pth"
#         if not os.path.exists(uap_path):
#             print(f"⚠️ UAP file not found for {video_name}, skipping...")
#             continue
#         uap = torch.load(uap_path, map_location=device)
#
#         video_subset = Subset(custom_dataset, indices)
#         video_loader = DataLoader(video_subset, batch_size=1, collate_fn=collate_fn, shuffle=False)
#         video_result_path = f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/alluap1/adv/{args.train_dataset}_{args.checkpoints}_{args.seed}_{args.attack}/{video_name}"
#         video_result_clean = f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/alluap1/clean/{args.train_dataset}_{args.checkpoints}_{args.seed}_{args.attack}/{video_name}"
#
#         mask_gt_dict = {}
#         frame_counter = {}
#         start_P_dict = {}
#         for batch in tqdm(video_loader):
#             total_frame_count += 1
#             images, P_list, sample_ids, mask_gt_list, point = batch
#             image = images[0]
#             mask_gt = mask_gt_list[0]
#             img_ID = sample_ids[0]
#             video_name = img_ID.split('/')[0]
#
#             if args.test_dataset == 'segtrack':
#                 frame_idx = get_frame_index_seg(img_ID)
#             else:
#                 frame_idx = get_frame_index(img_ID)
#             print(f"Processing frame {frame_idx} from video {video_name}")
#
#             if video_name not in printed_videos:
#                 start_frame_idx = custom_dataset.get_start_frame_idx(video_name)
#                 if start_frame_idx is not None:
#                     print(f"Start frame index for video {video_name}: {start_frame_idx}")
#                 else:
#                     print(f"Could not find start frame index for video {video_name}")
#                 printed_videos.add(video_name)
#
#             if video_name not in frame_counter:
#                 frame_counter[video_name] = 0
#
#             indexed_frame_idx = frame_counter[video_name]
#             if frame_idx == start_frame_idx:
#                 start_point = point[0]
#                 print("Load new video and initialize memory........................")
#
#             if video_name not in start_P_dict:
#                 start_P_dict[video_name] = start_point
#                 print(f"🎯 存储 {video_name} 的 start_P")
#
#             X = sam_fwder.transform_image(image).to(device)
#             benign_img = denorm(X).to(device)
#             adv_img = torch.clamp(benign_img + uap, 0, 1)
#
#             adv_image = adv_img.clone().detach().squeeze().permute(1, 2, 0).cpu().numpy()
#             adv_image = (adv_image * 255).astype('uint8')
#             save_image_only(adv_image, video_name, frame_idx, video_result_path)
#
#             im = benign_img.clone().detach().squeeze().permute(1, 2, 0).cpu().numpy()
#             im = (im * 255).astype('uint8')
#             save_image_only(im, video_name, frame_idx, video_result_clean)
#
#             frame_counter[video_name] += 1
#             mask_gt_dict[(video_name, indexed_frame_idx)] = mask_gt
#
#         mask_gt_dict_all.update(mask_gt_dict)
#         start_P_dict_all.update(start_P_dict)
#         video_result_paths[video_name] = video_result_path
#         video_result_cleans[video_name] = video_result_clean
#         video_count += 1
#
#     iou_count = 0
#     iou_count_adv = 0
#     skipped_frames = []
#
#
#     for video_name in video_result_paths.keys():
#
#         #print("video_result_paths.keys()")
#
#         video_output_adv = f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/alluap1/save_image_adv/{args.train_dataset}_{args.checkpoints}_{args.seed}_{args.attack}/{video_name}"
#         video_output_clean = f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/alluap1/save_image_clean/{args.train_dataset}_{args.checkpoints}_{args.seed}_{args.attack}/{video_name}"
#         print(f"video_output_adv: {video_output_adv}")
#
#         miou_clean, iou_count_video, skipped_frames_video = process_videos(
#             video_result_cleans[video_name],
#             video_output_clean,
#             mask_gt_dict_all,
#             start_P_dict_all,
#             predictor,
#             category="clean",
#             args=args)
#         miou_adv, iou_count_adv_video, _ = process_videos(
#             video_result_paths[video_name],
#             video_output_adv,
#             mask_gt_dict_all,
#             start_P_dict_all,
#             predictor,
#             category="adversarial",
#             skipped_frames=skipped_frames_video,
#             args=args)
#
#         total_miou_clean += miou_clean
#         total_miou_adv += miou_adv
#         iou_count += iou_count_video
#         iou_count_adv += iou_count_adv_video
#
#     avg_miou_clean = total_miou_clean / video_count if video_count > 0 else 0
#     avg_miou_adv = total_miou_adv / video_count if video_count > 0 else 0
#
#     return video_count, avg_miou_clean, avg_miou_adv, iou_count, iou_count_adv, total_frame_count
def run(args, custom_dataset):
    device = "cuda:1"
    sam_fwder, predictor = load_model(args, device=device)

    video_to_indices = get_video_to_indices(custom_dataset)
    total_miou_clean = 0.0
    total_miou_adv = 0.0
    iou_count = 0
    iou_count_adv = 0
    video_count = 0
    printed_videos = set()
    denorm = lambda x: sam_fwder.denorm_image(x)

    total_frame_count = 0
    mask_gt_dict_all = {}
    start_P_dict_all = {}
    video_result_paths = {}
    video_result_cleans = {}

    for video_name, indices in video_to_indices.items():
        print(f"\n{'=' * 40} Processing video: {video_name} {'=' * 40}")

        uap_id = f"{args.train_dataset}_{args.train_prompts}_{args.seed}_{args.attack}_uap"
        uap_path = f"uap_file_all/{uap_id}.pth"
        if not os.path.exists(uap_path):
            print(f"⚠️ UAP file not found for {video_name}, skipping...")
            continue
        uap = torch.load(uap_path, map_location=device)

        video_subset = Subset(custom_dataset, indices)
        video_loader = DataLoader(video_subset, batch_size=1, collate_fn=collate_fn, shuffle=False)
        video_result_path = f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/alluap1/adv/{args.train_dataset}_{args.checkpoints}_{args.seed}_{args.attack}/{video_name}"
        video_result_clean = f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/alluap1/clean/{args.train_dataset}_{args.checkpoints}_{args.seed}_{args.attack}/{video_name}"

        mask_gt_dict = {}
        frame_counter = {}
        start_P_dict = {}
        for batch in tqdm(video_loader):
            total_frame_count += 1
            images, P_list, sample_ids, mask_gt_list, point = batch
            image = images[0]
            mask_gt = mask_gt_list[0]
            img_ID = sample_ids[0]
            video_name = img_ID.split('/')[0]

            if args.test_dataset == 'segtrack':
                frame_idx = get_frame_index_seg(img_ID)
            else:
                frame_idx = get_frame_index(img_ID)
            print(f"Processing frame {frame_idx} from video {video_name}")

            if video_name not in printed_videos:
                start_frame_idx = custom_dataset.get_start_frame_idx(video_name)
                if start_frame_idx is not None:
                    print(f"Start frame index for video {video_name}: {start_frame_idx}")
                else:
                    print(f"Could not find start frame index for video {video_name}")
                printed_videos.add(video_name)

            if video_name not in frame_counter:
                frame_counter[video_name] = 0

            indexed_frame_idx = frame_counter[video_name]
            if frame_idx == start_frame_idx:
                start_point = point[0]
                print("Load new video and initialize memory........................")

            if video_name not in start_P_dict:
                start_P_dict[video_name] = start_point
                print(f"🎯 存储 {video_name} 的 start_P")

            X = sam_fwder.transform_image(image).to(device)
            benign_img = denorm(X).to(device)
            adv_img = torch.clamp(benign_img + uap, 0, 1)

            adv_image = adv_img.clone().detach().squeeze().permute(1, 2, 0).cpu().numpy()
            adv_image = (adv_image * 255).astype('uint8')
            save_image_only(adv_image, video_name, frame_idx, video_result_path)

            im = benign_img.clone().detach().squeeze().permute(1, 2, 0).cpu().numpy()
            im = (im * 255).astype('uint8')
            save_image_only(im, video_name, frame_idx, video_result_clean)

            frame_counter[video_name] += 1
            mask_gt_dict[(video_name, indexed_frame_idx)] = mask_gt

        mask_gt_dict_all.update(mask_gt_dict)
        start_P_dict_all.update(start_P_dict)
        video_result_paths[video_name] = video_result_path
        video_result_cleans[video_name] = video_result_clean
        video_count += 1

    skipped_frames = []

    for video_name in video_result_paths.keys():
        video_output_adv = f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/alluap1/save_image_adv/{args.train_dataset}_{args.checkpoints}_{args.seed}_{args.attack}/{video_name}"
        video_output_clean = f"/HARD-DRIVE/ZZQ/songyufei/Advsam2/alluap1/save_image_clean/{args.train_dataset}_{args.checkpoints}_{args.seed}_{args.attack}/{video_name}"
        print(f"video_output_adv: {video_output_adv}")

        miou_clean, iou_count_video, skipped_frames_video = process_videos_test(
            video_result_cleans[video_name],
            video_output_clean,
            mask_gt_dict_all,
            start_P_dict_all,
            predictor,
            category="clean",
            args=args)

        miou_adv, iou_count_adv_video, _ = process_videos_test(
            video_result_paths[video_name],
            video_output_adv,
            mask_gt_dict_all,
            start_P_dict_all,
            predictor,
            category="adversarial",
            skipped_frames=skipped_frames_video,
            args=args)

        # ✅ 修改：按帧数量加权累加
        total_miou_clean += miou_clean * iou_count_video
        total_miou_adv += miou_adv * iou_count_adv_video
        iou_count += iou_count_video
        iou_count_adv += iou_count_adv_video

    # ✅ 修改：帧级平均
    avg_miou_clean = total_miou_clean / iou_count if iou_count > 0 else 0
    avg_miou_adv = total_miou_adv / iou_count_adv if iou_count_adv > 0 else 0
    print(f'iou_count: {iou_count}')

    return video_count, avg_miou_clean, avg_miou_adv, iou_count, iou_count_adv, total_frame_count
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Your script description here")

    parser.add_argument('--limit_img', default=100, type=int, help='limit run image count, set -1 for all')
    parser.add_argument('--limit_frames', default=15, type=int, help='limit run image count, set -1 for all')
    parser.add_argument('--train_dataset', default='YOUTUBE')
    parser.add_argument('--test_dataset', default='YOUTUBE')
    parser.add_argument('--point', help='point coord formatted as h,w; e.g. 0.3,0.4 or 200,300')
    parser.add_argument('--train_prompts', choices=['bx', 'pt'], default='pt', help='type of prompts (box or point)')
    parser.add_argument('--test_prompts', choices=['bx', 'pt'], default='pt', help='type of prompts (box or point)')
    parser.add_argument('--checkpoints', default='sam2-t', help='model checkpoint')

    parser.add_argument('--seed', default=30, type=int, help='rand seed')
    parser.add_argument('--eps', default=10, type=float)
    parser.add_argument('--P_num', default=10, type=int)

    parser.add_argument('--save', default='True', type=bool, help='save the csv')
    parser.add_argument('--save_img_with_mask', action='store_true', help='save the image with mask')
    parser.add_argument('--save_mask', action='store_true', help='save the mask')

    parser.add_argument('--attack', default='darksam')

    return parser

def get_args(parser: ArgumentParser) -> Namespace:
    args = parser.parse_args()
    args.fps = -1
    args.debug = False
    return args
if __name__ == '__main__':

    parser = get_parser()
    args = get_args(parser)
    seed_everything(args.seed)
    log_save_path = os.path.join('result', 'test', 'log')
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
    now_time = make_print_to_file(path=log_save_path)
    with open(log_save_path + '/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    custom_dataset = choose_dataset(args)
    video_test, miouimg, miouadv,frames_clean_test,frames_adv_test,frames_train= run(args, custom_dataset)
    print(f":: miouimg: {miouimg * 100:.2f} %, miouadv: {miouadv * 100:.2f} %, video_test: {video_test} , frame_clean_test:{frames_clean_test} , frames_adv_test:{frames_adv_test},frame_train:{frames_train}")

    if args.save:
        final_log_save_path = os.path.join('result', 'duibi_new')
        if not os.path.exists(final_log_save_path):
            os.makedirs(final_log_save_path)
        final_result = [{"seed": args.seed,
                         "now_time": now_time,
                         "final_log_save_path": final_log_save_path,
                         "checkpoints":args.checkpoints,
                         "train_dataset": args.train_dataset,
                         "test_dataset": args.test_dataset,
                         "train_prompt": args.train_prompts,
                         "test_prompt": args.test_prompts,
                         "P_num":args.P_num,
                         "eps": args.eps,
                         "video_test": video_test,
                         "frames_clean_test": frames_clean_test,
                         "frames_adv_test": frames_adv_test,
                         "frames_train": frames_train,
                         "attack":args.attack,
                         "miouimg": f"{miouimg * 100:.2f} %",
                         "miouadv": f"{miouadv * 100:.2f} %"}]

        with open(os.path.join(final_log_save_path, 'final_results.csv'), 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=final_result[0].keys())
            writer.writeheader()
            writer.writerows(final_result)
