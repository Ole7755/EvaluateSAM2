import copy
import os
import random
import re
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from imagecorruptions import corrupt
from torch.autograd import grad
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm, trange

from deformer import Deformer
from sam2_util_cuda0 import get_frame_index, get_frame_index_seg, get_frame_index_from_segtrack, collate_fn, \
    choose_dataset, get_video_to_indices, load_model, show_mask, UAD
from attack_setting_cuda0_copy import SamForwarder, make_prompts, make_multi_prompts,seed_everything
import torch.nn.functional as F
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

SAV_ROOT = Path("/HARD-DATA2/SYF/HYF/darksam/data/sav_test/JPEGImages_24fps")
DAVIS_ROOT = Path("/HARD-DATA2/SYF/HYF/darksam/data/DAVIS/JPEGImages/480p")
SEGTRACK_ROOT = Path("/HARD-DATA2/SYF/HYF/darksam/data/SegTrackv2/JPEGImages")
YOUTUBE_ROOT = Path("/HARD-DATA2/SYF/HYF/darksam/data/YOUTUBE/train/JPEGImages")

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Your script description here")
    parser.add_argument('--limit_img', default=1, type=int, help='limit run image count, set -1 for all')
    parser.add_argument('--limit_frames', default=15, type=int, help='limit run image count, set -1 for all')
    parser.add_argument('--fea_num', default=20, type=int)
    parser.add_argument('--train_dataset', default='YOUTUBE')
    parser.add_argument('--test_dataset', default='YOUTUBE')
    parser.add_argument('--point', help='point coord formatted as h,w; e.g. 0.3,0.4 or 200,300')
    parser.add_argument('--train_prompts', choices=['bx', 'pt'], default='pt', help='type of prompts (box or point)')
    parser.add_argument('--checkpoints', default='sam2-s', help='model checkpoint')
    parser.add_argument('--target_image_dir', type=str, default=str(SAV_ROOT))

    parser.add_argument('--seed', default=30, type=int, help='rand seed')
    parser.add_argument('--eps', default = 10, type=int)
    parser.add_argument('--alpha', default= 2 / 255, type=float)
    parser.add_argument('--P_num', default=10, type=int)
    parser.add_argument('--prompts_num', default=256, type=int)
    parser.add_argument('--weight_fea', default=0.000001, type=float)
    parser.add_argument('--beta', default=0.95, type=float)

    parser.add_argument('--ema', action="store_true", help='use ema')
    parser.add_argument('--random_point', action="store_true", help='use random point')
    parser.add_argument("--ma", action="store_true", help="Enable momentum accumulation for gradient updates")
    parser.add_argument('--ags', action="store_true", help='use ags')
    parser.add_argument('--rop', action="store_true", help='use rop')
    parser.add_argument('--gr', action="store_true", help='use gr')


    # parser.add_argument('--loss_mem', action='store_true')
    # parser.add_argument('--loss_fea', action='store_true')
    # parser.add_argument('--loss_diff', action='store_true')
    # parser.add_argument('--loss_t', action='store_true')
    parser.add_argument('--loss_fea', default = True)
    parser.add_argument('--loss_diff', default = True)
    parser.add_argument('--loss_t',default = True)

    parser.add_argument('--loss_mem_ll', choices=['mse', 'nce','cos'],default='cos', type=str)
    parser.add_argument('--loss_diff_ll', choices=['mse', 'nce','cos'],default='cos', type=str)


    return parser
def get_args(parser: ArgumentParser) -> Namespace:
    args = parser.parse_args()
    args.fps = -1
    args.debug = False
    return args
def dice_loss(pred, target, smooth=1e-6):

    # Flatten the tensors to simplify computation
    pred = pred.view(-1)
    target = target.view(-1)

    # 计算交集
    intersection = (pred * target).sum()

    # 计算 Dice 系数
    dice_coeff = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    # Dice Loss 为 1 - Dice 系数
    return 1 - dice_coeff
def infonce_loss(adv_feature, original_feature, target_feature, temperature=0.1):

    # 展平特征
    adv_feature_flat = adv_feature.reshape(1, -1)
    original_feature_flat = original_feature.reshape(1, -1)
    target_feature_flat = target_feature.reshape(1, -1)
    similarity_adv_original = torch.matmul(adv_feature_flat, original_feature_flat.T) / temperature
    similarity_adv_target = torch.matmul(adv_feature_flat, target_feature_flat.T) / temperature

    # 构建损失函数
    # 最大化 adv_feature 和 target_feature 的相似度，最小化 adv_feature 和 original_feature 的相似度
    loss = -similarity_adv_target + similarity_adv_original

    return loss.squeeze()
def dg(image, args):

    corrupted_image = corrupt(image, corruption_name=args.cor_type, severity=args.severity)

    # 将处理后的图像转换为张量
    transform = transforms.ToTensor()
    corrupted_image_tensor = transform(corrupted_image)
    corrupted_image_tensor = corrupted_image_tensor.unsqueeze(0)  # 添加批量维度

    return corrupted_image_tensor

# def get_fused_prototype(img_list, sam_fwder, device):
#     features = []
#     for img in img_list:
#         with torch.no_grad():
#             feat = sam_fwder.get_image_feature(img.to(device)).cpu()  # 立刻转到CPU
#         features.append(feat)
#         del img; torch.cuda.empty_cache()
#     return torch.mean(torch.cat(features, dim=0), dim=0, keepdim=True)
def get_fused_prototype(img_list, sam_fwder, device):
    features = []
    for img in img_list:
        #with torch.no_grad():
        feat = sam_fwder.get_image_feature(img.to(device))#.cpu()  # 立刻转到CPU
        features.append(feat)
        #del img; torch.cuda.empty_cache()
    return torch.mean(torch.cat(features, dim=0), dim=0, keepdim=True)
# def get_fused_prototype(img_list, sam_fwder, device):
#     features = []
#     for img in img_list:
#         with torch.no_grad():
#             feat = sam_fwder.get_image_feature(img.to(device))
#         features.append(feat)
#         del img; torch.cuda.empty_cache()
#     return torch.mean(torch.cat(features, dim=0), dim=0, keepdim=True)
def safe_item(x):
    return x.item() if isinstance(x, torch.Tensor) else float(x)
if __name__ == '__main__':
    parser = get_parser()
    args = get_args(parser)
    seed_everything(seed=args.seed)

    device = "cuda:0"

    sam_fwder, predictor = load_model(args, device=device)

    custom_dataset = choose_dataset(args)
    video_to_indices = get_video_to_indices(custom_dataset)
    data_loader = DataLoader(custom_dataset, batch_size=1, collate_fn=collate_fn, num_workers=0, shuffle=False)

    denorm = lambda x: sam_fwder.denorm_image(x)
    weight_Y = -1

    loss_fn = F.mse_loss
    mse_loss = torch.nn.MSELoss()
    cosine_loss = F.cosine_similarity
    cosfn = torch.nn.CosineSimilarity(dim=-1)

    tensor_shape = (1, 3, 1024, 1024)
    shape_tensor = torch.empty(tensor_shape)

    eps = args.eps / 255

    loss_mem = 0
    feature_diff = 0
    loss_fea = 0
    loss_t = 0
    loss_ft = 0

    weight_loss_fea = 0
    weight_loss_mem = 0
    weight_loss_diff = 0
    weight_loss_t = 0
    weight_loss_ft = 0

    # 初始化全局扰动
    perturbation = torch.empty_like(shape_tensor).uniform_(-eps, eps).to(device)

    sample_step_count = 0
    sample_total_g = torch.zeros_like(perturbation, device=device)

    prev_adv_feature = None

    if args.train_dataset == 'SA-V':
        target_image_dir = "/HARD-DATA2/SYF/HYF/darksam/data/DAVIS/JPEGImages/480p"
    else :
        target_image_dir = '/HARD-DATA2/SYF/HYF/darksam/data/sav_test/JPEGImages_24fps'#args.target_image_dir    #YOUTUBE数据集
    folders = [f for f in os.listdir(target_image_dir) if os.path.isdir(os.path.join(target_image_dir, f))]

    if len(folders) >= args.fea_num:
        selected_folders = random.sample(folders, args.fea_num)
    else:
        selected_folders = folders

    loss_t_list = []
    loss_fea_list = []
    loss_diff_list = []
    video_losses = []
    for step in range(args.P_num):

        for video_name, indices in video_to_indices.items():
            print(f"\n{'=' * 40} Processing video: {video_name} {'=' * 40}")

            folder = random.choice(selected_folders)
            folder_path = os.path.join(target_image_dir, folder)
            image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                image_path = random.choice(image_files)
                image = Image.open(image_path).convert("RGB")
                image = image.resize((1024, 1024), Image.Resampling.BICUBIC)
                image = np.array(image)
                tgt = sam_fwder.transform_image(image).to(device)
                tgt = denorm(tgt)
                target_feature = sam_fwder.get_image_feature(tgt)
            else:
                print("No images found in the selected folder.")
                continue

            # 创建当前视频的数据子集
            video_subset = Subset(custom_dataset, indices)
            video_loader = DataLoader(video_subset, batch_size=1, collate_fn=collate_fn, shuffle=False, num_workers=0)

            pre_dict = None
            pre_dict_adv = None
            mask_pre = None
            mask_pre_adv = None
            start_frame_processed = False

            for images, P_list, img_ids, gt, point in tqdm(video_loader):
                img_ID, img, mask_gt, P_gt = img_ids[0], images[0], gt[0], P_list[0]
                video_name = img_ID.split('/')[0]
                frame_idx = get_frame_index(img_ID) if args.train_dataset != 'segtrack' else get_frame_index_seg(img_ID)

                X = sam_fwder.transform_image(img).to(device)
                benign_img = denorm(X)
                H, W, _ = img.shape
                Y = torch.ones([1, 1, H, W]).to(X.device, torch.float32) * weight_Y
                Y_bin = Y.bool()
                assert Y_bin.dtype in ['bool', bool, torch.bool]
                print(f"args.train_dataset: {args.train_dataset} ")

                # dg_image = dg(img, args)
                #
                # smoothed_img = total_variation_minimization(image_array, num_iterations=args.num_iterations,learning_rate=args.learning_rate)
                #
                #
                # smoothed_img = torch.tensor(smoothed_img, dtype=torch.float32).to(device)
                #
                # reduced_image = reduce_bit_depth(image_array, target_bit_depth=args.target_bit_depth)
                #
                # noise_std = 2 / 255.0  # 扰动预算为 2/255
                # noise = np.random.normal(loc=0, scale=noise_std, size=image_array.shape)
                # noisy_image = image_array + noise  # 直接将噪声添加到图像中
                # # 将像素值限制在 [0, 255] 范围内
                # noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

                transform1 = transforms.RandomRotation(degrees=15)
                transform2 = transforms.Lambda(lambda img: img + 0.03 * torch.rand_like(img))
                transform3 = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08)

                aug_img1 = transform1(benign_img)
                aug_img2 = transform2(benign_img)
                aug_img3 = transform3(benign_img)

                # aug_img1_vis = aug_img1.squeeze(0).permute(1, 2, 0).cpu().numpy()
                # # aug_img1_vis = aug_img1_vis.clip(0, 1)
                # save_path = "./augmented_image1.png"  # 你可以改成你想要的保存路径
                #
                # # 显示图像并保存
                # plt.figure(figsize=(6, 6))
                # plt.imshow(aug_img1_vis)
                # plt.title('Augmented Image 1 (RandomRotation)')
                # plt.axis('off')
                #
                # # 保存成文件（注意：savefig要在show之前或之后调用都行，但要在figure存在的时候）
                # plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)  # 可以加bbox_inches='tight'去掉多余白边
                #
                # plt.show()
                #
                # print(f"Augmented image saved at {save_path}")
                # exit()

                img_list = [aug_img1, aug_img2, aug_img3]
                prototype_feature = get_fused_prototype(img_list, sam_fwder, device)
                prototype_feature = prototype_feature.to(device)

                if not start_frame_processed:
                    start_frame_idx = frame_idx
                    pre_dict = None
                    pre_dict_adv = None
                    mask_pre = None
                    mask_pre_adv = None
                    start_frame_processed = True
                    start_P = P_gt

                    # masks = mask_generator.generate(img)  # len(18)
                    # bboxes = [mask['bbox'] for mask in masks]
                    # # mk = [mask['segmentation'] for mask in masks]
                    # all_prompt_points = []
                    # for bbox in bboxes:
                    #     x, y, width, height = bbox
                    #     prompt_point_x = random.randint(x, x + width)
                    #     prompt_point_y = random.randint(y, y + height)
                    #     prompt_point = [prompt_point_x, prompt_point_y]
                    #     all_prompt_points.append(prompt_point)
                    #
                    # coords = np.array(all_prompt_points, dtype=np.int32)
                    # labels = np.ones((coords.shape[0],), dtype=np.int32)
                    # prompts = (coords, labels, None, None)
                    # print(f"Initializing video {video_name} at frame {start_frame_idx}")
                    #
                    # del all_prompt_points

                prompts = make_multi_prompts(args.point, (1024, 1024), args.prompts_num)
                P = sam_fwder.transform_prompts(*prompts)

                logits_clean = sam_fwder.forward(benign_img, *P)
                mask_clean = logits_clean > sam_fwder.mask_threshold

                mask_pre = mask_clean.clone().detach()
                output_dict = sam_fwder.get_current_out(frame_idx, benign_img, mask_pre)
                pre_dict = output_dict

                #original_feature = sam_fwder.get_image_feature(benign_img)
                #mem_feature = sam_fwder.get_pix_feat(benign_img, frame_idx, pre_dict)[0]

                adv_img = benign_img + perturbation
                adv_img = torch.clamp(adv_img, 0, 1)
                adv_img.requires_grad = True

                # aug_img1_adv = transform1(adv_img)
                # aug_img2_adv = transform2(adv_img)
                # aug_img3_adv = transform3(adv_img)
                #
                # img_list_adv = [aug_img1, aug_img2, aug_img3]
                #
                # prototype_feature_adv = get_fused_prototype(img_list_adv, sam_fwder, device)
                # prototype_feature_adv = prototype_feature_adv.to(device)

                logits = sam_fwder.forward(adv_img, *P)
                mask = logits > sam_fwder.mask_threshold

                # adv_im = adv_img.detach().squeeze().permute(1, 2, 0).cpu().numpy()
                # out_mask = mask.cpu().detach().numpy()
                # fig, ax = plt.subplots(figsize=(6, 4))
                # ax.imshow(adv_im)
                # show_mask(out_mask, ax)
                #
                # if isinstance(prompts[0], np.ndarray) and len(prompts[0].shape) == 2:
                #     points = prompts[0]  # 获取提示点坐标数组
                #     x_coords, y_coords = points[:, 0], points[:, 1]  # 分别提取 x 和 y 坐标
                #
                #     # 绘制提示点
                #     ax.scatter(x_coords, y_coords, color='red', s=10, label='Prompts')
                #
                # save_path = 'output_image.png'  # 可以修改为你想要的文件名和路径
                # plt.savefig(save_path)
                # plt.show()
                #exit()

                mask_pre_adv = mask.clone().detach()
                output_dict_adv = sam_fwder.get_current_out(frame_idx, adv_img, mask_pre_adv)
                pre_dict_adv = output_dict_adv

                adv_feature = sam_fwder.get_image_feature(adv_img)


                if args.loss_t:
                    attacked = mask == Y_bin
                    output = attacked * logits
                    output_f = ~attacked * (1 - logits)
                    loss_t = F.binary_cross_entropy_with_logits(output, Y)
                    print(f'loss_t: {loss_t.item()}')
                    loss_ft = -F.binary_cross_entropy_with_logits(output_f, Y)
                    print(f'loss_f: {loss_ft.item()}')
                    weight_loss_t = 1

                # if args.loss_mem:
                #     mem_feature_adv = sam_fwder.get_pix_feat(adv_img, frame_idx, pre_dict_adv)[0]
                #
                #     if args.loss_mem_ll =='cos':
                #         loss_mem = -cosine_loss(mem_feature, mem_feature_adv).mean()
                #         weight_loss_mem = 1
                #     elif args.loss_mem_ll =='nce':
                #         loss_mem = infonce_loss(mem_feature, mem_feature_adv,target_feature)
                #         weight_loss_mem = 0.0000001
                #     elif args.loss_mem_ll =='mse':
                #         loss_mem = -loss_fn(mem_feature, mem_feature_adv)
                #         weight_loss_mem = 1
                #     print("loss_mem: ", loss_mem)

                if args.loss_diff:
                    if prev_adv_feature is not None:

                        if args.loss_diff_ll == 'cos':
                            feature_diff = -cosine_loss(prev_adv_feature, adv_feature).mean()
                            weight_loss_diff = 1
                        elif args.loss_diff_ll == 'nce':
                            feature_diff = infonce_loss(prev_adv_feature, adv_feature, target_feature)
                            weight_loss_diff = 0.000001
                        elif args.loss_diff_ll == 'mse':
                            feature_diff = -loss_fn(prev_adv_feature, adv_feature)
                            weight_loss_diff = 1
                        print(f"当前帧与上一帧特征的差异: {feature_diff}")

                    else:
                        feature_diff = 0

                if args.loss_fea:
                    loss_fea = infonce_loss(adv_feature, prototype_feature, target_feature)
                    #loss_fea = -cosine_loss(adv_feature, prototype_feature).mean()
                    print(f'loss_fea: {loss_fea}')
                    #先用余弦损失，再用对比学习损失


                # loss_fea_abs = abs(loss_fea)
                # if loss_fea_abs >= 100000:
                #     args.weight_fea = 0.000001
                # elif loss_fea_abs >= 10000:
                #     args.weight_fea = 0.00001
                # elif loss_fea_abs >= 1000:
                #     args.weight_fea = 0.0001
                # elif loss_fea_abs >= 100:
                #     args.weight_fea = 0.001
                # elif loss_fea_abs >= 10:
                #     args.weight_fea = 0.01
                # else:
                #     args.weight_fea = 0.1
                # print(args.weight_fea)

                loss = weight_loss_t*loss_t + 0.01*loss_ft + weight_loss_mem * loss_mem +  weight_loss_diff * feature_diff + args.weight_fea*loss_fea

                g = grad(loss, adv_img, loss)[0]

                if args.ma:
                    print("use ma...")
                    momentum = 0.9
                    sample_total_g = momentum * sample_total_g + (1 - momentum) * g.detach()
                elif args.ags:
                    print("use ags...")
                    g_norm = torch.norm(g, p=2)
                    adaptive_alpha = args.alpha / (1e-6 + g_norm)
                    sample_total_g += g.detach()
                elif args.rop:
                    print("use rop..")
                    g_orth = g - torch.sum(g * perturbation, dim=0, keepdim=True) * perturbation
                    g_orth = g_orth / (torch.norm(g_orth, p=2) + 1e-8)
                    sample_total_g += g_orth.detach()
                elif args.ema:
                    print("use ema...")
                    beta = 0.95  # EMA系数
                    if step == 0:
                        ema_grad = g.detach()
                    else:
                        ema_grad = beta * ema_grad + (1 - beta) * g.detach()
                    sample_total_g += ema_grad
                elif args.gr:
                    print("use gr...")
                    lambda_reg = 0.1  # 正则化系数
                    reg_term = lambda_reg * torch.norm(sample_total_g, p=2)
                    loss_reg = loss + reg_term
                    g_reg = grad(loss_reg, adv_img, retain_graph=True)[0]
                    sample_total_g += g_reg.detach()
                else:
                    print(".........")
                    sample_total_g += g.detach()

                sample_step_count += 1
                if sample_step_count > 0:
                    avg_gradient = sample_total_g / sample_step_count
                    perturbation = (perturbation - avg_gradient.sign() * args.alpha).clamp(-eps, eps).detach()
                prev_adv_feature = adv_feature.detach()

                loss_t_val = safe_item(loss_t) if args.loss_t else 0
                loss_fea_val = safe_item(loss_fea) if args.loss_fea else 0
                loss_diff_val = safe_item(feature_diff) if args.loss_diff else 0

                loss_t_list.append(loss_t_val)
                loss_fea_list.append(loss_fea_val)
                loss_diff_list.append(loss_diff_val)
        video_losses.append({
            'Video': video_name,
            'Loss_T': loss_t_val,
            'Loss_Fea': loss_fea_val,
            'Loss_Diff': loss_diff_val
        })
    epochs = list(range(1, len(loss_t_list) + 1))
    # plt.figure(figsize=(10, 6))
    #
    # plt.plot(epochs, loss_t_list, label='Loss T (Binary CE)', marker='o')
    # plt.plot(epochs, loss_fea_list, label='Loss Fea (Infonce)', marker='s')
    # plt.plot(epochs, loss_diff_list, label='Loss Diff (Inter-frame)', marker='^')
    #
    # plt.xlabel('Step')
    # plt.ylabel('Loss Value')
    # plt.title('Loss Trends over Steps')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('loss_curve.png')
    # plt.show()

    # df = pd.DataFrame({
    #     'Step': epochs,
    #     'Loss_T': loss_t_list,
    #     'Loss_Fea': loss_fea_list,
    #     'Loss_Diff': loss_diff_list
    # })
    df = pd.DataFrame(video_losses)
    df.to_csv('loss_log_per_video.csv', index=False)

    #df.to_csv('loss_log.csv', index=False)


    # 保存全局扰动
    # uap_save_path = f"uap_file_main0/{args.train_dataset}_{args.train_prompts}_{args.seed}_{args.P_num}_{args.checkpoints}_{args.prompts_num}_uap.pth"
    # torch.save(perturbation.cpu(), uap_save_path)
    # print(f"\n✅ Global UAP saved to {uap_save_path}")

























