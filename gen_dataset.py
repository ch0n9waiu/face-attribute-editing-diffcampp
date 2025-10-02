import os
import cv2
import random
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
# from pytorch_grad_cam.utils.image import show_cam_on_image
from options.gen_dataset_opts import GenDatasetOpts
from model.pSp.psp import pSp
from model.DA import DiffCam

# define mapping from direction names to output indices
direction2idx = {'Bushy_Eyebrows': 6, 'Eyeglasses': 7, 'Mouth_Open': 10, 'Narrow_Eyes': 11, 'Beard': 12, 'Smiling': 15,
                 'Old': 16}

from face_parsing.face_parsing import evaluate
from model.blending import blend_imgs


class Empty:
    pass


def tensor2np(tensor):
    tensor = tensor.squeeze(0) \
        .float().detach().cpu().clamp_(0, 1)
    img_np = tensor.numpy()
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    img_np = (img_np * 255.0).round()
    img_np = img_np.astype(np.uint8)
    return img_np


def main(opts):
    opts.device = "cuda:" + str(opts.device)
    diffcam = DiffCam(opts.diffcam_num_class)
    diffcam_state = torch.load(opts.diffcam_ckpt_path)
    diffcam.load_state_dict(diffcam_state)
    psp_opts = Empty()
    for attr in dir(opts):
        if 'psp' in attr:
            exec(f"psp_opts.{attr.replace('psp_', '')} = opts.{attr}")
    psp_opts.device = opts.device
    psp = pSp(psp_opts)
    psp = psp.to(opts.device)
    diffcam = diffcam.to(opts.device)
    psp.eval(), diffcam.eval()
    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    totensor2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([512, 512]),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    for path in os.listdir(opts.direction_dir):
        assert os.path.splitext(path)[0] in direction2idx.keys(), \
            'direction name not in dict'
    direction_paths = [os.path.join(opts.direction_dir, path)
                       for path in os.listdir(opts.direction_dir)]
    direction_names = [os.path.splitext(os.path.basename(path))[0]
                       for path in direction_paths]
    directions = [np.load(path) for path in direction_paths]
    directions = [direction / np.sqrt((direction * direction).sum())
                  for direction in directions]
    directions = [torch.from_numpy(direction).float().to(opts.device).unsqueeze(0)
                  for direction in directions]

    cp = 'cp/79999_iter.pth'

    for path in tqdm(os.listdir(opts.src_image_dir)):
        # parsing
        image_path = os.path.join(opts.src_image_dir, path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img1 = totensor(img)

        idx = random.randint(0, len(direction_names) - 1)
        target_idx = direction2idx[direction_names[idx]]
        direction = directions[idx]
        with torch.no_grad():
            origin_1024 = img1.unsqueeze(0).to(opts.device)
            origin = F.interpolate(origin_1024,
                                   (opts.diffcam_img_size, opts.diffcam_img_size),
                                   mode='area')
            inverted, latent = psp(origin, resize=True, return_latents=True)

            latent_pi = latent + opts.alpha * direction
            manipulated, _ = psp.decoder([latent_pi], input_is_latent=True, return_latents=False)
            manipulated_256 = psp.face_pool(manipulated)
            image_forward = torch.cat((inverted, manipulated_256), dim=1)
            manipulated_np = tensor2np((manipulated + 1) / 2)

            img2 = totensor2(manipulated_np)
            parsing123 = evaluate(img2, cp)
            new_arr = np.zeros_like(parsing123)
            values = (
                [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14]
                if target_idx in [12, 16]
                else [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
            )
            new_arr[np.isin(parsing123, values)] = 1
            parsing = cv2.resize(new_arr, img.shape[0:2], interpolation=cv2.INTER_NEAREST)
            parsing = torch.from_numpy(parsing).cuda().unsqueeze(0)
            parsing_np = parsing.squeeze().cpu().detach().numpy()
            height, width = parsing_np.shape
            image_manipu = np.zeros((height, width, 3), dtype=np.uint8)
            image_manipu[parsing_np == 1] = [59, 144, 238]
            image_manipu[parsing_np == 0] = [255, 255, 255]

            '''
            4,5,6是左眼，右眼，眼镜
            '''
            img1 = totensor2(img)
            parsing456 = evaluate(img1, cp)
            new_arr1 = np.zeros_like(parsing456)

            _BASE = (1, 2, 3, 10)
            _EXTRA_12 = (14,)
            _EXTRA_16 = (11, 12, 13, 14)
            _EXTRA_DEFAULT = (11, 12, 13)

            _TARGET_VALUES = {
                6: _BASE,
                12: _BASE + _EXTRA_12,
                16: _BASE + _EXTRA_16,
            }

            values = _TARGET_VALUES.get(target_idx, _BASE + _EXTRA_DEFAULT)
            new_arr1[np.isin(parsing456, values)] = 1
            parsing1 = cv2.resize(new_arr1, img.shape[0:2], interpolation=cv2.INTER_NEAREST)
            parsing1 = torch.from_numpy(parsing1).cuda().unsqueeze(0)
            parsing_np1 = parsing1.squeeze().cpu().detach().numpy()
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[parsing_np1 == 1] = [70, 199, 245]
            image[parsing_np1 == 0] = [255, 255, 255]

            new_arr2 = np.ones_like(parsing456)
            _TARGET_IDXS = {15, 16}  # set
            _VALUES_IF_TARGET = (11, 12, 13)  # tuple (immutable, faster)
            _EMPTY_VALUES = ()  # Empty tuple

            values = _VALUES_IF_TARGET if target_idx in _TARGET_IDXS else _EMPTY_VALUES
            new_arr2[np.isin(parsing456, values)] = 0
            parsing2 = cv2.resize(new_arr2, img.shape[0:2], interpolation=cv2.INTER_NEAREST)
            parsing2 = torch.from_numpy(parsing2).cuda().unsqueeze(0)
            parsing_np2 = parsing2.squeeze().cpu().detach().numpy()
            image2 = np.zeros((height, width, 3), dtype=np.uint8)
            image2[parsing_np2 == 1] = [59, 144, 238]
            image2[parsing_np2 == 0] = [255, 255, 255]

            new_arr3 = np.ones_like(parsing123)
            _TARGET_IDXS = {15, 16}  # set
            _VALUES_IF_TARGET = (11, 12, 13)  # tuple (immutable, faster)
            _EMPTY_VALUES = ()  # Empty tuple

            values = _VALUES_IF_TARGET if target_idx in _TARGET_IDXS else _EMPTY_VALUES
            new_arr3[np.isin(parsing123, values)] = 0
            parsing3 = cv2.resize(new_arr3, img.shape[0:2], interpolation=cv2.INTER_NEAREST)
            parsing3 = torch.from_numpy(parsing3).cuda().unsqueeze(0)
            parsing_np3 = parsing3.squeeze().cpu().detach().numpy()
            image3 = np.zeros((height, width, 3), dtype=np.uint8)
            image3[parsing_np3 == 1] = [70, 199, 245]
            image3[parsing_np3 == 0] = [255, 255, 255]

            mask = parsing * parsing1
            mask2 = parsing2 * parsing3
            mask_np = mask.squeeze().cpu().detach().numpy()
            image_mask = np.zeros((height, width, 3), dtype=np.uint8)
            image_mask[mask_np == 1] = [87, 161, 73]
            image_mask[mask_np == 0] = [255, 255, 255]

            mask2_np = mask2.squeeze().cpu().detach().numpy()
            image_mask2 = np.zeros((height, width, 3), dtype=np.uint8)
            image_mask2[mask2_np == 1] = [87, 161, 73]
            image_mask2[mask2_np == 0] = [255, 255, 255]
        heat_map = diffcam.cam(
            input_tensor=image_forward,
            target_category=target_idx,
            aug_smooth=True,
            eigen_smooth=target_idx != 16  # 只有当target_idx不等于16时才启用eigen_smooth
        )

        with torch.no_grad():
            heat_map = heat_map.unsqueeze(0)
            heat_map = F.interpolate(heat_map, (1024, 1024))
            # DiffCam_np = heat_map.squeeze().cpu().detach().numpy()
            # heat_map = heat_map * mask
            # SD_np = heat_map.squeeze().cpu().detach().numpy()

            fused = manipulated * (heat_map * mask2 + (1 - mask2)) + origin_1024 * mask2 * (1 - heat_map)
            blend = blend_imgs(fused, origin_1024, mask.byte()).to(opts.device)

        fused_np = tensor2np((fused + 1) / 2)
        blend_np = tensor2np((blend + 1) / 2)
        # visual1 = np.concatenate((fused_np, blend_np), 1)
        # origin_np = tensor2np((origin_1024 + 1) / 2)
        # manipulated_np = tensor2np((manipulated + 1) / 2)
        # heat_map = heat_map.squeeze().cpu().detach().numpy()
        # visual1 = np.concatenate((origin_np, fused_np, blend_np), 1)
        # visual1 = np.concatenate((origin_np, manipulated_np, fused_np1, fused_np, blend_np), 1)
        # heat_visual = np.concatenate((origin_np, manipulated_np, fused_np, blend_np), 1)
        # heat_visual1 = np.concatenate((heat_visual, fused_np), 1)
        # heat_visual1 = np.concatenate((fused_np, blend_np), 1)
        # os.makedirs('./new/org_seg', exist_ok=True)
        os.makedirs('./new/manipu_seg', exist_ok=True)
        os.makedirs('./new/manipu_seg2', exist_ok=True)
        visual = np.concatenate((image_manipu, image, image_mask), 1)
        visual2 = np.concatenate((image2, image3, image_mask2), 1)
        cv2.imwrite(os.path.join('./new/manipu_seg', path), visual)
        cv2.imwrite(os.path.join('./new/manipu_seg2', path), visual2)
        # os.makedirs('./new/org', exist_ok=True)
        # os.makedirs('./new/manipu', exist_ok=True)
        # os.makedirs('./new/mask', exist_ok=True)
        # os.makedirs('./new/Diff', exist_ok=True)
        # os.makedirs('./new/SD', exist_ok=True)
        # os.makedirs(opts.dst_image_dir, exist_ok=True)
        # os.makedirs(opts.blend_image_dir, exist_ok=True)
        # os.makedirs('./gen_diff/conc', exist_ok=True)
        # cv2.imwrite(os.path.join('./new/org_seg', path), image)
        # cv2.imwrite(os.path.join('./new/manipu_seg', path), image_manipu)
        # cv2.imwrite(os.path.join('./new/org', path), origin_np)
        # cv2.imwrite(os.path.join('./new/manipu', path), manipulated_np)
        # cv2.imwrite(os.path.join('./new/mask', path), image_mask)
        # cv2.imwrite(os.path.join(opts.dst_image_dir, path), fused_np)
        # cv2.imwrite(os.path.join(opts.blend_image_dir, path), blend_np)
        # cv2.imwrite(os.path.join('./gen_diff/conc', path), visual1)


if __name__ == '__main__':
    opts = GenDatasetOpts().parse()
    print(opts.src_image_dir)
    main(opts)
