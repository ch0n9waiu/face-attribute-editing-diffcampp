import warnings
warnings.filterwarnings("ignore", message="Default upsampling behavior when mode=bilinear is changed to align_corners=False")
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from face_parsing.face_parsing import evaluate
from options.image_process_opts import ImageProcessOpts
from model.pSp.psp import pSp
from model.deghosting.deghosting import Deghosting
from model.deghosting.MAResUNet import MAResUNet
from model.DA import DiffCam
from pytorch_grad_cam.utils.image import show_cam_on_image
from model.blending import blend_imgs

direction2idx = {'Bushy_Eyebrows': 6, 'Eyeglasses': 7, 'Mouth_Open': 10, 'Narrow_Eyes': 11, 'Beard': 12, 'Smiling': 15,
                 'Old': 16}


# direction2idx = {'Bushy_Eyebrows': 1, 'Eyeglasses': 2, 'Mouth_Open': 3, 'Narrow_Eyes': 4, 'Beard': 0, 'Smiling': 6,
#                   'Old': 5}


class Empty:
    pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


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
    # deghosting = Deghosting(opts.deghosting_in_size, opts.deghosting_out_size)
    deghosting = MAResUNet()
    total_params = count_parameters(deghosting)
    print(f"Total parameters: {total_params:,}")
    deghosting_state = torch.load(opts.deghosting_ckpt_path)
    deghosting.load_state_dict(deghosting_state)
    psp_opts = Empty()
    for attr in dir(opts):
        if 'psp' in attr:
            exec(f"psp_opts.{attr.replace('psp_', '')} = opts.{attr}")
    psp_opts.device = opts.device
    psp = pSp(psp_opts)
    psp = psp.to(opts.device)
    diffcam = diffcam.to(opts.device)
    deghosting = deghosting.to(opts.device)
    psp.eval(), diffcam.eval(), deghosting.eval()
    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    totensor2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([512, 512]),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    direction_name = os.path.splitext(os.path.basename(opts.direction_path))[0]
    assert direction_name in direction2idx.keys(), 'direction name not in dict'
    target_idx = direction2idx[direction_name]
    direction = np.load(opts.direction_path)
    direction = direction / np.sqrt((direction * direction).sum())
    direction = torch.from_numpy(direction).float().to(opts.device).unsqueeze(0)
    cp = 'cp/79999_iter.pth'

    def load_single_image(image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV默认BGR → 转RGB
        return img  # 添加 batch 维度 (1, C, H, W)

    image_path = opts.image_path  # 替换成你的文件路径
    file_name_with_ext = os.path.basename(image_path)
    input_tensor = load_single_image(image_path)
    print("Tensor shape:", input_tensor.shape)  # 应该是 [1, 3, H, W]
    img1 = totensor(input_tensor)
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
        # new_arr = np.zeros_like(parsing123)
        # if target_idx in [12, 16]:
        #     values = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14]
        # else:
        #     values = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
        # new_arr[np.isin(parsing123, values)] = 1
        # parsing = cv2.resize(new_arr, img.shape[0:2], interpolation=cv2.INTER_NEAREST)
        # parsing = torch.from_numpy(parsing).cuda().unsqueeze(0)
        # parsing_np = parsing.squeeze().cpu().detach().numpy()
        # height, width = parsing_np.shape
        # image_manipu = np.zeros((height, width, 3), dtype=np.uint8)
        # image_manipu[parsing_np == 1] = [59, 144, 238]
        # image_manipu[parsing_np == 0] = [255, 255, 255]

        '''
        4,5,6是左眼，右眼，眼镜
        '''
        img1 = totensor2(input_tensor)
        parsing456 = evaluate(img1, cp)
        # new_arr1 = np.zeros_like(parsing456)
        #
        # if target_idx in [6]:
        #     values = [1, 2, 3, 10]
        # elif target_idx in [12]:
        #     values = [1, 2, 3, 10, 14]
        # elif target_idx in [16]:
        #     values = [1, 2, 3, 10, 11, 12, 13, 14]
        # else:
        #     values = [1, 2, 3, 10, 11, 12, 13]
        # new_arr1[np.isin(parsing456, values)] = 1
        # parsing1 = cv2.resize(new_arr1, img.shape[0:2], interpolation=cv2.INTER_NEAREST)
        # parsing1 = torch.from_numpy(parsing1).cuda().unsqueeze(0)
        # parsing_np1 = parsing1.squeeze().cpu().detach().numpy()

        # image = np.zeros((height, width, 3), dtype=np.uint8)
        # image[parsing_np1 == 1] = [70, 199, 245]
        # image[parsing_np1 == 0] = [255, 255, 255]

        new_arr2 = np.ones_like(parsing456)
        if target_idx in [15, 16]:
            values = [11, 12, 13]
        else:
            values = []
        new_arr2[np.isin(parsing456, values)] = 0
        parsing2 = cv2.resize(new_arr2, input_tensor.shape[0:2], interpolation=cv2.INTER_NEAREST)
        parsing2 = torch.from_numpy(parsing2).cuda().unsqueeze(0)

        new_arr3 = np.ones_like(parsing123)
        if target_idx in [15, 16]:
            values = [11, 12, 13]
        else:
            values = []
        new_arr3[np.isin(parsing123, values)] = 0
        parsing3 = cv2.resize(new_arr3, input_tensor.shape[0:2], interpolation=cv2.INTER_NEAREST)
        parsing3 = torch.from_numpy(parsing3).cuda().unsqueeze(0)

        # mask = parsing * parsing1
        mask2 = parsing2 * parsing3
        # mask2[mask2 > 0] = 1
        # mask_np = mask.squeeze().cpu().detach().numpy()
        # image_mask = np.zeros((height, width, 3), dtype=np.uint8)
        # image_mask[mask_np == 1] = [87, 161, 73]
        # image_mask[mask_np == 0] = [255, 255, 255]

    heat_map = diffcam.cam(
        input_tensor=image_forward,
        target_category=target_idx,
        aug_smooth=False,
        eigen_smooth=target_idx != 16  # 只有当target_idx不等于16时才启用eigen_smooth
    )

    with torch.no_grad():
        heat_map = heat_map.unsqueeze(0)
        heat_map = F.interpolate(heat_map, (1024, 1024))
        # heat_map = heat_map * mask
        fused2 = manipulated * (heat_map * mask2 + (1 - mask2)) + origin_1024 * mask2 * (1 - heat_map)
        # fused = F.interpolate(fused,
        #                       (opts.deghosting_in_size, opts.deghosting_in_size),
        #                       mode='area')
        output = deghosting(torch.cat((fused2, origin_1024), dim=1))
        # output = deghosting(fused)
        # blend = blend_imgs(fused2, origin_1024, mask.byte()).to(opts.device)
    # blend_np = tensor2np((blend + 1) / 2)
    fused_np = tensor2np((fused2 + 1) / 2)
    inverted_np = tensor2np((inverted + 1) / 2)
    output_np = tensor2np((output + 1) / 2)
    origin_1024_np = tensor2np((origin_1024 + 1) / 2)
    manipulated_np = tensor2np((manipulated + 1) / 2)
    heat_map = heat_map.squeeze().cpu().detach().numpy()
    heat_visual = show_cam_on_image(manipulated_np / 255.0, heat_map)
    img_np = np.concatenate((heat_visual, output_np), axis=1)
    os.makedirs('output', exist_ok=True)
    cv2.imwrite(os.path.join('output', file_name_with_ext), output_np)


if __name__ == '__main__':
    opts = ImageProcessOpts().parse()
    main(opts)
