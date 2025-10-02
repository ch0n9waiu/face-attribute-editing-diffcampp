import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
import torchvision.models as models
import torch.nn.functional as F
# class GANLoss(nn.Module):
#
#     def __init__(self,
#                  gan_type,
#                  real_label_val=1.0,
#                  fake_label_val=0.0,
#                  loss_weight=1.0):
#         super().__init__()
#         self.gan_type = gan_type
#         self.loss_weight = loss_weight
#         self.real_label_val = real_label_val
#         self.fake_label_val = fake_label_val
#
#         if self.gan_type == 'vanilla':
#             self.loss = nn.BCEWithLogitsLoss()
#         elif self.gan_type == 'lsgan':
#             self.loss = nn.MSELoss()
#         elif self.gan_type == 'wgan':
#             self.loss = self._wgan_loss
#         elif self.gan_type == 'hinge':
#             self.loss = nn.ReLU()
#         else:
#             raise NotImplementedError(
#                 f'GAN type {self.gan_type} is not implemented.')
#
#     def _wgan_loss(self, input, target):
#         return -input.mean() if target else input.mean()
#
#     def get_target_label(self, input, target_is_real):
#         if self.gan_type == 'wgan':
#             return target_is_real
#         target_val = (
#             self.real_label_val if target_is_real else self.fake_label_val)
#         return input.new_ones(input.size()) * target_val
#
#     def forward(self, input, target_is_real, is_disc=False):
#         target_label = self.get_target_label(input, target_is_real)
#         if self.gan_type == 'hinge':
#             if is_disc:  # for discriminators in hinge-gan
#                 input = -input if target_is_real else input
#                 loss = self.loss(1 + input).mean()
#             else:  # for generators in hinge-gan
#                 loss = -input.mean()
#         else:  # other gan types
#             loss = self.loss(input, target_label)
#
#         # loss_weight is always 1.0 for discriminators
#         return loss if is_disc else loss * self.loss_weight
#
class PerceptualVGG(nn.Module):

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg16',
                 use_input_norm=True,
                 pretrained=True):
        super().__init__()
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm

        # get vgg model and load pretrained vgg weight
        # remove _vgg from attributes to avoid `find_unused_parameters` bug
        _vgg = getattr(vgg, vgg_type)(pretrained=pretrained)
        # self.init_weights(_vgg, pretrained)
        num_layers = max(map(int, layer_name_list)) + 1
        assert len(_vgg.features) >= num_layers
        # only borrow layers that will be used from _vgg to avoid unused params
        self.vgg_layers = _vgg.features[:num_layers].cuda()

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                'mean',
                torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda())
            # the std is for image with range [-1, 1]
            self.register_buffer(
                'std',
                torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())

        for v in self.vgg_layers.parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}

        for name, module in self.vgg_layers.named_children():
            x = module(x)
            if name in self.layer_name_list:
                output[name] = x.clone()
        return output

class PerceptualLoss(nn.Module):

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg16',
                 use_input_norm=True,
                 perceptual_weight=1.0,
                 style_weight=1.0,
                 norm_img=True,
                 pretrained=True,
                 criterion='l1'):
        super().__init__()
        self.norm_img = norm_img
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = PerceptualVGG(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            pretrained=pretrained)
        # print(self.vgg)
        criterion = criterion.lower()
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in'
                ' this version.')

    def forward(self, x, gt):

        if self.norm_img:
            x = (x + 1.) * 0.5
            gt = (gt + 1.) * 0.5
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                percep_loss += self.criterion(
                    x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                style_loss += self.criterion(
                    self._gram_mat(x_features[k]),
                    self._gram_mat(gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        (n, c, h, w) = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

# from kornia.filters import laplacian
# def LaplacianLoss(x,y):
#     sobel_x = laplacian(x,3)
#     sobel_y = laplacian(y,3)
#     loss = F.l1_loss(sobel_x,sobel_y)
#     return loss


# class PerceptualLoss(nn.Module):
#     def __init__(self):
#         super(PerceptualLoss, self).__init__()
#         self.vgg = models.vgg19(pretrained=True).features[:15]
#
#         for param in self.vgg.parameters():
#             param.requires_grad = False
#
#     def forward(self, input, target):
#         # print(self.vgg)
#         input_vgg = self.vgg(input)
#         target_vgg = self.vgg(target)
#         loss = nn.MSELoss()(input_vgg, target_vgg)
#         return loss

import torch
import torch.nn.functional as F
from math import exp
import numpy as np


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels=x.shape[1], device=x.device))


def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out


def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current - up
        pyr.append(diff)
        current = down
    return pyr


class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=3, device=torch.device("cuda")):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)

    def forward(self, input, target):
        pyr_input = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

def random_crop_correlation_loss(input, target, crop_size=128):
    # get image size
    _, _, height, width = input.size()

    # calculate max crop position
    max_x = width - crop_size
    max_y = height - crop_size

    # generate random crop position
    x = torch.randint(0, max_x, size=(1,))
    y = torch.randint(0, max_y, size=(1,))

    # crop input and target
    input_cropped = input[:, :, y:y+crop_size, x:x+crop_size]
    target_cropped = target[:, :, y:y+crop_size, x:x+crop_size]

    # calculate correlation loss
    loss = 1 - F.cosine_similarity(input_cropped, target_cropped, dim=1).mean()

    return loss

class SobelLoss(torch.nn.Module):
    def __init__(self):
        super(SobelLoss, self).__init__()
        self.conv_x = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight.data = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]*3], dtype=torch.float)
        self.conv_y.weight.data = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]*3], dtype=torch.float)

    def forward(self, fake, real):
        fake_grad_x = self.conv_x(fake)
        real_grad_x = self.conv_x(real)
        fake_grad_y = self.conv_y(fake)
        real_grad_y = self.conv_y(real)
        grad_loss = torch.mean(torch.abs(fake_grad_x - real_grad_x) + torch.abs(fake_grad_y - real_grad_y))
        return grad_loss

class GANLoss(nn.Module):

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super().__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        return -input.mean() if target else input.mean()

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan':
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight

import torch
import torch.nn as nn
from torchvision import models
import torch.utils.model_zoo as model_zoo

__all__ = [
    'VGG',
    'vgg19',
    'vgg_fcn'
]

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, verification=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),   # for input 256, 8x8 instead of 7x7
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if verification:
            self.classifier = nn.Sequential(*list(self.classifier.children())[:-1])
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg19(num_classes=1000, pretrained=False, batch_norm=True, verifcation=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        num_classes(int): the number of classes at dataset
        pretrained (bool): If True, returns a model pre-trained on ImageNet
                            with a new FC layer 512x8x8 instead of 512x7x7
        batch_norm: if you want to introduce batch normalization
        verifcation (bool): Toggle verification mode (removes last fc from classifier)
    """
    if pretrained:
        kwargs['init_weights'] = True
    model = VGG(make_layers(cfg['E'], batch_norm=batch_norm), num_classes,  **kwargs)

    # if verifcation:
    #     verifier = nn.Sequential()
    #     for x in range(2):
    #         verifier.add_module(str(x), model.classifier[x])
    #     for x in range(3, 5):
    #         verifier.add_module(str(x), model.classifier[x])
    #     model.classifier = verifier

    if pretrained:
        # loading weights
        if batch_norm:
            pretrained_weights = model_zoo.load_url(model_urls['vgg19_bn'])
        else:
            pretrained_weights = model_zoo.load_url(model_urls['vgg19'])
        # loading only CONV layers weights
        for i in [0, 3, 6]:
            w = 'classifier.{}.weight'.format(str(i))
            new_w = 'not_used_{}'.format(str(i))
            b = 'classifier.{}.bias'.format(str(i))
            new_b ='not_used_{}'.format(str(i*10))
            pretrained_weights[new_w] = pretrained_weights.pop(w)
            pretrained_weights[new_b] = pretrained_weights.pop(b)

        model.load_state_dict(pretrained_weights, strict=False)

    return model


def vgg_fcn(num_classes=1000, pretrained=False, batch_norm=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
       num_classes(int): the number of classes at dataset
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        batch_norm: if you want to introduce batch normalization
    """
    if pretrained:
        kwargs['init_weights'] = True
    model = VGG(make_layers(cfg['D'], batch_norm=batch_norm), num_classes, **kwargs)

    if pretrained:
        # loading weights
        if batch_norm:
            pretrained_weights = model_zoo.load_url(model_urls['vgg19_bn'])
        else:
            pretrained_weights = model_zoo.load_url(model_urls['vgg19'])
        model.load_state_dict(pretrained_weights, strict=False)

    return model
# Adapted from https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
class Vgg19(torch.nn.Module):
    """ First layers of the VGG 19 model for the VGG loss.
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        model_path (str): Path to model weights file (.pth)
        requires_grad (bool): Enables or disables the "requires_grad" flag for all model parameters
    """
    def __init__(self, model_path: str = None, requires_grad: bool = False):
        super(Vgg19, self).__init__()
        if model_path is None:
            vgg_pretrained_features = models.vgg19(pretrained=True).features
        else:
            model = vgg19(pretrained=False)
            checkpoint = torch.load(model_path)
            del checkpoint['state_dict']['classifier.6.weight']
            del checkpoint['state_dict']['classifier.6.bias']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            vgg_pretrained_features = model.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Adapted from https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
class VGGLoss(nn.Module):
    """ Defines a criterion that captures the high frequency differences between two images.
    `"Perceptual Losses for Real-Time Style Transfer and Super-Resolution" <https://arxiv.org/pdf/1603.08155.pdf>`_

    Args:
        model_path (str): Path to model weights file (.pth)
    """
    def __init__(self, model_path: str = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19(model_path)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss