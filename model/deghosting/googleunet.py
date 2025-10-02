import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools


####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim * 4)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(2, 0, 1, 3)  # (B, C, H, W) -> (H, B, C, W)
        x = self.layer(x)
        x = x.permute(1, 2, 0, 3)  # (H, B, C, W) -> (B, C, H, W)
        x = self.norm(x)
        return x


class TransformerUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.transformer1 = TransformerBlock(64)
        self.transformer2 = TransformerBlock(128)
        self.transformer3 = TransformerBlock(256)
        self.transformer4 = TransformerBlock(512)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(64 + 256, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(self.pool(x1)))
        x3 = F.relu(self.conv3(self.pool(x2)))
        x4 = F.relu(self.conv4(self.pool(x3)))

        x1 = self.transformer1(x1)
        x2 = self.transformer2(x2)
        x3 = self.transformer3(x3)
        x4 = self.transformer4(x4)

        # Decoder
        x = self.upconv4(x4)
        x = torch.cat([x, x3], dim=1)
        x = F.relu(self.conv5(x))

        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.conv6(x))

        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = F.relu(self.conv7(x))
        x = self.conv8(x)
        return x


def test():
    x = torch.randn((1, 3, 1024, 1024))
    model = TransformerUNet(3, 3)
    # print(model)
    preds = model(x)
    # print(preds.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
