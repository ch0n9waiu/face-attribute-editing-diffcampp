# Copyright (c) 2020, InterDigital R&D France. All rights reserved.
#
# This source code is made available under the license found in the
# LICENSE.txt in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm


class Conv2d(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, conv='conv', pad='mirror', norm='in', activ='relu',
                 sn=False):
        super(Conv2d, self).__init__()
        # Define padding
        if pad == 'mirror':
            self.padding = nn.ReflectionPad2d(kernel_size // 2)
        elif pad == 'none':
            self.padding = None
        else:
            self.padding = nn.ReflectionPad2d(pad)
        # Define conv layer
        if conv == 'conv':
            self.conv = nn.Conv2d(input_size, output_size, kernel_size=kernel_size, stride=stride)
        # Define norm layer
        if norm == 'in':
            self.norm = nn.InstanceNorm2d(output_size, affine=True)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(output_size)
        elif norm == 'none':
            self.norm = None
        # Define activation layer
        if activ == 'relu':
            self.relu = nn.ReLU()
        elif activ == 'leakyrelu':
            self.relu = nn.LeakyReLU(0.2)
        elif activ == 'none':
            self.relu = None
        # Use spectral norm
        if sn == True:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        if self.padding:
            out = self.padding(x)
        else:
            out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        if self.relu:
            out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, input_size, kernel_size, stride, conv='conv', pad='mirror', norm='in', activ='relu', sn=False):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            Conv2d(input_size, input_size, kernel_size=kernel_size, stride=stride, conv=conv, pad=pad, norm=norm,
                   activ=activ, sn=sn),
            Conv2d(input_size, input_size, kernel_size=kernel_size, stride=stride, conv=conv, pad=pad, norm=norm,
                   activ=activ, sn=sn)
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, input_size=3, activ='leakyrelu'):
        super(Encoder, self).__init__()
        self.conv_1 = Conv2d(input_size, 32, kernel_size=9, stride=1, activ=activ, sn=True)
        self.conv_2 = Conv2d(32, 64, kernel_size=3, stride=2, activ=activ, sn=True)
        self.conv_3 = Conv2d(64, 128, kernel_size=3, stride=2, activ=activ, sn=True)
        self.res_block = nn.Sequential(
            ResBlock(128, kernel_size=3, stride=1, activ=activ, sn=True),
            ResBlock(128, kernel_size=3, stride=1, activ=activ, sn=True),
            ResBlock(128, kernel_size=3, stride=1, activ=activ, sn=True),
            ResBlock(128, kernel_size=3, stride=1, activ=activ, sn=True)
        )

    def forward(self, x):
        out_1 = self.conv_1(x)
        out_2 = self.conv_2(out_1)
        out_3 = self.conv_3(out_2)
        out = self.res_block(out_3)
        return out, out_3, out_2


class Decoder(nn.Module):
    def __init__(self, output_size=3, activ='leakyrelu'):
        super(Decoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2d(256, 64, kernel_size=3, stride=1, activ=activ, sn=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2d(128, 32, kernel_size=3, stride=1, activ=activ, sn=True)
        )
        self.conv_3 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, output_size, kernel_size=9, stride=1)
        )
        self.fc_mix = nn.Linear(1024, 128, bias=False)

    def forward(self, x, skip_1, skip_2):
        b_s = x.size(0)
        z = torch.randn(b_s, 1024).type_as(x)
        y = F.sigmoid(z)
        y = self.fc_mix(y)
        y = F.sigmoid(y)
        b, c = y.size()
        y = y.view(b, c, 1, 1)
        out = x * y
        out = torch.cat((out, skip_1), 1)
        out = self.conv_1(out)
        out = torch.cat((out, skip_2), 1)
        out = self.conv_2(out)
        out = self.conv_3(out)
        return out


class Dis_PatchGAN(nn.Module):
    def __init__(self, input_size=3):
        super(Dis_PatchGAN, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(input_size, 32, kernel_size=4, stride=2, norm='none', activ='leakyrelu', sn=True),
            Conv2d(32, 64, kernel_size=4, stride=2, norm='batch', activ='leakyrelu', sn=True),
            Conv2d(64, 128, kernel_size=4, stride=2, norm='batch', activ='leakyrelu', sn=True),
            Conv2d(128, 256, kernel_size=4, stride=2, norm='batch', activ='leakyrelu', sn=True),
            Conv2d(256, 512, kernel_size=4, stride=1, norm='batch', activ='leakyrelu', sn=True),
            Conv2d(512, 1, kernel_size=4, stride=1, norm='none', activ='none', sn=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

def create_pyramid(img, n=1):
    """ Create an image pyramid.

    Args:
        img (torch.Tensor): An image tensor of shape (B, C, H, W)
        n (int): The number of pyramids to create

    Returns:
        list of torch.Tensor: The computed image pyramid.
    """
    # If input is a list or tuple return it as it is (probably already a pyramid)
    if isinstance(img, (list, tuple)):
        return img

    pyd = [img]
    for i in range(n - 1):
        pyd.append(nn.functional.avg_pool2d(pyd[-1], 3, stride=2, padding=1, count_include_pad=False))

    return pyd

def test():
    x = torch.randn((4, 3, 1024, 1024))
    code1, skip1, skip2 = Encoder(x)
    preds = Decoder(code1, skip1, skip2)
    print(preds.shape)
    assert preds.shape == x.shape
    print(preds.shape)


if __name__ == "__main__":
    test()
