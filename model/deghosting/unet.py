import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools


def init_weights(m):
    """Initialize layers with Xavier uniform distribution"""
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.01)
    elif type(m) == nn.Linear:
        nn.init.uniform_(m.weight, 0.0, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)




import numpy as np
def tensor2np(tensor):
    tensor = tensor.squeeze(0) \
        .float().detach().cpu().clamp_(0, 1)
    img_np = tensor.numpy()
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    img_np = (img_np * 255.0).round()
    img_np = img_np.astype(np.uint8)
    return img_np


def test():
    x = torch.randn((1, 6, 1024, 1024)).cuda()
    model = tensor2np((x + 1) / 2)
    # print(model)
    # preds = model(x)
    # print(preds.shape)
    # assert preds.shape == x.shape
    print(model.shape)


if __name__ == "__main__":
    test()
