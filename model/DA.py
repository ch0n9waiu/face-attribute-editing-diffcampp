import torch
import torch.nn as nn
from torchvision.models.resnet import resnet101
import numpy as np
import ttach as tta
import torch.nn.functional as F
import os
import matplotlib
matplotlib.use("Agg")  # 无GUI环境安全保存
import matplotlib.pyplot as plt
import numpy as np
import torch
def _save_matrix_heatmap(mat, save_path, title=None, cmap="coolwarm", add_colorbar=True):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(mat, torch.Tensor):
        mat = mat.detach().float().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    im = ax.imshow(mat, aspect="auto", cmap=cmap)
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Component index")
    if title: ax.set_title(title, fontsize=9)
    if add_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def _save_heatmap(array2d, save_path, title=None):
    """array2d: torch.Tensor[H,W] 或 np.ndarray[H,W]，保存为伪彩色图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(array2d, torch.Tensor):
        array2d = array2d.detach().float().cpu().numpy()
    vmin, vmax = np.min(array2d), np.max(array2d)
    fig = plt.figure(figsize=(4, 4), dpi=200)
    plt.imshow(array2d, cmap="magma", vmin=vmin, vmax=vmax)
    plt.axis("off")
    if title: plt.title(title, fontsize=8)
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def _save_spectrum(singular_values, save_path, title=None):
    """保存奇异值谱和累计解释率"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(singular_values, torch.Tensor):
        s = singular_values.detach().float().cpu().numpy()
    else:
        s = np.asarray(singular_values, dtype=np.float32)
    power = s**2
    ratio = power / (power.sum() + 1e-8)
    cum = np.cumsum(ratio)

    fig = plt.figure(figsize=(6, 3), dpi=200)
    ax1 = plt.gca()
    ax1.plot(s, marker='o', linewidth=1)
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Singular value")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(cum, marker='.', linewidth=1, color="orange")
    ax2.set_ylabel("Cumulative explained var.")
    ax2.set_ylim(0, 1.02)

    if title: plt.title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def _save_vector_bar(vector, save_path, title=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(vector, torch.Tensor):
        vector = vector.detach().float().cpu().numpy()
    fig = plt.figure(figsize=(6, 3), dpi=200)
    plt.bar(np.arange(len(vector)), vector, color="royalblue")
    if title: plt.title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _save_sigma_diag_as_heatmap(singular_values, save_path, title=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(singular_values, torch.Tensor):
        s = singular_values.detach().float().cpu().numpy()
    else:
        s = np.asarray(singular_values, dtype=np.float32)

    # 构造对角矩阵
    diag_matrix = np.zeros((len(s), len(s)), dtype=np.float32)
    np.fill_diagonal(diag_matrix, s)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    im = ax.imshow(diag_matrix, cmap="viridis")  # 比 magma 更亮
    plt.axis("off")
    if title: plt.title(title, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # 加色条
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def _save_heatmap(array2d, save_path, title=None):
    """干净版 + 正方形像素：用于 HxW 小图 (如 CAM、u_k_map)"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(array2d, torch.Tensor):
        array2d = array2d.detach().float().cpu().numpy()

    H, W = array2d.shape
    vmin, vmax = float(np.min(array2d)), float(np.max(array2d))

    # 目标最长边像素数，自动控制文件大小；每个 cell 等边
    max_pixels = 1024  # 可按需调，比如 2048
    cell_px = max(1, int(max_pixels / max(H, W)))
    dpi = 100
    fig_w, fig_h = (W * cell_px / dpi, H * cell_px / dpi)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(array2d, cmap="magma", vmin=vmin, vmax=vmax,
              interpolation="nearest", aspect="equal", origin="upper")
    ax.set_axis_off()
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_matrix_heatmap(mat, save_path, title=None, cmap="coolwarm", add_colorbar=True):
    """干净版 + 正方形像素：用于 U / Vh / S(1xr) 等任意矩阵"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(mat, torch.Tensor):
        mat = mat.detach().float().cpu().numpy()

    H, W = mat.shape  # 行×列
    # 设定最大输出像素长度，保证不爆图；每个 cell 为正方形
    max_pixels = 4096  # Vh 列很多时建议 4096；需要更细可增大
    cell_px = max(1, int(max_pixels / max(H, W)))
    dpi = 100
    fig_w, fig_h = (W * cell_px / dpi, H * cell_px / dpi)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(mat, cmap=cmap, interpolation="nearest", aspect="equal", origin="upper")
    ax.set_axis_off()
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_sigma_diag_as_heatmap(singular_values, save_path, title=None):
    """Σ 的对角矩阵热图：干净版 + 正方形像素"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(singular_values, torch.Tensor):
        s = singular_values.detach().float().cpu().numpy()
    else:
        s = np.asarray(singular_values, dtype=np.float32)

    S = np.zeros((len(s), len(s)), dtype=np.float32)
    np.fill_diagonal(S, s)

    # 复用矩阵热图渲染，保证正方形像素
    _save_matrix_heatmap(S, save_path, cmap="viridis")


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform

        target_layer.register_forward_hook(self.save_activation)

        #Backward compitability with older pytorch versions:
        if hasattr(target_layer, 'register_full_backward_hook'):
            target_layer.register_full_backward_hook(self.save_gradient)
        else:
            target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        # self.activations.append(activation.cpu().detach())
        self.activations.append(activation)

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        # print('save_gradient')
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        # self.gradients = [grad.cpu().detach()] + self.gradients
        self.gradients = [grad] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)


class BaseCAM:
    def __init__(self,
                 model,
                 target_layer,
                 use_cuda=False,
                 reshape_transform=None):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(self.model,
            target_layer, reshape_transform)


    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self,
                      input_tensor,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth=False):
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        weighted_activations = weights[:, :, None, None] * activations
        cam = torch.sum(weighted_activations, dim=1)  # weighted_activations.sum(axis=1)
        return cam

    def forward(self, input_tensor, target_category=None, eigen_smooth=False):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if type(target_category) is int:
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        else:
            assert(len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=False)

        activations = self.activations_and_grads.activations[-1]  # .cpu().data.numpy()
        grads = self.activations_and_grads.gradients[-1]  # .cpu().data.numpy()

        cam = self.get_cam_image(input_tensor, target_category,
            activations, grads, eigen_smooth)

        cam[cam < 0] = 0  # cam = np.maximum(cam, 0)

        result = []
        for img in cam:
            img = img.unsqueeze(0)
            img = img.unsqueeze(0)
            img = F.interpolate(img, input_tensor.shape[-2:][::-1], mode='bilinear')  # img = cv2.resize(img, input_tensor.shape[-2:][::-1])
            img = img.squeeze()
            img = img - torch.min(img)
            img = img / torch.max(img)
            result.append(img.unsqueeze(dim=0))
        result = torch.cat(result, dim=0)
        # result = np.float32(result)
        return result

    def forward_augmentation_smoothing(self,
                                       input_tensor,
                                       target_category=None,
                                       eigen_smooth=False):
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                target_category, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor,
                 target_category=None,
                 aug_smooth=False,
                 eigen_smooth=False):
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(input_tensor,
                target_category, eigen_smooth)

        return self.forward(input_tensor,
            target_category, eigen_smooth)

class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False,
        reshape_transform=None):
        super(GradCAM, self).__init__(model, target_layer, use_cuda, reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        return torch.mean(grads, dim=(2, 3))  # np.mean(grads, axis=(2, 3))


class GradCAMPP(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(GradCAMPP, self).__init__(model, target_layer, use_cuda, reshape_transform)

    def get_cam_weights(self, input_tensor, target_category, activations, grads):
        """
        GradCAM++的权重计算：
        1. 计算二阶梯度（通过梯度平方近似）
        2. 加权全局平均梯度
        """
        # 计算梯度平方（近似二阶梯度）
        grads_power_2 = grads.pow(2)
        # 计算梯度三次方（近似三阶梯度）
        grads_power_3 = grads_power_2 * grads

        # 公式中的分子项：二阶梯度
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        eps = 1e-6  # 避免除零
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)

        # 加权梯度权重
        weights = torch.maximum(grads, torch.zeros_like(grads)) * aij.squeeze()
        weights = torch.sum(weights, dim=(2, 3))  # 沿空间维度求和

        return weights

    def get_cam_image(self, input_tensor, target_category, activations, grads, eigen_smooth=False):
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        weighted_activations = weights[:, :, None, None] * activations
        cam = torch.sum(weighted_activations, dim=1)
        return cam


class GradCAMPlusPlus(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None,
                 debug=False, debug_dir="svd_debug", debug_topk=3):
        super(GradCAMPlusPlus, self).__init__(model, target_layer, use_cuda, reshape_transform)
        self.debug = debug
        self.debug_dir = debug_dir
        self.debug_topk = debug_topk  # 保存前K个左奇异向量热图


    def get_cam_weights(self, input_tensor, target_category, activations, grads):
        grads_power_2 = grads.pow(2)
        grads_power_3 = grads_power_2 * grads
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        eps = 1e-6
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
        weights = torch.maximum(grads, torch.zeros_like(grads)) * aij.squeeze()
        return torch.sum(weights, dim=(2, 3))

    def get_cam_image(self, input_tensor, target_category, activations, grads, eigen_smooth=False):
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        weighted_activations = weights[:, :, None, None] * activations

        if eigen_smooth:
            cam = self._eigen_smooth(weighted_activations)
        else:
            cam = torch.sum(weighted_activations, dim=1)
        return cam

    def _eigen_smooth(self, weighted_activations):
        B, C, H, W = weighted_activations.shape
        out = []
        with torch.no_grad():
            for i in range(B):
                A = weighted_activations[i]  # [C,H,W]
                cam = A.sum(0)  # [H,W]

                X = A.permute(1, 2, 0).reshape(H * W, C)
                X = X - X.mean(dim=0, keepdim=True)

                try:
                    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
                    u1 = U[:, 0]
                except RuntimeError:
                    U, S, V = torch.pca_lowrank(X, q=1, center=False)
                    u1 = U[:, 0];
                    Vh = V.t()

                if torch.dot(u1, cam.view(-1)) < 0:
                    u1 = -u1

                u1 = (u1 - u1.min()) / (u1.max() - u1.min() + 1e-8)
                u1_map = u1.view(H, W)

                smoothed = torch.relu(u1_map * cam)
                out.append(smoothed)

                # ================== 调试可视化 ==================
                if self.debug:
                    # 为本样本创建独立目录（可带 batch 索引）
                    sample_dir = os.path.join(self.debug_dir, f"batch_{i:03d}")
                    os.makedirs(sample_dir, exist_ok=True)

                    # 1) 基本可视化：cam / u1_map / smoothed
                    _save_heatmap(cam, os.path.join(sample_dir, "cam.png"), title="CAM (H×W)")
                    _save_heatmap(u1_map, os.path.join(sample_dir, "u1_map.png"), title="u1 map (H×W)")
                    _save_heatmap(smoothed, os.path.join(sample_dir, "smoothed.png"), title="Smoothed = ReLU(u1 ⊙ CAM)")

                    # 2) S（奇异值）：三种视图（谱图 / 1×r 热图 / 对角阵热图）
                    _save_spectrum(S, os.path.join(sample_dir, "S_spectrum.png"),
                                   title="Singular values & Cumulative ratio")
                    _save_matrix_heatmap(
                        S.unsqueeze(0), os.path.join(sample_dir, "S_row_heat.png"),
                        title="Sigma as 1×r heatmap", cmap="viridis", add_colorbar=True
                    )
                    _save_sigma_diag_as_heatmap(
                        S, os.path.join(sample_dir, "S_diag_heat.png"),
                        title="Sigma diagonal heatmap"
                    )

                    # 3) U / Vh 矩阵热力图（U: [HW,r], Vh: [r,C]）
                    _save_matrix_heatmap(
                        U, os.path.join(sample_dir, "U_heatmap.png"),
                        title=f"U heatmap (rows=pixels {H * W}, cols=components {U.size(1)})",
                        cmap="coolwarm", add_colorbar=True
                    )
                    _save_matrix_heatmap(
                        Vh, os.path.join(sample_dir, "Vh_heatmap.png"),
                        title=f"Vh heatmap (rows=components {Vh.size(0)}, cols=channels {Vh.size(1)})",
                        cmap="coolwarm", add_colorbar=True
                    )

                    # 4) 前K个左奇异向量 u_k 映射回 H×W（空间模式）
                    K = min(self.debug_topk, U.size(1))
                    for k in range(K):
                        uk = U[:, k]
                        # 与 CAM 对齐符号，避免整列取反导致的视觉翻色
                        if torch.dot(uk, cam.view(-1)) < 0:
                            uk = -uk
                        uk = (uk - uk.min()) / (uk.max() - uk.min() + 1e-8)
                        uk_map = uk.view(H, W)
                        _save_heatmap(uk_map, os.path.join(sample_dir, f"u{k + 1}_map.png"),
                                      title=f"u{k + 1} map (H×W)")

        return torch.stack(out, dim=0)

    def forward(self, input_tensor, target_category=None, eigen_smooth=False):
        """修改后的forward方法，添加retain_graph选项"""
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if type(target_category) is int:
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=False)  # 关键修改：允许保留计算图

        activations = self.activations_and_grads.activations[-1]
        grads = self.activations_and_grads.gradients[-1]

        cam = self.get_cam_image(input_tensor, target_category,
                                 activations, grads, eigen_smooth)

        # 清理计算图
        del output, loss
        # torch.cuda.empty_cache()

        cam = torch.relu(cam)
        result = []
        for img in cam:
            img = img.unsqueeze(0).unsqueeze(0)
            img = F.interpolate(img, input_tensor.shape[-2:][::-1], mode='bilinear')
            img = img.squeeze()
            img = (img - img.min()) / (img.max() - img.min() + 1e-10)
            result.append(img.unsqueeze(0))
        return torch.cat(result, dim=0)

    def forward_augmentation_smoothing(self, input_tensor, target_category=None, eigen_smooth=False):
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.Multiply(factors=[0.9, 1, 1.1]),
        ])

        cams = []
        for transform in transforms:
            try:
                augmented_tensor = transform.augment_image(input_tensor)
                cam = self.forward(augmented_tensor, target_category, eigen_smooth)

                # 处理维度并反增强
                cam = cam.unsqueeze(1)  # Bx1xHxW
                cam = transform.deaugment_mask(cam)
                cam = cam.squeeze(1)  # BxHxW
                cams.append(cam)
            finally:
                # 确保每次迭代后清理计算图
                torch.cuda.empty_cache()

        return torch.mean(torch.stack(cams), dim=0)

    def __call__(self, input_tensor, target_category=None, aug_smooth=False, eigen_smooth=False):
        try:
            if aug_smooth:
                return self.forward_augmentation_smoothing(input_tensor, target_category, eigen_smooth)
            return self.forward(input_tensor, target_category, eigen_smooth)
        finally:
            # 最终清理
            torch.cuda.empty_cache()


class DiffCam(nn.Module):
    def __init__(self, num_class):
        super(DiffCam, self).__init__()
        resnet = list(resnet101(pretrained=True).children())
        resnet_layer3_1 = resnet[6][:10]
        resnet_layer3_2 = resnet[6][10:]
        self.shared = torch.nn.Sequential(*(resnet[0:6]), *resnet_layer3_1)
        self.classifier = torch.nn.Sequential(*resnet_layer3_2, *(resnet[7:9]))
        # self.cam = GradCAMPlusPlus(model=self, target_layer=self.classifier[-2], debug=False, debug_dir="svd_debug", debug_topk=3)
        self.cam = GradCAM(model=self, target_layer=self.classifier[-2])
        self.fc = nn.Linear(2048, num_class)

    def forward(self, image):  # [n, 6, h, w]
        inverted = image[:, 0:3, :, :]
        manipulated = image[:, 3:6, :, :]
        feat1, feat2 = self.shared(inverted), self.shared(manipulated)
        feat_delta = feat2 - feat1
        out = self.classifier(feat_delta)  # [n, 2048, 1, 1]
        out = out.squeeze()
        out = self.fc(out)
        if len(out.shape) == 1:
            out = out.unsqueeze(0)
        return out