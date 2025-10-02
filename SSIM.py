import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
import time
from torchmetrics import StructuralSimilarityIndexMeasure


def check_gpu_available():
    """检查GPU是否可用"""
    return torch.cuda.is_available()


def load_image_as_tensor(image_path, device):
    """加载图像并转换为PyTorch张量，移动到指定设备"""
    img = Image.open(image_path).convert('RGB')
    return ToTensor()(img).unsqueeze(0).to(device)


def calculate_ssim_between_images(img1, img2):
    """计算两张图片之间的SSIM"""
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(img1.device)
    return ssim(img1, img2).item()


def process_folders(folder1, folder2, device='cuda', relative_path=None):
    """处理文件夹并返回统计结果"""
    files1 = {f for f in os.listdir(folder1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    files2 = {f for f in os.listdir(folder2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}

    common_files = sorted(files1 & files2)
    print(f"找到 {len(common_files)} 对同名图片")

    if not common_files:
        raise ValueError("两个文件夹中没有同名的图片文件")

    ssim_values = []
    failed_count = 0

    for filename in tqdm(common_files, desc=f"计算 {relative_path or os.path.basename(folder2)} 的SSIM"):
        img1_path = os.path.join(folder1, filename)
        img2_path = os.path.join(folder2, filename)

        try:
            img1 = load_image_as_tensor(img1_path, device)
            img2 = load_image_as_tensor(img2_path, device)
            ssim = calculate_ssim_between_images(img1, img2)
            ssim_values.append(ssim)

            del img1, img2
            if device == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            failed_count += 1
            continue

    ssim_array = np.array(ssim_values)

    return {
        'folder_path': relative_path or folder2,
        'total_pairs': len(common_files),
        'valid_pairs': len(ssim_array),
        'failed_pairs': failed_count,
        'mean_ssim': np.mean(ssim_array),
        'std_ssim': np.std(ssim_array)
    }


def save_summary_to_txt(output_path, all_results, main_folder1, main_folder2):
    """保存所有子文件夹的统计摘要到一个TXT文件"""
    with open(output_path, 'w') as f:
        f.write(f"图片SSIM统计摘要\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"参考文件夹: {main_folder1}\n")
        f.write(f"目标主文件夹: {main_folder2}\n\n")
        f.write("=" * 50 + "\n\n")

        for result in all_results:
            f.write(f"子文件夹路径: {result['folder_path']}\n")
            f.write(f"比较图片对数: {result['total_pairs']}\n")
            f.write(f"成功对数: {result['valid_pairs']}\n")
            f.write(f"失败对数: {result['failed_pairs']}\n")
            f.write(f"平均SSIM: {result['mean_ssim']:.4f}\n")  # 保留4位小数
            f.write(f"SSIM标准差: {result['std_ssim']:.4f}\n")
            f.write("-" * 50 + "\n\n")


def get_all_subfolders(main_folder, base_folder, exclude=("conc", "heatmap")):
    """递归获取所有子文件夹，返回(完整路径, 相对路径)元组"""
    subfolders = []
    for root, dirs, files in os.walk(main_folder):
        # 排除掉不想遍历的子文件夹
        dirs[:] = [d for d in dirs if d not in exclude]
        # 当前文件夹本身如果在排除列表中，也跳过
        if os.path.basename(root) in exclude:
            continue
        relative_path = os.path.relpath(root, base_folder)
        if relative_path != '.':  # 排除根目录自身
            subfolders.append((root, relative_path))
    return subfolders



if __name__ == "__main__":
    # 配置参数
    main_folder1 = '/home/chongyu/Documents/Editing-Out-of-Domain-master/datasets/celeba1000'  # 参考文件夹
    main_folder2 = "tu6"  # 包含子文件夹的主目录

    # 输出文件路径
    summary_file = os.path.join(main_folder2, "ssim_summary.txt")

    # 检查文件夹存在
    if not os.path.isdir(main_folder1) or not os.path.isdir(main_folder2):
        raise ValueError("指定的文件夹路径不存在")

    # 获取所有子文件夹（带相对路径）
    sub_folders = get_all_subfolders(main_folder2, main_folder2)
    print(f"找到 {len(sub_folders)} 个子文件夹")

    # 设置计算设备
    device = torch.device('cuda' if check_gpu_available() else 'cpu')
    print(f"使用设备: {device}")

    # 处理所有子文件夹
    all_results = []
    for folder_path, relative_path in sub_folders:
        print(f"\n正在处理: {relative_path}")
        try:
            results = process_folders(main_folder1, folder_path, device, relative_path)
            all_results.append(results)

            # 打印当前结果
            print(f"\n[结果] {relative_path}")
            print(f"图片对数: {results['total_pairs']}")
            print(f"平均SSIM: {results['mean_ssim']:.4f} ± {results['std_ssim']:.4f}")

        except Exception as e:
            print(f"处理失败: {relative_path} - {str(e)}")
            continue

    # 保存汇总结果
    save_summary_to_txt(summary_file, all_results, main_folder1, main_folder2)
    print(f"\n处理完成！结果已保存至: {summary_file}")