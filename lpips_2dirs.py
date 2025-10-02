import argparse
import os
import lpips
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0', '--dir0', type=str, default='/home/chongyu/Documents/Editing-Out-of-Domain-master/datasets/celeba1000', help='Reference directory')
parser.add_argument('-d1', '--dir1', type=str, default="tu6", help='Directory containing subfolders to compare')
parser.add_argument('-v', '--version', type=str, default='0.1', help='LPIPS version')

opt = parser.parse_args()

# Initialize the model
loss_fn = lpips.LPIPS(net='alex', version=opt.version)
loss_fn.cuda()

# Recursively find all subfolders in dir1, excluding those named "conc"
# Recursively find all subfolders in dir1, excluding those named "conc" or "heatmap"
subfolders = []
for root, dirs, files in os.walk(opt.dir1):
    # 跳过 conc 和 heatmap 文件夹
    dirs[:] = [d for d in dirs if d not in ("conc", "heatmap")]
    # 当前目录如果本身不是 conc 或 heatmap，就加入
    if os.path.basename(root) not in ("conc", "heatmap"):
        subfolders.append(root)

# 可选：排除根目录自身
subfolders = [f for f in subfolders if f != opt.dir1]
subfolders.sort()


# Output file path
output_file = os.path.join(opt.dir1, 'lpips_results.txt')

with open(output_file, 'w') as f:
    f.write('LPIPS Results Summary\n')
    f.write('====================\n\n')

    for subfolder in subfolders:
        dir1_path = subfolder
        distances = []

        # Get list of files in dir0 that exist in the current subfolder
        files = [file for file in os.listdir(opt.dir0)
                 if os.path.exists(os.path.join(dir1_path, file))]

        if not files:
            print(f'No matching files found between {opt.dir0} and {dir1_path}')
            continue

        for file in tqdm(files, desc=f'Processing {subfolder}'):
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0, file)))
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(dir1_path, file)))

            img0 = img0.cuda()
            img1 = img1.cuda()

            # Compute distance
            dist01 = loss_fn.forward(img0, img1).item()
            distances.append(dist01)

        # Compute statistics if we have valid distances
        if distances:
            mean = np.mean(distances)
            variance = np.var(distances)
            std_dev = np.std(distances)

            # Write only the summary statistics
            f.write(f'Subfolder: {subfolder}\n')
            f.write(f'Mean LPIPS: {mean:.6f}\n')
            f.write(f'Variance: {variance:.6f}\n')
            f.write(f'Standard Deviation: {std_dev:.6f}\n')
            f.write(f'Number of compared pairs: {len(distances)}\n')
            f.write('-' * 50 + '\n\n')

            print(f'Processed {subfolder}: Mean LPIPS = {mean:.6f} (based on {len(distances)} pairs)')

print(f'\nAll results saved to {output_file}')