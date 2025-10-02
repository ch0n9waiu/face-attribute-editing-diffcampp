import os
import subprocess
from pathlib import Path


def find_leaf_dirs(root_dir):
    """递归查找所有最后一层的子文件夹，跳过名为'conc'的文件夹"""
    leaf_dirs = []
    for root, dirs, files in os.walk(root_dir):
        # 跳过名为'conc'的文件夹
        dirs[:] = [d for d in dirs if d != 'conc']
        if not dirs:  # 如果没有子文件夹，就是最后一层
            leaf_dirs.append(root)
    return leaf_dirs


def compute_fid(reference, target_dirs, output_file):
    """计算每个目标文件夹与参考文件之间的FID"""
    with open(output_file, 'w') as f:
        for target_dir in target_dirs:
            # 如果路径最后一层文件夹名是 heatmap，则跳过
            if Path(target_dir).name == "heatmap":
                print(f"Skipping {target_dir} (heatmap folder)")
                f.write(f"{target_dir}: SKIPPED (heatmap folder)\n")
                continue

            print(f"Processing: {target_dir}")
            try:
                cmd = f"python -m pytorch_fid {reference} {target_dir}"
                result = subprocess.run(cmd, shell=True, check=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True)

                output = result.stdout
                fid_line = [line for line in output.split('\n') if line.startswith('FID:')][0]
                fid_value = fid_line.split(':')[1].strip()

                f.write(f"{target_dir}: {fid_value}\n")
                print(f"Completed: {target_dir} - FID: {fid_value}")
            except subprocess.CalledProcessError as e:
                error_msg = f"Error processing {target_dir}: {e.stderr}"
                f.write(f"{target_dir}: ERROR - {error_msg}\n")
                print(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error with {target_dir}: {str(e)}"
                f.write(f"{target_dir}: ERROR - {error_msg}\n")
                print(error_msg)



if __name__ == "__main__":
    # 设置路径
    reference_npz = "./pic4score/datasets.npz"
    score_dir = "tu6"
    output_file = os.path.join(score_dir, "summary.txt")

    # 查找所有最后一层子文件夹
    leaf_dirs = find_leaf_dirs(score_dir)
    print(f"Found {len(leaf_dirs)} leaf directories to process.")

    # 计算FID并保存结果
    compute_fid(reference_npz, leaf_dirs, output_file)
    print(f"All done! Results saved to {output_file}")