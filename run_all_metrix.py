import subprocess
import sys
import os

# 定义要顺序运行的脚本
scripts = [
    "compute_fid.py",
    "lpips_2dirs.py",
    "MAE_loss.py",
    "PSNR.py",
    "SSIM.py"
]

def run_script(script_name):
    """运行单个脚本"""
    print(f"\n==== 开始运行: {script_name} ====\n")
    try:
        # sys.executable 可以保证使用当前 Python 解释器
        result = subprocess.run([sys.executable, script_name],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"运行 {script_name} 出错：")
        print(e.stderr)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for script in scripts:
        script_path = os.path.join(base_dir, script)
        if os.path.exists(script_path):
            run_script(script_path)
        else:
            print(f"脚本不存在: {script_path}")

    print("\n==== 所有脚本执行完成 ====\n")
