"""
下载缺失瓦片的完整流程脚本
1. 从 AEF 索引中查找缺失城市对应的瓦片
2. 下载这些瓦片
3. 重新运行提取脚本

用法:
    python run_download_missing.py [--step STEP] [--year YEAR]

参数:
    --step: 运行哪一步
        1: 只运行查找瓦片
        2: 只运行下载
        3: 只运行重新提取
        all: 运行全部步骤 (默认)
    --year: 只处理指定年份
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_step1():
    """运行 step1: 查找缺失瓦片"""
    print("\n" + "="*60)
    print("Step 1: 从 AEF 索引中查找缺失城市对应的瓦片")
    print("="*60 + "\n")

    script_path = Path(__file__).parent / "step1_find_missing_tiles.py"
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    return result.returncode == 0


def run_step2(year=None):
    """运行 step2: 下载瓦片"""
    print("\n" + "="*60)
    print("Step 2: 下载缺失的瓦片")
    print("="*60 + "\n")

    script_path = Path(__file__).parent / "step2_download_missing_tiles.py"
    cmd = [sys.executable, str(script_path)]

    if year:
        cmd.extend(['--year', str(year)])

    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def run_step3():
    """运行 step3: 重新提取城市数据"""
    print("\n" + "="*60)
    print("Step 3: 重新运行城市数据提取")
    print("="*60 + "\n")

    script_path = Path(__file__).parent / "run_extraction.py"
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='下载缺失瓦片并重新提取数据')
    parser.add_argument('--step', choices=['1', '2', '3', 'all'], default='all',
                        help='运行哪一步 (1/2/3/all)')
    parser.add_argument('--year', type=int, default=None,
                        help='只处理指定年份')

    args = parser.parse_args()

    print("="*60)
    print("缺失瓦片下载与数据提取流程")
    print("="*60)

    if args.step in ['1', 'all']:
        success = run_step1()
        if not success and args.step == 'all':
            print("\n[ERROR] Step 1 failed, stopping.")
            return

    if args.step in ['2', 'all']:
        success = run_step2(args.year)
        if not success and args.step == 'all':
            print("\n[WARNING] Step 2 had some failures, but continuing...")

    if args.step in ['3', 'all']:
        run_step3()

    print("\n" + "="*60)
    print("流程完成！")
    print("="*60)


if __name__ == "__main__":
    main()
