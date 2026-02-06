"""
下载缺失城市的 AEF 卫星数据瓦片
基于 step1_find_missing_tiles.py 生成的配置文件进行下载

用法:
    python step2_download_missing_tiles.py [--year YEAR] [--no-vrt] [--retries N]

示例:
    python step2_download_missing_tiles.py              # 下载所有缺失瓦片
    python step2_download_missing_tiles.py --year 2020  # 只下载2020年的
    python step2_download_missing_tiles.py --no-vrt     # 不下载VRT文件
"""

import os
import json
import requests
from tqdm import tqdm
from pathlib import Path
import time
import argparse


class MissingTileDownloader:
    def __init__(self, config_file=None, output_base_dir=None):
        """
        初始化下载器

        Args:
            config_file: 包含瓦片信息的JSON文件路径
            output_base_dir: 下载文件的基础输出目录
        """
        # 设置默认路径
        base_dir = Path("/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/data")

        if config_file is None:
            self.config_file = base_dir / "pretrain_city_tiles" / "missing_tiles_to_download.json"
        else:
            self.config_file = Path(config_file)

        if output_base_dir is None:
            # 下载到与原始数据相同的目录
            self.output_dir = base_dir / "china_city_data_1211" / "downloads"
        else:
            self.output_dir = Path(output_base_dir)

        self.base_url = "https://data.source.coop/tge-labs/aef/v1/annual"

        # 创建下载目录
        self.tiff_dir = self.output_dir / "tiff_files"
        self.vrt_dir = self.output_dir / "vrt_files"
        self.tiff_dir.mkdir(parents=True, exist_ok=True)
        self.vrt_dir.mkdir(parents=True, exist_ok=True)

        # 加载配置
        self.load_config()

        # 下载统计
        self.stats = {
            'total': 0,
            'downloaded': 0,
            'skipped': 0,
            'failed': 0
        }

    def load_config(self):
        """加载下载配置"""
        print(f"Loading configuration from {self.config_file}...")
        with open(self.config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.tiles = data['tiles']
        self.total_tiles = data['total_tiles']
        print(f"Found {self.total_tiles} tiles to download")

    def download_file(self, url, output_path, desc=""):
        """
        下载单个文件

        Args:
            url: 下载URL
            output_path: 输出路径
            desc: 进度条描述

        Returns:
            bool: 下载是否成功
        """
        if output_path.exists():
            self.stats['skipped'] += 1
            return True

        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))

                with open(output_path, 'wb') as f, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=desc,
                    leave=False
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            self.stats['downloaded'] += 1
            return True

        except Exception as e:
            print(f"\n[ERROR] Failed to download {url}: {e}")
            self.stats['failed'] += 1
            # 删除部分下载的文件
            if output_path.exists():
                output_path.unlink()
            return False

    def extract_filename_from_s3_url(self, s3_url):
        """从 s3_url 中提取实际的文件名"""
        return s3_url.split('/')[-1]

    def construct_vrt_filename(self, tiff_filename):
        """根据 TIFF 文件名构建对应的 VRT 文件名"""
        return tiff_filename.replace('.tiff', '.vrt')

    def download_tile(self, tile_info):
        """
        下载单个瓦片（包括TIFF和VRT）

        Args:
            tile_info: 瓦片信息字典

        Returns:
            tuple: (tiff_success, vrt_success)
        """
        year = tile_info['year']
        zone = tile_info['utm_zone']
        s3_url = tile_info['s3_url']

        # 从 s3_url 中提取实际的 TIFF 文件名
        tiff_name = self.extract_filename_from_s3_url(s3_url)

        # 构建对应的 VRT 文件名
        vrt_name = self.construct_vrt_filename(tiff_name)

        # 构建 TIFF URL
        tiff_url = s3_url.replace(
            "s3://us-west-2.opendata.source.coop",
            "https://data.source.coop"
        )

        # 构建 VRT URL
        vrt_url = tiff_url.replace('.tiff', '.vrt')

        # 构建输出路径
        tiff_out = self.tiff_dir / f"{year}_{zone}_{tiff_name}"
        vrt_out = self.vrt_dir / f"{year}_{zone}_{vrt_name}"

        # 下载TIFF
        tiff_success = self.download_file(
            tiff_url,
            tiff_out,
            f"{year} {zone} TIFF"
        )

        # 下载VRT
        vrt_success = self.download_file(
            vrt_url,
            vrt_out,
            f"{year} {zone} VRT"
        )

        return tiff_success, vrt_success

    def download_all(self, download_vrt=True, max_retries=3):
        """
        下载所有瓦片

        Args:
            download_vrt: 是否同时下载VRT文件
            max_retries: 失败重试次数
        """
        print(f"\n{'='*60}")
        print(f"开始下载 {len(self.tiles)} 个缺失瓦片")
        print(f"输出目录: {self.output_dir}")
        print(f"  - TIFF 文件: {self.tiff_dir}")
        print(f"  - VRT 文件: {self.vrt_dir}")
        print(f"{'='*60}\n")

        failed_tiles = []

        for i, tile in enumerate(tqdm(self.tiles, desc="Overall progress")):
            self.stats['total'] += 1

            cities = ', '.join(tile['cities'][:3])
            if len(tile['cities']) > 3:
                cities += f" (+{len(tile['cities'])-3} more)"

            print(f"\n[{i+1}/{len(self.tiles)}] {tile['year']} {tile['utm_zone']} - {cities}")

            success = False
            for attempt in range(max_retries):
                tiff_success, vrt_success = self.download_tile(tile)

                if tiff_success and (vrt_success or not download_vrt):
                    success = True
                    break

                if attempt < max_retries - 1:
                    print(f"  Retry {attempt + 1}/{max_retries}...")
                    time.sleep(2)

            if not success:
                failed_tiles.append(tile)

        # 打印统计信息
        self.print_summary(failed_tiles)

    def print_summary(self, failed_tiles):
        """打印下载摘要"""
        print(f"\n{'='*60}")
        print("下载完成！")
        print(f"{'='*60}")
        print(f"总计: {self.stats['total']} 个瓦片")
        print(f"已下载: {self.stats['downloaded']} 个文件")
        print(f"跳过（已存在）: {self.stats['skipped']} 个文件")
        print(f"失败: {self.stats['failed']} 个文件")

        if failed_tiles:
            print(f"\n失败的瓦片列表：")
            for tile in failed_tiles:
                print(f"  - {tile['year']} {tile['utm_zone']} {tile['hash']}")

            # 保存失败列表
            failed_file = self.output_dir / "failed_missing_tiles.json"
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_tiles, f, ensure_ascii=False, indent=2)
            print(f"\n失败列表已保存到: {failed_file}")

    def download_by_year(self, year, download_vrt=True):
        """只下载指定年份的瓦片"""
        year_tiles = [t for t in self.tiles if t['year'] == year]
        print(f"找到 {len(year_tiles)} 个 {year} 年的瓦片")

        original_tiles = self.tiles
        self.tiles = year_tiles
        self.download_all(download_vrt)
        self.tiles = original_tiles

    def download_by_zone(self, zone, download_vrt=True):
        """只下载指定UTM Zone的瓦片"""
        zone_tiles = [t for t in self.tiles if t['utm_zone'] == zone]
        print(f"找到 {len(zone_tiles)} 个 {zone} 区的瓦片")

        original_tiles = self.tiles
        self.tiles = zone_tiles
        self.download_all(download_vrt)
        self.tiles = original_tiles


def main():
    parser = argparse.ArgumentParser(description='批量下载缺失的 AEF 卫星数据瓦片')
    parser.add_argument('--config', default=None,
                        help='配置文件路径 (default: missing_tiles_to_download.json)')
    parser.add_argument('--output', default=None,
                        help='输出目录 (default: china_city_data_1211/downloads)')
    parser.add_argument('--year', type=int, default=None,
                        help='只下载指定年份的数据')
    parser.add_argument('--zone', type=str, default=None,
                        help='只下载指定UTM Zone的数据 (如 50N)')
    parser.add_argument('--no-vrt', action='store_true',
                        help='不下载 VRT 文件')
    parser.add_argument('--retries', type=int, default=3,
                        help='失败重试次数 (default: 3)')

    args = parser.parse_args()

    downloader = MissingTileDownloader(args.config, args.output)

    if args.year:
        downloader.download_by_year(args.year, download_vrt=not args.no_vrt)
    elif args.zone:
        downloader.download_by_zone(args.zone, download_vrt=not args.no_vrt)
    else:
        downloader.download_all(download_vrt=not args.no_vrt, max_retries=args.retries)


if __name__ == "__main__":
    main()
