"""
预训练数据提取脚本

功能：
1. 读取中国所有333个地级市坐标数据
2. 从Alpha Earth TIFF文件中提取每个城市中心10km x 10km区域
3. 支持2018-2024年所有年份的完整数据提取
4. 用于模型预训练的完整数据集准备

切分规格：
- 切片尺寸: 10km x 10km (1000 x 1000 pixels @ 10m/pixel)
- 波段数: 64 (Alpha Earth embedding维度)
- 年份: 2018-2024
- 城市数: 333个地级市

作者: Auto-generated for pretrain data preparation
日期: 2024-12
"""

import os
import json
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """提取配置"""
    # 数据路径
    tiff_dir: str = "/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/data/china_city_data_1211/downloads/tiff_files"
    cities_json: str = "/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/data/china_city_data_1211/config/china_prefecture_cities.json"
    output_dir: str = "/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/data/pretrain_city_tiles"

    # 切片参数
    tile_size_meters: int = 10000  # 10km x 10km
    pixel_size: int = 10  # 10米/像素

    # 年份范围
    years: List[int] = None

    # 处理参数
    max_workers: int = 4
    overwrite: bool = False  # 是否覆盖已存在的文件

    def __post_init__(self):
        if self.years is None:
            self.years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]


class PretrainDataExtractor:
    """预训练数据提取器 - 用于提取中国所有地级市的卫星影像"""

    def __init__(self, config: ExtractionConfig):
        """
        初始化提取器

        Args:
            config: 提取配置
        """
        self.config = config
        self.tiff_dir = Path(config.tiff_dir)
        self.cities_json = Path(config.cities_json)
        self.output_dir = Path(config.output_dir)

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载城市数据
        self.cities = self._load_cities()
        logger.info(f"加载了 {len(self.cities)} 个城市")

        # 构建TIFF文件索引
        self.tiff_index = self._build_tiff_index()

    def _load_cities(self) -> List[Dict]:
        """加载城市坐标数据"""
        with open(self.cities_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['cities']

    def _get_utm_zone(self, lon: float) -> int:
        """根据经度计算UTM区号"""
        return int((lon + 180) / 6) + 1

    def _get_utm_epsg(self, lon: float, lat: float) -> int:
        """获取UTM投影的EPSG代码"""
        zone = self._get_utm_zone(lon)
        # 北半球使用 326xx，南半球使用 327xx
        if lat >= 0:
            return 32600 + zone
        else:
            return 32700 + zone

    def _build_tiff_index(self) -> Dict:
        """
        构建TIFF文件索引

        返回格式: {year: {utm_zone: [tiff_info, ...]}}
        """
        logger.info("构建TIFF文件索引...")
        index = {}

        tiff_files = list(self.tiff_dir.glob("*.tiff"))
        logger.info(f"发现 {len(tiff_files)} 个TIFF文件")

        for tiff_path in tqdm(tiff_files, desc="索引TIFF文件"):
            # 解析文件名: {year}_{zone}N_{tile_id}-{offset_x}-{offset_y}.tiff
            filename = tiff_path.stem
            parts = filename.split('_')
            if len(parts) < 2:
                continue

            try:
                year = int(parts[0])
                zone_str = parts[1]  # 如 "50N"
                zone = int(zone_str[:-1])  # 去掉N/S

                # 读取TIFF边界信息
                with rasterio.open(tiff_path) as src:
                    bounds = src.bounds
                    crs = src.crs

                # 构建索引
                if year not in index:
                    index[year] = {}
                if zone not in index[year]:
                    index[year][zone] = []

                index[year][zone].append({
                    'path': str(tiff_path),
                    'bounds': bounds,
                    'crs': crs,
                    'filename': filename
                })
            except (ValueError, IndexError) as e:
                logger.debug(f"跳过文件: {filename}, 原因: {e}")
                continue

        # 统计信息
        total_tiles = 0
        for year in sorted(index.keys()):
            zones = list(index[year].keys())
            year_tiles = sum(len(v) for v in index[year].values())
            total_tiles += year_tiles
            logger.info(f"  {year}年: {len(zones)}个UTM区, 共{year_tiles}个瓦片")

        logger.info(f"总计: {total_tiles}个TIFF瓦片已索引")
        return index

    def _wgs84_to_utm(self, lon: float, lat: float, target_epsg: int) -> Tuple[float, float]:
        """将WGS84坐标转换为UTM坐标"""
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
        x, y = transformer.transform(lon, lat)
        return x, y

    def _find_containing_tiff(self, x: float, y: float, year: int, zone: int) -> Optional[Dict]:
        """找到包含指定坐标的TIFF文件"""
        if year not in self.tiff_index:
            return None
        if zone not in self.tiff_index[year]:
            return None

        for tiff_info in self.tiff_index[year][zone]:
            bounds = tiff_info['bounds']
            # 检查点是否在边界内
            min_y = min(bounds.bottom, bounds.top)
            max_y = max(bounds.bottom, bounds.top)
            if (bounds.left <= x <= bounds.right and
                min_y <= y <= max_y):
                return tiff_info

        return None

    def _extract_tile(self,
                      tiff_path: str,
                      center_x: float,
                      center_y: float,
                      output_path: str) -> Tuple[bool, str]:
        """
        从TIFF中提取指定区域

        Args:
            tiff_path: TIFF文件路径
            center_x: 中心点X坐标（UTM）
            center_y: 中心点Y坐标（UTM）
            output_path: 输出文件路径

        Returns:
            (是否成功, 错误信息或成功信息)
        """
        try:
            with rasterio.open(tiff_path) as src:
                # 计算窗口边界
                half_size = self.config.tile_size_meters / 2

                # 计算像素坐标
                transform = src.transform
                resolution = abs(src.res[0])  # 使用绝对值确保正确

                # 计算中心点在图像中的像素位置
                col_center = (center_x - transform.c) / transform.a
                row_center = (center_y - transform.f) / transform.e

                # 计算窗口大小（像素）
                window_size_pixels = int(self.config.tile_size_meters / resolution)
                half_window = window_size_pixels // 2

                # 计算窗口起始位置
                col_start = int(col_center - half_window)
                row_start = int(row_center - half_window)

                # 边界检查和调整
                if col_start < 0:
                    col_start = 0
                if row_start < 0:
                    row_start = 0

                # 确保不超出图像边界
                col_end = min(col_start + window_size_pixels, src.width)
                row_end = min(row_start + window_size_pixels, src.height)

                actual_width = col_end - col_start
                actual_height = row_end - row_start

                if actual_width <= 0 or actual_height <= 0:
                    return False, f"窗口大小无效: width={actual_width}, height={actual_height}"

                # 如果实际大小小于期望大小，记录警告但仍然继续
                if actual_width < window_size_pixels or actual_height < window_size_pixels:
                    logger.debug(f"窗口被裁剪: {actual_width}x{actual_height} (期望 {window_size_pixels}x{window_size_pixels})")

                # 创建窗口
                window = Window(col_start, row_start, actual_width, actual_height)

                # 读取数据
                data = src.read(window=window)

                # 如果尺寸不足，进行padding
                if actual_width < window_size_pixels or actual_height < window_size_pixels:
                    padded_data = np.zeros((src.count, window_size_pixels, window_size_pixels), dtype=data.dtype)
                    padded_data[:, :actual_height, :actual_width] = data
                    data = padded_data
                    actual_width = window_size_pixels
                    actual_height = window_size_pixels

                # 计算新的transform
                new_transform = rasterio.windows.transform(window, src.transform)

                # 保存输出
                output_profile = src.profile.copy()
                output_profile.update({
                    'width': actual_width,
                    'height': actual_height,
                    'transform': new_transform,
                    'compress': 'lzw'
                })

                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with rasterio.open(output_path, 'w', **output_profile) as dst:
                    dst.write(data)

                return True, f"成功提取: {actual_width}x{actual_height}像素"

        except Exception as e:
            return False, str(e)

    def extract_city(self, city: Dict, year: int) -> Dict:
        """
        提取单个城市的卫星影像

        Args:
            city: 城市信息字典
            year: 年份

        Returns:
            提取结果字典
        """
        city_name = city['name']
        lon = city['lon']
        lat = city['lat']
        province = city.get('province', 'unknown')

        result = {
            'city': city_name,
            'province': province,
            'year': year,
            'lon': lon,
            'lat': lat,
            'success': False,
            'output_path': None,
            'error': None,
            'utm_zone': None,
            'utm_x': None,
            'utm_y': None
        }

        try:
            # 确定UTM区
            zone = self._get_utm_zone(lon)
            epsg = self._get_utm_epsg(lon, lat)

            result['utm_zone'] = zone

            # 转换坐标
            x, y = self._wgs84_to_utm(lon, lat, epsg)
            result['utm_x'] = x
            result['utm_y'] = y

            # 构建输出路径
            output_path = self.output_dir / province / city_name / f"{year}_{city_name}.tiff"

            # 检查是否已存在
            if output_path.exists() and not self.config.overwrite:
                result['success'] = True
                result['output_path'] = str(output_path)
                result['error'] = "已存在，跳过"
                return result

            # 查找包含该坐标的TIFF
            tiff_info = self._find_containing_tiff(x, y, year, zone)

            if tiff_info is None:
                result['error'] = f"未找到包含坐标({lon:.4f}, {lat:.4f})的TIFF文件 (UTM Zone {zone})"
                return result

            # 提取瓦片
            success, message = self._extract_tile(
                tiff_info['path'],
                x, y,
                str(output_path)
            )

            if success:
                result['success'] = True
                result['output_path'] = str(output_path)
            else:
                result['error'] = message

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"处理城市 {city_name} ({year}年) 失败: {e}")

        return result

    def extract_all_cities(self) -> List[Dict]:
        """
        提取所有城市的所有年份数据

        Returns:
            所有提取结果的列表
        """
        all_results = []

        # 准备任务列表
        tasks = []
        for year in self.config.years:
            if year not in self.tiff_index:
                logger.warning(f"年份 {year} 没有可用的TIFF数据")
                continue
            for city in self.cities:
                tasks.append((city, year))

        logger.info(f"共有 {len(tasks)} 个任务待处理")
        logger.info(f"城市数: {len(self.cities)}, 年份: {self.config.years}")

        # 统计变量
        success_count = 0
        skip_count = 0
        fail_count = 0

        # 执行提取
        with tqdm(total=len(tasks), desc="提取城市影像") as pbar:
            for city, year in tasks:
                result = self.extract_city(city, year)
                all_results.append(result)

                if result['success']:
                    if result['error'] == "已存在，跳过":
                        skip_count += 1
                        status = "⏭"
                    else:
                        success_count += 1
                        status = "✓"
                else:
                    fail_count += 1
                    status = "✗"

                pbar.set_postfix_str(f"{city['name']} ({year}) {status}")
                pbar.update(1)

        # 统计结果
        logger.info(f"\n{'='*60}")
        logger.info(f"提取完成统计:")
        logger.info(f"  新提取成功: {success_count}")
        logger.info(f"  已存在跳过: {skip_count}")
        logger.info(f"  提取失败: {fail_count}")
        logger.info(f"  总计: {len(all_results)}")
        logger.info(f"{'='*60}")

        # 保存报告
        self._save_report(all_results, success_count, skip_count, fail_count)

        return all_results

    def _save_report(self, results: List[Dict], success: int, skip: int, fail: int):
        """保存提取报告"""
        report_path = self.output_dir / "extraction_report.json"

        # 按省份和城市组织统计
        province_stats = {}
        for r in results:
            province = r['province']
            if province not in province_stats:
                province_stats[province] = {'success': 0, 'failed': 0, 'cities': set()}
            if r['success']:
                province_stats[province]['success'] += 1
            else:
                province_stats[province]['failed'] += 1
            province_stats[province]['cities'].add(r['city'])

        # 转换set为list用于JSON序列化
        for p in province_stats:
            province_stats[p]['cities'] = len(province_stats[p]['cities'])

        # 统计失败原因
        error_stats = {}
        for r in results:
            if not r['success'] and r['error']:
                error_type = r['error'].split(':')[0] if ':' in r['error'] else r['error'][:50]
                error_stats[error_type] = error_stats.get(error_type, 0) + 1

        report = {
            'summary': {
                'total_tasks': len(results),
                'new_success': success,
                'skipped_existing': skip,
                'failed': fail,
                'tile_size_meters': self.config.tile_size_meters,
                'years': self.config.years,
                'total_cities': len(self.cities)
            },
            'province_stats': province_stats,
            'error_stats': error_stats,
            'results': results
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"报告已保存至: {report_path}")

        # 同时保存简化版报告（便于快速查看）
        summary_path = self.output_dir / "extraction_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("预训练数据提取报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"配置:\n")
            f.write(f"  切片尺寸: {self.config.tile_size_meters}m x {self.config.tile_size_meters}m\n")
            f.write(f"  年份范围: {self.config.years}\n")
            f.write(f"  城市总数: {len(self.cities)}\n\n")
            f.write(f"结果统计:\n")
            f.write(f"  新提取成功: {success}\n")
            f.write(f"  已存在跳过: {skip}\n")
            f.write(f"  提取失败: {fail}\n")
            f.write(f"  总计任务: {len(results)}\n\n")
            f.write("各省份统计:\n")
            for p, stats in sorted(province_stats.items()):
                f.write(f"  {p}: {stats['cities']}个城市, 成功{stats['success']}, 失败{stats['failed']}\n")
            f.write("\n失败原因统计:\n")
            for error, count in sorted(error_stats.items(), key=lambda x: -x[1]):
                f.write(f"  {error}: {count}次\n")

        logger.info(f"摘要已保存至: {summary_path}")

    def extract_single_city_test(self, city_name: str = "北京市", year: int = 2018) -> Dict:
        """
        测试提取单个城市

        Args:
            city_name: 城市名称
            year: 年份

        Returns:
            提取结果
        """
        # 查找城市
        city = None
        for c in self.cities:
            if c['name'] == city_name:
                city = c
                break

        if city is None:
            logger.error(f"未找到城市: {city_name}")
            return {'success': False, 'error': f"未找到城市: {city_name}"}

        logger.info(f"测试提取: {city_name} ({year}年)")
        logger.info(f"  坐标: ({city['lon']}, {city['lat']})")

        result = self.extract_city(city, year)

        if result['success']:
            logger.info(f"✓ 提取成功: {result['output_path']}")
        else:
            logger.error(f"✗ 提取失败: {result['error']}")

        return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='预训练数据提取工具')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'test', 'years'],
                       help='提取模式: all=所有城市所有年份, test=测试单个城市, years=指定年份')
    parser.add_argument('--years', type=int, nargs='+', default=None,
                       help='指定要提取的年份 (如: --years 2018 2019 2020)')
    parser.add_argument('--test-city', type=str, default='北京市',
                       help='测试模式下要提取的城市名')
    parser.add_argument('--test-year', type=int, default=2018,
                       help='测试模式下的年份')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='自定义输出目录')
    parser.add_argument('--overwrite', action='store_true',
                       help='覆盖已存在的文件')

    args = parser.parse_args()

    # 配置
    config = ExtractionConfig()
    if args.years:
        config.years = args.years
    if args.output_dir:
        config.output_dir = args.output_dir
    config.overwrite = args.overwrite

    print("=" * 60)
    print("预训练数据提取工具")
    print("=" * 60)
    print(f"  TIFF目录: {config.tiff_dir}")
    print(f"  城市坐标: {config.cities_json}")
    print(f"  输出目录: {config.output_dir}")
    print(f"  切片大小: {config.tile_size_meters}米 x {config.tile_size_meters}米")
    print(f"  处理年份: {config.years}")
    print(f"  覆盖模式: {config.overwrite}")
    print("=" * 60)

    # 创建提取器
    extractor = PretrainDataExtractor(config)

    if args.mode == 'test':
        # 测试单个城市
        result = extractor.extract_single_city_test(args.test_city, args.test_year)

    elif args.mode in ['all', 'years']:
        # 提取所有城市
        print(f"\n即将提取 {len(extractor.cities)} 个城市, {len(config.years)} 个年份")
        print(f"预计任务数: {len(extractor.cities) * len(config.years)}")

        confirm = input("\n确认开始提取? (y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消")
            return

        results = extractor.extract_all_cities()

    print("\n处理完成!")


if __name__ == "__main__":
    main()
