"""
查找缺失城市的 AEF 卫星数据瓦片
基于已有的 download_list.json 中的城市坐标，从 AEF 索引中查找对应的瓦片

用法:
    python step1_find_missing_tiles.py
"""

import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path


def load_aef_index(csv_path):
    """加载 AEF 索引数据"""
    colnames = [
        "wgs84_geom", "utm_crs", "s3_url", "year", "utm_zone",
        "utm_xmin", "utm_ymin", "utm_xmax", "utm_ymax",
        "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"
    ]
    print(f"Loading AEF index from {csv_path}...")
    df = pd.read_csv(csv_path, header=None, names=colnames)
    print(f"Loaded {len(df)} tiles from index")
    return df


def extract_hash(s3_url):
    """从 s3_url 中提取 hash"""
    filename = s3_url.split('/')[-1]
    hash_part = filename.split('-')[0]
    return hash_part


def find_tiles_for_coordinate(df, lon, lat, year=None):
    """查找覆盖指定坐标的瓦片"""
    mask = (
        (df["wgs84_west"] <= lon) &
        (df["wgs84_east"] >= lon) &
        (df["wgs84_south"] <= lat) &
        (df["wgs84_north"] >= lat)
    )
    if year is not None:
        mask = mask & (df["year"] == year)

    tiles = df[mask].copy()
    return tiles


def main():
    # 配置路径
    base_dir = Path("/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data")

    # AEF 索引文件
    aef_index_path = base_dir / "aef_index.csv"

    # 缺失城市列表（从之前生成的文件读取）
    missing_list_path = base_dir / "data" / "pretrain_city_tiles" / "download_list.json"

    # 输出路径
    output_dir = base_dir / "data" / "pretrain_city_tiles"
    output_json = output_dir / "missing_tiles_to_download.json"
    output_csv = output_dir / "missing_tiles_to_download.csv"

    # 加载缺失城市列表
    print(f"Loading missing cities list from {missing_list_path}...")
    with open(missing_list_path, 'r', encoding='utf-8') as f:
        missing_data = json.load(f)

    print(f"Found {len(missing_data['tiles'])} missing tile locations")

    # 加载 AEF 索引
    df = load_aef_index(aef_index_path)

    # 存储找到的瓦片
    all_tiles = {}
    not_found = []

    print(f"\n开始查找覆盖缺失城市的瓦片...")

    for tile_info in tqdm(missing_data['tiles'], desc="Processing missing locations"):
        lon = tile_info['center_lon']
        lat = tile_info['center_lat']
        cities = tile_info['cities']
        years = tile_info['years']

        for year in years:
            # 查找覆盖该坐标的瓦片
            tiles = find_tiles_for_coordinate(df, lon, lat, year)

            if len(tiles) == 0:
                not_found.append({
                    'cities': cities,
                    'lon': lon,
                    'lat': lat,
                    'year': year
                })
                continue

            # 提取 hash 并记录
            tiles['hash'] = tiles['s3_url'].apply(extract_hash)

            for _, row in tiles.iterrows():
                tile_key = f"{row['year']}_{row['utm_zone']}_{row['hash']}"

                if tile_key not in all_tiles:
                    all_tiles[tile_key] = {
                        'year': int(row['year']),
                        'utm_zone': row['utm_zone'],
                        'hash': row['hash'],
                        's3_url': row['s3_url'],
                        'cities': []
                    }

                # 添加城市到瓦片记录
                for city in cities:
                    if city not in all_tiles[tile_key]['cities']:
                        all_tiles[tile_key]['cities'].append(city)

    # 转换为列表格式
    tiles_list = list(all_tiles.values())

    # 保存为 JSON
    output_data = {
        'total_tiles': len(tiles_list),
        'total_missing_locations': len(missing_data['tiles']),
        'not_found_in_index': not_found,
        'tiles': tiles_list
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 保存为 CSV
    csv_data = []
    for tile in tiles_list:
        csv_data.append({
            'year': tile['year'],
            'utm_zone': tile['utm_zone'],
            'hash': tile['hash'],
            'cities_count': len(tile['cities']),
            'cities': '; '.join(tile['cities'][:5]) + ('...' if len(tile['cities']) > 5 else ''),
            's3_url': tile['s3_url']
        })

    df_output = pd.DataFrame(csv_data)
    df_output.to_csv(output_csv, index=False, encoding='utf-8-sig')

    # 统计信息
    print(f"\n{'='*60}")
    print(f"查找完成！")
    print(f"{'='*60}")
    print(f"总共找到 {len(tiles_list)} 个需要下载的瓦片")
    print(f"未在索引中找到: {len(not_found)} 条记录")

    if not_found:
        print(f"\n未找到的记录示例 (前10条):")
        for item in not_found[:10]:
            cities_str = ', '.join(item['cities'][:2])
            print(f"  - {item['year']} ({item['lon']}, {item['lat']}) - {cities_str}")

    # 按年份统计
    print(f"\n按年份统计：")
    year_stats = {}
    for tile in tiles_list:
        year = tile['year']
        year_stats[year] = year_stats.get(year, 0) + 1

    for year in sorted(year_stats.keys()):
        print(f"  {year}: {year_stats[year]} 个瓦片")

    # 按 UTM Zone 统计
    print(f"\n按 UTM Zone 统计：")
    zone_stats = {}
    for tile in tiles_list:
        zone = tile['utm_zone']
        zone_stats[zone] = zone_stats.get(zone, 0) + 1

    for zone in sorted(zone_stats.keys()):
        print(f"  {zone}: {zone_stats[zone]} 个瓦片")

    # 估算数据量
    avg_tiff_size_mb = 45  # 基于之前提取的文件大小
    estimated_total_gb = len(tiles_list) * avg_tiff_size_mb / 1024
    print(f"\n预估下载数据量: ~{estimated_total_gb:.1f} GB")

    print(f"\n结果已保存到：")
    print(f"  - {output_json}")
    print(f"  - {output_csv}")

    return output_json


if __name__ == "__main__":
    main()
