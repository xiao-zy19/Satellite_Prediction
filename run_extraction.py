"""
非交互式预训练数据提取脚本
直接执行完整切分，无需确认
"""

import sys
sys.path.insert(0, '/home/xiaozhenyu/degree_essay/Alpha_Earth/AEF_Data/Baseline_Pretrain')

from pretrain_data_extractor import PretrainDataExtractor, ExtractionConfig

if __name__ == "__main__":
    # 配置
    config = ExtractionConfig()
    config.overwrite = False  # 不覆盖已存在的文件

    print("=" * 60)
    print("预训练数据提取 - 自动执行")
    print("=" * 60)
    print(f"  TIFF目录: {config.tiff_dir}")
    print(f"  城市坐标: {config.cities_json}")
    print(f"  输出目录: {config.output_dir}")
    print(f"  切片大小: {config.tile_size_meters}米 x {config.tile_size_meters}米")
    print(f"  处理年份: {config.years}")
    print("=" * 60)

    # 创建提取器
    extractor = PretrainDataExtractor(config)

    print(f"\n开始提取 {len(extractor.cities)} 个城市, {len(config.years)} 个年份")
    print(f"预计任务数: {len(extractor.cities) * len(config.years)}")
    print("=" * 60)

    # 执行提取
    results = extractor.extract_all_cities()

    print("\n提取完成!")
