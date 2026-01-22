"""
验证脚本：检查 multimodal 模块中政策数据的正确性

Step 1: 检查 JSONL/JSON 文件中的省份-年份数据完整性
Step 2: 验证城市到省份的映射覆盖率
Step 3: 验证 PolicyFeatureExtractor 解析的特征是否与原始数据一致
Step 4: 检查特定省份-年份对的数据正确性
Step 5: 检查时间滞后(lag)机制是否正确应用
"""

import json
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from policy_features import (
    PolicyFeatureExtractor,
    CITY_TO_PROVINCE,
    MINORITY_REGIONS,
    get_policy_features
)


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def step1_check_data_completeness():
    """Step 1: 检查原始数据文件的完整性"""
    print_header("Step 1: 检查原始数据文件完整性")

    base_dir = Path(__file__).parent.parent / "data_local" / "Fertility_Policy"
    jsonl_path = base_dir / "fertility_policies_by_province_year.jsonl"
    json_path = base_dir / "fertility_policies_by_province_year.json"

    issues = []

    # Check JSONL file
    if jsonl_path.exists():
        print(f"\n[JSONL] 文件路径: {jsonl_path}")
        df = pd.read_json(jsonl_path, lines=True)
        print(f"  - 总记录数: {len(df)}")
        print(f"  - 省份数量: {df['province'].nunique()}")
        print(f"  - 年份范围: {df['year'].min()} - {df['year'].max()}")

        # Check for expected provinces (31 provinces)
        expected_provinces = 31
        actual_provinces = df['province'].nunique()
        if actual_provinces != expected_provinces:
            issues.append(f"省份数量不符: 期望 {expected_provinces}, 实际 {actual_provinces}")

        # Check for expected years (2013-2024 = 12 years)
        expected_years = list(range(2013, 2025))
        actual_years = sorted(df['year'].unique())
        if actual_years != expected_years:
            issues.append(f"年份范围不符: 期望 {expected_years}, 实际 {actual_years}")

        # Check completeness: each province should have all years
        print(f"\n  检查每个省份的年份完整性:")
        incomplete_provinces = []
        for province in df['province'].unique():
            province_years = sorted(df[df['province'] == province]['year'].tolist())
            if province_years != expected_years:
                missing = set(expected_years) - set(province_years)
                incomplete_provinces.append((province, missing))

        if incomplete_provinces:
            for prov, missing in incomplete_provinces:
                print(f"    [ERROR] {prov}: 缺少年份 {missing}")
                issues.append(f"{prov} 缺少年份 {missing}")
        else:
            print(f"    [OK] 所有 {actual_provinces} 个省份都有完整的 {len(expected_years)} 年数据")

        # List all provinces
        print(f"\n  省份列表:")
        for i, prov in enumerate(sorted(df['province'].unique()), 1):
            print(f"    {i:2d}. {prov}")

        # Check required fields
        required_fields = [
            'province', 'year', 'policy_phase',
            'maternity_leave_days', 'paternity_leave_days',
            'parental_leave_days', 'marriage_leave_days',
            'birth_subsidy_amount', 'housing_subsidy_amount',
            'childcare_subsidy_monthly', 'tax_deduction_monthly',
            'has_childcare_policy', 'has_ivf_insurance', 'is_minority_favorable'
        ]

        print(f"\n  检查必需字段:")
        missing_fields = [f for f in required_fields if f not in df.columns]
        if missing_fields:
            print(f"    [ERROR] 缺少字段: {missing_fields}")
            issues.append(f"缺少字段: {missing_fields}")
        else:
            print(f"    [OK] 所有 {len(required_fields)} 个必需字段都存在")

        return df, issues

    # Fallback to JSON
    elif json_path.exists():
        print(f"\n[JSON] 文件路径: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        policies = data.get('policies', {})
        print(f"  - 省份数量: {len(policies)}")

        # Count records
        total_records = sum(len(years) for years in policies.values())
        print(f"  - 总记录数: {total_records}")

        return None, issues

    else:
        issues.append("政策数据文件不存在!")
        print(f"  [ERROR] 政策数据文件不存在!")
        return None, issues


def step2_check_city_mapping():
    """Step 2: 验证城市到省份的映射"""
    print_header("Step 2: 验证城市到省份的映射")

    issues = []

    # Load satellite data cities
    from dataset import find_satellite_data
    sat_data = find_satellite_data()
    satellite_cities = set(sat_data.keys())

    print(f"\n卫星数据中的城市数量: {len(satellite_cities)}")

    # Check mapping coverage
    mapped_cities = set(CITY_TO_PROVINCE.keys())
    print(f"映射表中的城市数量: {len(mapped_cities)}")

    # Find unmapped cities
    unmapped = satellite_cities - mapped_cities
    if unmapped:
        print(f"\n  [WARNING] 未映射的城市 ({len(unmapped)} 个):")
        for city in sorted(unmapped):
            print(f"    - {city}")
            # Check if city name matches province name directly
            if city.endswith('市') or city.endswith('省'):
                print(f"      (可能直接使用城市名作为省份名)")
        issues.append(f"{len(unmapped)} 个城市未在映射表中")
    else:
        print(f"\n  [OK] 所有卫星数据城市都有映射")

    # Check province coverage in mapping
    provinces_in_mapping = set(CITY_TO_PROVINCE.values())
    print(f"\n映射表涵盖的省份数量: {len(provinces_in_mapping)}")

    # Check minority regions
    minority_in_mapping = provinces_in_mapping & MINORITY_REGIONS
    print(f"\n少数民族地区检查:")
    print(f"  定义的少数民族地区: {MINORITY_REGIONS}")
    print(f"  映射中的少数民族地区: {minority_in_mapping}")

    return issues


def step3_check_feature_extraction():
    """Step 3: 验证 PolicyFeatureExtractor 的特征提取"""
    print_header("Step 3: 验证 PolicyFeatureExtractor 特征提取")

    issues = []

    # Load raw data
    base_dir = Path(__file__).parent.parent / "data_local" / "Fertility_Policy"
    jsonl_path = base_dir / "fertility_policies_by_province_year.jsonl"

    if not jsonl_path.exists():
        print("  [ERROR] JSONL 文件不存在，跳过此步骤")
        return issues

    df = pd.read_json(jsonl_path, lines=True)

    # Create extractor with no lag for direct comparison
    extractor = PolicyFeatureExtractor(policy_lag=0)

    print(f"\n比较原始数据与提取器输出:")
    print(f"  测试样本数: 10 (每个省份随机选取)")

    # Sample some records
    test_records = df.sample(min(10, len(df)), random_state=42)

    mismatches = []
    for _, row in test_records.iterrows():
        province = row['province']
        year = int(row['year'])

        # Get features from extractor (省份名直接传入)
        extracted = extractor.get_features(province, year, apply_lag=False)

        # Build expected features from raw data
        expected = np.array([
            float(row['maternity_leave_days']),
            float(row['paternity_leave_days']),
            float(row['parental_leave_days']),
            float(row['marriage_leave_days']),
            float(row['birth_subsidy_amount']),
            float(row['housing_subsidy_amount']),
            float(row['childcare_subsidy_monthly']),
            float(row['tax_deduction_monthly']),
            1.0 if row['has_childcare_policy'] else 0.0,
            1.0 if row['has_ivf_insurance'] else 0.0,
            1.0 if row['is_minority_favorable'] else 0.0,
            float(row['policy_phase'])
        ], dtype=np.float32)

        # Compare
        if not np.allclose(extracted, expected, atol=1e-5):
            mismatches.append({
                'province': province,
                'year': year,
                'extracted': extracted,
                'expected': expected,
                'diff': extracted - expected
            })
            print(f"\n  [MISMATCH] {province} {year}:")
            print(f"    提取值: {extracted}")
            print(f"    期望值: {expected}")
            print(f"    差异:   {extracted - expected}")
        else:
            print(f"  [OK] {province} {year}")

    if mismatches:
        issues.append(f"{len(mismatches)} 个记录的特征提取不匹配")
    else:
        print(f"\n  [OK] 所有测试样本的特征提取正确")

    return issues


def step4_check_specific_cases():
    """Step 4: 检查特定省份-年份对的数据正确性"""
    print_header("Step 4: 检查特定省份-年份对的数据")

    issues = []

    # Define test cases with known expected values
    test_cases = [
        # (城市, 年份, 期望的关键特征)
        {
            'city': '北京市',
            'year': 2021,
            'expected': {
                'maternity_leave_days': 158.0,  # 2021年改革后
                'policy_phase': 2.0,  # 三孩政策
                'tax_deduction_monthly': 0.0,  # 2022年才开始
            }
        },
        {
            'city': '北京市',
            'year': 2022,
            'expected': {
                'maternity_leave_days': 158.0,
                'policy_phase': 2.0,
                'tax_deduction_monthly': 1000.0,  # 2022年开始1000元
            }
        },
        {
            'city': '北京市',
            'year': 2023,
            'expected': {
                'tax_deduction_monthly': 2000.0,  # 2023年提高到2000元
            }
        },
        {
            'city': '上海市',
            'year': 2020,
            'expected': {
                'maternity_leave_days': 128.0,  # 改革前
                'parental_leave_days': 0.0,  # 改革前无育儿假
                'policy_phase': 1.0,  # 全面二孩
            }
        },
        {
            'city': '上海市',
            'year': 2021,
            'expected': {
                'maternity_leave_days': 158.0,  # 改革后
                'parental_leave_days': 5.0,  # 改革后有育儿假
                'policy_phase': 2.0,  # 三孩政策
            }
        },
        {
            'city': '呼和浩特市',
            'year': 2022,
            'expected': {
                'is_minority_favorable': 1.0,  # 内蒙古是少数民族地区
            }
        },
        {
            'city': '拉萨市',
            'year': 2022,
            'expected': {
                'is_minority_favorable': 1.0,  # 西藏是少数民族地区
            }
        },
        {
            'city': '成都市',
            'year': 2022,
            'expected': {
                'is_minority_favorable': 0.0,  # 四川不是少数民族自治区
            }
        },
    ]

    extractor = PolicyFeatureExtractor(policy_lag=0)
    feature_names = extractor.feature_names

    print(f"\n测试 {len(test_cases)} 个关键案例:")

    for case in test_cases:
        city = case['city']
        year = case['year']
        expected = case['expected']

        # Get features
        features = extractor.get_features(city, year, apply_lag=False)

        print(f"\n  {city} {year}:")
        all_match = True
        for feat_name, expected_val in expected.items():
            feat_idx = feature_names.index(feat_name)
            actual_val = features[feat_idx]
            match = abs(actual_val - expected_val) < 1e-5

            status = "[OK]" if match else "[FAIL]"
            print(f"    {status} {feat_name}: {actual_val:.1f} (期望: {expected_val:.1f})")

            if not match:
                all_match = False
                issues.append(f"{city} {year} {feat_name}: 实际 {actual_val} != 期望 {expected_val}")

    return issues


def step5_check_temporal_lag():
    """Step 5: 检查时间滞后机制"""
    print_header("Step 5: 检查时间滞后(lag)机制")

    issues = []

    # Create extractor with lag=1
    extractor_lag1 = PolicyFeatureExtractor(policy_lag=1)
    extractor_no_lag = PolicyFeatureExtractor(policy_lag=0)

    print(f"\n验证 lag=1 时使用前一年的政策数据:")

    test_cases = [
        ('北京市', 2022),  # 应该使用2021年政策
        ('上海市', 2022),  # 应该使用2021年政策
        ('北京市', 2021),  # 应该使用2020年政策（改革前后差异大）
    ]

    for city, year in test_cases:
        # With lag
        features_with_lag = extractor_lag1.get_features(city, year, apply_lag=True)

        # Without lag - get previous year
        features_prev_year = extractor_no_lag.get_features(city, year - 1, apply_lag=False)

        # Compare
        match = np.allclose(features_with_lag, features_prev_year, atol=1e-5)

        status = "[OK]" if match else "[FAIL]"
        print(f"\n  {status} {city} {year} (with lag=1) vs {city} {year-1} (no lag):")

        if not match:
            print(f"    With lag:   {features_with_lag}")
            print(f"    Prev year:  {features_prev_year}")
            print(f"    Diff:       {features_with_lag - features_prev_year}")
            issues.append(f"时间滞后机制错误: {city} {year}")
        else:
            # Show key features that prove the lag is working
            feature_names = extractor_lag1.feature_names
            mat_idx = feature_names.index('maternity_leave_days')
            tax_idx = feature_names.index('tax_deduction_monthly')
            phase_idx = feature_names.index('policy_phase')

            print(f"    maternity_leave: {features_with_lag[mat_idx]:.0f}")
            print(f"    tax_deduction:   {features_with_lag[tax_idx]:.0f}")
            print(f"    policy_phase:    {features_with_lag[phase_idx]:.0f}")

    # Special case: 2021 with lag should use 2020 (pre-reform) data
    print(f"\n  关键验证: 预测2021年结果时应使用2020年(改革前)政策")

    features_2021_with_lag = extractor_lag1.get_features('北京市', 2021, apply_lag=True)
    features_2020_direct = extractor_no_lag.get_features('北京市', 2020, apply_lag=False)

    feature_names = extractor_lag1.feature_names
    mat_idx = feature_names.index('maternity_leave_days')

    # 2020年应该是128天（改革前），2021年应该是158天（改革后）
    mat_2021_with_lag = features_2021_with_lag[mat_idx]
    mat_2020_direct = features_2020_direct[mat_idx]

    if abs(mat_2021_with_lag - 128.0) < 1e-5 and abs(mat_2020_direct - 128.0) < 1e-5:
        print(f"    [OK] 预测2021年使用2020年政策: 产假={mat_2021_with_lag:.0f}天 (改革前标准)")
    else:
        print(f"    [FAIL] 产假天数不符: 实际={mat_2021_with_lag:.0f}, 期望=128")
        issues.append("时间滞后机制在改革年份边界处理错误")

    return issues


def step6_check_dataset_integration():
    """Step 6: 检查数据集集成"""
    print_header("Step 6: 检查数据集集成 (dataset_policy.py)")

    issues = []

    try:
        from dataset import load_population_data, find_satellite_data, create_dataset_samples

        print("\n加载数据集样本...")
        pop_df = load_population_data()
        sat_data = find_satellite_data()
        samples = create_dataset_samples(pop_df, sat_data)

        print(f"  总样本数: {len(samples)}")

        # Check sample structure
        if samples:
            sample = samples[0]
            print(f"\n  样本结构:")
            for key in sample:
                print(f"    - {key}: {type(sample[key]).__name__}")

            # Verify city names are in mapping
            unique_cities = set(s['city'] for s in samples)
            print(f"\n  数据集中的唯一城市数: {len(unique_cities)}")

            unmapped = unique_cities - set(CITY_TO_PROVINCE.keys())
            if unmapped:
                print(f"  [WARNING] 未映射城市: {unmapped}")
            else:
                print(f"  [OK] 所有城市都有映射")

            # Check year range
            years = set(s['year'] for s in samples)
            print(f"  年份范围: {min(years)} - {max(years)}")

        # Test policy feature extraction for some samples
        print(f"\n测试样本的政策特征提取:")
        extractor = PolicyFeatureExtractor(policy_lag=1)

        test_samples = samples[:5] if len(samples) >= 5 else samples
        for sample in test_samples:
            city = sample['city']
            year = sample['year']

            try:
                features = extractor.get_normalized_features(city, year)
                print(f"  [OK] {city} {year}: 特征维度={features.shape}, 范围=[{features.min():.3f}, {features.max():.3f}]")
            except Exception as e:
                print(f"  [FAIL] {city} {year}: {e}")
                issues.append(f"样本特征提取失败: {city} {year}")

    except Exception as e:
        print(f"  [ERROR] 数据集加载失败: {e}")
        issues.append(f"数据集加载失败: {e}")

    return issues


def main():
    print("\n" + "=" * 70)
    print("  Multimodal 政策数据验证脚本")
    print("=" * 70)

    all_issues = []

    # Step 1
    df, issues = step1_check_data_completeness()
    all_issues.extend(issues)

    # Step 2
    issues = step2_check_city_mapping()
    all_issues.extend(issues)

    # Step 3
    issues = step3_check_feature_extraction()
    all_issues.extend(issues)

    # Step 4
    issues = step4_check_specific_cases()
    all_issues.extend(issues)

    # Step 5
    issues = step5_check_temporal_lag()
    all_issues.extend(issues)

    # Step 6
    issues = step6_check_dataset_integration()
    all_issues.extend(issues)

    # Summary
    print_header("验证总结")

    if all_issues:
        print(f"\n发现 {len(all_issues)} 个问题:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n[SUCCESS] 所有验证通过！政策数据正确无误。")

    return len(all_issues)


if __name__ == "__main__":
    sys.exit(main())
