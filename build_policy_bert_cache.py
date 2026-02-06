"""
Build BERT policy embedding cache for training.

This script pre-computes BERT embeddings for all province-year combinations,
storing them in a cache file for efficient training.

Usage:
    # Build cache from policy_full_texts_collected.json (recommended)
    python build_policy_bert_cache.py --from-collected

    # Build cache from fertility_policies.json (structured format)
    python build_policy_bert_cache.py --from-structured

    # Build cache from custom JSON file (auto-detect format)
    python build_policy_bert_cache.py --input policy_texts.json --output policy_bert_cache

    # Build cache with custom BERT model
    python build_policy_bert_cache.py --from-collected --model hfl/chinese-macbert-base

    # Build cache with specific output dimension
    python build_policy_bert_cache.py --from-collected --dim 128

Supported JSON formats:
    1. policy_full_texts_collected.json format (--from-collected):
       {
           "national_policies": {"N001": {"full_text": "...", "release_date": "2021-01-01"}, ...},
           "provincial_policies": {"北京市": {"regulation_revision_2021": {...}}, ...},
           "municipal_policies": {"育儿补贴政策": {"SC_PZH001": {...}}, ...}
       }

    2. fertility_policies.json format (--from-structured):
       {
           "national_policies": [{...}, ...],
           "provincial_policies": {"华北地区": {"北京市": {"regulation_revisions": [...]}}}
       }

    3. Generic format (auto-detected with --input):
       - Nested: {"省份": {"年份": {"documents": [...]}}}
       - Flat: {"policies": [{"province": "...", "year": ..., "full_text": "..."}]}
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Union

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy_bert import (
    PolicyBertConfig,
    PolicyBertEncoder,
    MultiDocumentPolicyEncoder,
    PolicyEmbeddingCache,
    KeyArticleExtractor,
    load_policy_texts_from_json
)
from policy_features import CITY_TO_PROVINCE


def load_policy_texts_from_structured(
    json_path: str,
    include_fields: List[str] = None
) -> Dict[Tuple[str, int], List[str]]:
    """
    Load policy texts from the structured fertility_policies.json format.

    This extracts text from main_content fields and other descriptive fields,
    combining them into a single text per province-year.

    Args:
        json_path: Path to fertility_policies.json
        include_fields: List of fields to include in text extraction

    Returns:
        Dict mapping (province, year) to list of text strings
    """
    include_fields = include_fields or [
        'main_content', 'policy_name', 'key_changes', 'note'
    ]

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = {}

    # Extract national policies (as baseline)
    national_texts = {}
    for policy in data.get('national_policies', []):
        year = int(policy.get('release_date', '2020')[:4])
        text_parts = []
        for field in include_fields:
            if field in policy and policy[field]:
                text_parts.append(str(policy[field]))
        if text_parts:
            if year not in national_texts:
                national_texts[year] = []
            national_texts[year].append(' '.join(text_parts))

    # Extract provincial policies
    provincial_data = data.get('provincial_policies', {})

    for region_name, region_data in provincial_data.items():
        for province_name, province_data in region_data.items():
            # Collect texts by year
            year_texts = {}  # year -> list of texts

            # Process regulation revisions
            for reg in province_data.get('regulation_revisions', []):
                year = int(reg.get('release_date', '2020')[:4])
                text_parts = []
                for field in include_fields:
                    if field in reg and reg[field]:
                        text_parts.append(str(reg[field]))
                if text_parts:
                    if year not in year_texts:
                        year_texts[year] = []
                    year_texts[year].append(' '.join(text_parts))

            # Process economic support
            for item in province_data.get('economic_support', []):
                year = int(item.get('release_date', item.get('effective_date', '2020'))[:4])
                text_parts = []
                for field in include_fields:
                    if field in item and item[field]:
                        text_parts.append(str(item[field]))
                if text_parts:
                    if year not in year_texts:
                        year_texts[year] = []
                    year_texts[year].append(' '.join(text_parts))

            # Process housing support
            housing = province_data.get('housing_support', [])
            if isinstance(housing, dict):
                housing = [housing]
            for item in housing:
                if isinstance(item, dict):
                    year = int(item.get('release_date', '2020')[:4])
                    text_parts = []
                    for field in include_fields:
                        if field in item and item[field]:
                            text_parts.append(str(item[field]))
                    if text_parts:
                        if year not in year_texts:
                            year_texts[year] = []
                        year_texts[year].append(' '.join(text_parts))

            # Process medical insurance policies
            for item in province_data.get('medical_insurance_policies', []):
                year = int(item.get('release_date', item.get('effective_date', '2020'))[:4])
                text_parts = []
                for field in include_fields:
                    if field in item and item[field]:
                        text_parts.append(str(item[field]))
                if text_parts:
                    if year not in year_texts:
                        year_texts[year] = []
                    year_texts[year].append(' '.join(text_parts))

            # Process city policies
            for item in province_data.get('city_policies', []):
                year = int(item.get('release_date', item.get('effective_date', '2020'))[:4])
                text_parts = []
                for field in include_fields:
                    if field in item and item[field]:
                        text_parts.append(str(item[field]))
                if text_parts:
                    if year not in year_texts:
                        year_texts[year] = []
                    year_texts[year].append(' '.join(text_parts))

            # Add leave standards as descriptive text
            leave = province_data.get('current_leave_standards', {})
            if leave:
                leave_text = f"假期政策：产假{leave.get('maternity_leave_days', '未知')}天，" \
                            f"陪产假{leave.get('paternity_leave_days', '未知')}天，" \
                            f"育儿假{leave.get('parental_leave_days_per_year', '未知')}天/年，" \
                            f"婚假{leave.get('marriage_leave_days', '未知')}天。"
                if leave.get('note'):
                    leave_text += f" {leave['note']}"

                # Add to recent years (2021-2024)
                for year in range(2021, 2025):
                    if year not in year_texts:
                        year_texts[year] = []
                    year_texts[year].append(leave_text)

            # Store results
            for year, texts in year_texts.items():
                if texts:
                    # Add national policy context for this year
                    nat_context = []
                    for nat_year in sorted(national_texts.keys()):
                        if nat_year <= year:
                            nat_context = national_texts[nat_year]
                    if nat_context:
                        texts = nat_context + texts

                    result[(province_name, year)] = texts

    # Fill gaps: propagate policies to subsequent years
    provinces = set(k[0] for k in result.keys())
    for province in provinces:
        years_with_data = sorted([k[1] for k in result.keys() if k[0] == province])
        if not years_with_data:
            continue

        for year in range(min(years_with_data), 2025):
            if (province, year) not in result:
                # Find most recent year with data
                prev_years = [y for y in years_with_data if y < year]
                if prev_years:
                    result[(province, year)] = result[(province, max(prev_years))]

    return result


def load_policy_texts_from_collected_json(
    json_path: str,
    include_national: bool = True,
    include_municipal: bool = True,
    propagate_to_years: bool = True
) -> Dict[Tuple[str, int], List[str]]:
    """
    Load policy texts from policy_full_texts_collected.json format.

    This JSON file has a specific structure:
    - national_policies: dict[policy_id, policy_data] with "full_text" field
    - provincial_policies: dict[province, dict[regulation_key, policy_data]]
    - municipal_policies: dict[category, dict[policy_id, policy_data]]

    Args:
        json_path: Path to policy_full_texts_collected.json
        include_national: Whether to include national policies as context
        include_municipal: Whether to include municipal/city-level policies
        propagate_to_years: Whether to propagate policies to subsequent years

    Returns:
        Dict mapping (province/city, year) to list of text strings
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = {}
    text_fields = ['full_text', 'full_text_summary', 'main_content', 'content']

    def extract_text(policy_data: dict) -> str:
        """Extract text from policy data using multiple possible field names."""
        for field in text_fields:
            if field in policy_data and policy_data[field]:
                text = str(policy_data[field]).strip()
                if text:
                    return text
        return ""

    def extract_year(policy_data: dict, key_name: str = "") -> int:
        """Extract year from policy data or key name."""
        # First try to extract from key name (e.g., regulation_revision_2021)
        if key_name:
            match = re.search(r'(\d{4})', key_name)
            if match:
                return int(match.group(1))

        # Then try release_date or effective_date
        for date_field in ['release_date', 'effective_date', 'date']:
            if date_field in policy_data and policy_data[date_field]:
                date_str = str(policy_data[date_field])
                match = re.search(r'(\d{4})', date_str)
                if match:
                    return int(match.group(1))

        return None

    # ==========================================================================
    # 1. Process national_policies (dict format: {policy_id: policy_data})
    # ==========================================================================
    national_texts_by_year = {}  # year -> list of texts

    national_data = data.get('national_policies', {})
    if isinstance(national_data, dict):
        # policy_full_texts_collected.json format: dict indexed by policy_id
        for policy_id, policy in national_data.items():
            if not isinstance(policy, dict):
                continue

            year = extract_year(policy, policy_id)
            if not year:
                continue

            text = extract_text(policy)
            if text:
                if year not in national_texts_by_year:
                    national_texts_by_year[year] = []
                national_texts_by_year[year].append(text)

                # Also store as separate entries for "国家政策"
                if include_national:
                    key = ('国家政策', year)
                    if key not in result:
                        result[key] = []
                    result[key].append(text)

    elif isinstance(national_data, list):
        # fertility_policies.json format: list of policies
        for policy in national_data:
            if not isinstance(policy, dict):
                continue

            year = extract_year(policy)
            if not year:
                continue

            text = extract_text(policy)
            if text:
                if year not in national_texts_by_year:
                    national_texts_by_year[year] = []
                national_texts_by_year[year].append(text)

                if include_national:
                    key = ('国家政策', year)
                    if key not in result:
                        result[key] = []
                    result[key].append(text)

    print(f"  Loaded {sum(len(v) for v in national_texts_by_year.values())} national policy texts")

    # ==========================================================================
    # 2. Process provincial_policies
    # ==========================================================================
    provincial_data = data.get('provincial_policies', {})
    provincial_count = 0

    for province, province_policies in provincial_data.items():
        if not isinstance(province_policies, dict):
            continue

        # Skip if this is a region grouping (like "华北地区")
        # Check if the values are province data or policy data
        first_value = next(iter(province_policies.values()), None)
        if isinstance(first_value, dict) and 'regulation_revisions' in first_value:
            # This is fertility_policies.json format with region grouping
            # Skip - this will be handled by load_policy_texts_from_structured
            continue

        # This is policy_full_texts_collected.json format
        for policy_key, policy in province_policies.items():
            if not isinstance(policy, dict):
                continue

            year = extract_year(policy, policy_key)
            if not year:
                continue

            text = extract_text(policy)
            if text:
                key = (province, year)
                if key not in result:
                    result[key] = []
                result[key].append(text)
                provincial_count += 1

    print(f"  Loaded {provincial_count} provincial policy texts")

    # ==========================================================================
    # 3. Process municipal_policies (nested by category)
    # ==========================================================================
    municipal_count = 0

    if include_municipal:
        municipal_data = data.get('municipal_policies', {})

        for category, policies in municipal_data.items():
            if not isinstance(policies, dict):
                continue

            for policy_id, policy in policies.items():
                if not isinstance(policy, dict):
                    continue

                # Get city name
                city = policy.get('city', '')
                if not city:
                    # Try to extract from policy_id (e.g., "BJ_GJJ001" -> might need mapping)
                    continue

                year = extract_year(policy, policy_id)
                if not year:
                    continue

                text = extract_text(policy)
                if text:
                    # Map city to province if possible
                    province = CITY_TO_PROVINCE.get(city, city)

                    key = (province, year)
                    if key not in result:
                        result[key] = []
                    result[key].append(text)
                    municipal_count += 1

    print(f"  Loaded {municipal_count} municipal policy texts")

    # ==========================================================================
    # 4. Add national policy context to provincial entries
    # ==========================================================================
    if include_national and national_texts_by_year:
        for (province, year), texts in list(result.items()):
            if province == '国家政策':
                continue

            # Find applicable national policies (from years <= current year)
            national_context = []
            for nat_year in sorted(national_texts_by_year.keys()):
                if nat_year <= year:
                    national_context = national_texts_by_year[nat_year]

            # Prepend national context (but don't duplicate if already there)
            if national_context:
                # Add only the most recent applicable national policy as context
                result[(province, year)] = national_context[-1:] + texts

    # ==========================================================================
    # 5. Propagate policies to subsequent years (fill gaps)
    # ==========================================================================
    if propagate_to_years:
        provinces = set(k[0] for k in result.keys() if k[0] != '国家政策')
        target_years = range(2014, 2026)  # Cover 2014-2025

        for province in provinces:
            years_with_data = sorted([k[1] for k in result.keys() if k[0] == province])
            if not years_with_data:
                continue

            for year in target_years:
                if (province, year) not in result:
                    # Find most recent year with data
                    prev_years = [y for y in years_with_data if y < year]
                    if prev_years:
                        result[(province, year)] = result[(province, max(prev_years))].copy()

    print(f"  Total entries after propagation: {len(result)}")

    return result


def detect_json_format(json_path: str) -> str:
    """
    Auto-detect the format of a policy JSON file.

    Returns:
        'collected': policy_full_texts_collected.json format
        'structured': fertility_policies.json format
        'generic': generic format for load_policy_texts_from_json
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check for policy_full_texts_collected.json format
    # - national_policies is a dict (indexed by policy_id like "N001")
    # - has "municipal_policies" key
    if isinstance(data.get('national_policies'), dict):
        national_keys = list(data.get('national_policies', {}).keys())
        if national_keys and re.match(r'^[A-Z]\d+', national_keys[0]):
            return 'collected'

    # Check for fertility_policies.json format
    # - national_policies is a list
    # - provincial_policies has region grouping (like "华北地区")
    if isinstance(data.get('national_policies'), list):
        provincial = data.get('provincial_policies', {})
        if provincial:
            first_key = list(provincial.keys())[0]
            first_val = provincial.get(first_key, {})
            if isinstance(first_val, dict):
                # Check if first value contains province names with regulation_revisions
                inner_first = next(iter(first_val.values()), {})
                if isinstance(inner_first, dict) and 'regulation_revisions' in inner_first:
                    return 'structured'

    # Check for generic format
    if 'policies' in data:
        return 'generic'

    # Default: try collected format first (most feature-rich)
    if 'provincial_policies' in data:
        return 'collected'

    return 'generic'


def build_cache_from_texts(
    policy_texts: Dict[Tuple[str, int], Union[str, List[str]]],
    output_dir: str,
    config: PolicyBertConfig,
    device: str = "cuda"
):
    """
    Build BERT embedding cache from policy texts.

    Args:
        policy_texts: Dict mapping (province, year) to text or list of texts
        output_dir: Output directory for cache
        config: BERT encoder configuration
        device: Device to run BERT on
    """
    print(f"\nBuilding BERT cache...")
    print(f"  Encoder: {config.model_name}")
    print(f"  Output dim: {config.output_dim}")
    print(f"  Chunk strategy: {config.chunk_strategy}")
    print(f"  Total entries: {len(policy_texts)}")

    # Create encoder
    print(f"\nLoading BERT model...")
    encoder = PolicyBertEncoder(config)

    # Wrap with multi-document aggregation
    multi_doc_encoder = MultiDocumentPolicyEncoder(
        encoder,
        aggregation="attention"
    )
    multi_doc_encoder = multi_doc_encoder.to(device)
    multi_doc_encoder.eval()

    # Create cache
    cache = PolicyEmbeddingCache(
        cache_dir=output_dir,
        embedding_dim=config.output_dim,
        policy_lag=1
    )

    # Build cache
    cache.build(
        policy_texts=policy_texts,
        encoder=multi_doc_encoder,
        device=device
    )

    # Print statistics
    stats = cache.get_statistics()
    print(f"\nCache statistics:")
    print(f"  Entries: {stats['count']}")
    print(f"  Provinces: {stats['provinces']}")
    print(f"  Year range: {stats['year_range']}")
    print(f"  Mean embedding norm: {stats['mean_norm']:.4f}")

    return cache


def main():
    parser = argparse.ArgumentParser(
        description="Build BERT policy embedding cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build from policy_full_texts_collected.json (recommended, contains full texts)
  python build_policy_bert_cache.py --from-collected

  # Build from fertility_policies.json (structured format with summaries)
  python build_policy_bert_cache.py --from-structured

  # Build from custom JSON file with auto-detection
  python build_policy_bert_cache.py --input /path/to/policy.json

  # Specify output directory and model
  python build_policy_bert_cache.py --from-collected --output my_cache --model hfl/chinese-macbert-base
        """
    )

    # Input source options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--from-collected', action='store_true',
                             help='Build from policy_full_texts_collected.json (recommended)')
    input_group.add_argument('--from-structured', action='store_true',
                             help='Build from fertility_policies.json (structured format)')
    input_group.add_argument('--input', type=str, default=None,
                             help='Custom input JSON file (auto-detect format)')

    # Output and model options
    parser.add_argument('--output', type=str, default='policy_bert_cache',
                        help='Output directory for cache (default: policy_bert_cache)')
    parser.add_argument('--model', type=str, default='hfl/chinese-roberta-wwm-ext',
                        help='BERT model name (default: hfl/chinese-roberta-wwm-ext)')
    parser.add_argument('--dim', type=int, default=64,
                        help='Output embedding dimension (default: 64)')
    parser.add_argument('--chunk-strategy', type=str, default='hierarchical',
                        choices=['mean', 'max', 'hierarchical', 'attention'],
                        help='Chunk aggregation strategy (default: hierarchical)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu (default: cuda)')

    # Additional options
    parser.add_argument('--no-national', action='store_true',
                        help='Do not include national policies as context')
    parser.add_argument('--no-municipal', action='store_true',
                        help='Do not include municipal/city policies')
    parser.add_argument('--no-propagate', action='store_true',
                        help='Do not propagate policies to subsequent years')

    args = parser.parse_args()

    # ==========================================================================
    # Determine input source and load policy texts
    # ==========================================================================
    policy_texts = None
    base_dir = Path(__file__).parent.parent / "data_local" / "Fertility_Policy"

    if args.from_collected:
        # Use policy_full_texts_collected.json
        input_path = base_dir / "policy_full_texts_collected.json"
        if not input_path.exists():
            print(f"Error: {input_path} not found")
            print("Please ensure policy_full_texts_collected.json exists in data_local/Fertility_Policy/")
            sys.exit(1)

        print(f"Loading from collected format: {input_path}")
        print("=" * 60)
        policy_texts = load_policy_texts_from_collected_json(
            str(input_path),
            include_national=not args.no_national,
            include_municipal=not args.no_municipal,
            propagate_to_years=not args.no_propagate
        )

    elif args.from_structured:
        # Use fertility_policies.json
        input_path = base_dir / "fertility_policies.json"
        if not input_path.exists():
            print(f"Error: {input_path} not found")
            sys.exit(1)

        print(f"Loading from structured format: {input_path}")
        print("=" * 60)
        policy_texts = load_policy_texts_from_structured(str(input_path))

    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: {input_path} not found")
            sys.exit(1)

        # Auto-detect format
        print(f"Loading from: {input_path}")
        detected_format = detect_json_format(str(input_path))
        print(f"Auto-detected format: {detected_format}")
        print("=" * 60)

        if detected_format == 'collected':
            policy_texts = load_policy_texts_from_collected_json(
                str(input_path),
                include_national=not args.no_national,
                include_municipal=not args.no_municipal,
                propagate_to_years=not args.no_propagate
            )
        elif detected_format == 'structured':
            policy_texts = load_policy_texts_from_structured(str(input_path))
        else:
            policy_texts = load_policy_texts_from_json(str(input_path))

    else:
        # Default: try to use policy_full_texts_collected.json
        default_path = base_dir / "policy_full_texts_collected.json"
        if default_path.exists():
            print(f"No input specified, using default: {default_path}")
            print("=" * 60)
            policy_texts = load_policy_texts_from_collected_json(
                str(default_path),
                include_national=not args.no_national,
                include_municipal=not args.no_municipal,
                propagate_to_years=not args.no_propagate
            )
        else:
            print("Error: Please specify --from-collected, --from-structured, or --input")
            print(f"Or create the default file: {default_path}")
            parser.print_help()
            sys.exit(1)

    # ==========================================================================
    # Validate results
    # ==========================================================================
    if not policy_texts:
        print("\nError: No policy texts found!")
        print("Please check your input JSON file format.")
        sys.exit(1)

    print(f"\nLoaded {len(policy_texts)} province-year entries")

    # Show statistics
    provinces = set(k[0] for k in policy_texts.keys())
    years = sorted(set(k[1] for k in policy_texts.keys()))
    total_texts = sum(len(v) for v in policy_texts.values())

    print(f"  Provinces/Cities: {len(provinces)}")
    print(f"  Year range: {min(years)} - {max(years)}")
    print(f"  Total text segments: {total_texts}")

    # Show sample entries
    print("\nSample entries:")
    for i, (key, texts) in enumerate(list(policy_texts.items())[:5]):
        province, year = key
        text_preview = texts[0][:80] if texts else "N/A"
        print(f"  ({province}, {year}): {len(texts)} texts, preview: {text_preview}...")

    # ==========================================================================
    # Create BERT config and build cache
    # ==========================================================================
    config = PolicyBertConfig(
        model_name=args.model,
        output_dim=args.dim,
        chunk_strategy=args.chunk_strategy,
        freeze_bert=True
    )

    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("\nWarning: CUDA not available, using CPU (this will be slow)")
        device = "cpu"

    # Build cache
    cache = build_cache_from_texts(
        policy_texts=policy_texts,
        output_dir=args.output,
        config=config,
        device=device
    )

    # ==========================================================================
    # Print summary and usage instructions
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Cache build complete!")
    print("=" * 60)
    print(f"Cache saved to: {args.output}/")
    print(f"  - policy_embeddings_dim{args.dim}.pkl")

    print("\nTo use this cache for training:")
    print(f"  python train_multimodal_bert.py --exp bert_cnn_concat --cache-dir {args.output}")

    print("\nCache can be loaded in code with:")
    print(f"""
    from policy_bert import PolicyEmbeddingCache
    cache = PolicyEmbeddingCache(cache_dir='{args.output}', embedding_dim={args.dim})
    cache.load()
    embedding = cache.get_for_city('北京市', 2022)  # Returns numpy array of shape ({args.dim},)
    """)


if __name__ == "__main__":
    main()
