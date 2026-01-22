"""
Structured policy feature extraction for fertility policies.

This module extracts structured numerical features from fertility policy data
for each province and year, enabling multimodal prediction with satellite imagery.

Feature vector dimensions:
- maternity_leave_days: float (normalized)
- paternity_leave_days: float (normalized)
- parental_leave_days: float (normalized)
- marriage_leave_days: float (normalized)
- birth_subsidy_amount: float (normalized, 万元/年)
- housing_subsidy_amount: float (normalized, 万元)
- childcare_subsidy_monthly: float (normalized, 元/月)
- tax_deduction_monthly: float (normalized, 元/月)
- has_childcare_policy: float (0 or 1)
- has_ivf_insurance: float (0 or 1)
- is_minority_favorable: float (0 or 1)
- policy_phase_indicator: float (0-3, representing policy phase)

Total: 12 dimensions
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re


# Policy phase encoding
POLICY_PHASES = {
    "单独二孩": 0,
    "全面二孩": 1,
    "三孩政策": 2,
    "生育友好型社会": 3
}


@dataclass
class PolicyFeatures:
    """Structured policy features for a province-year."""
    maternity_leave_days: float = 158.0  # default national standard
    paternity_leave_days: float = 15.0
    parental_leave_days: float = 10.0
    marriage_leave_days: float = 10.0
    birth_subsidy_amount: float = 0.0  # 万元/年
    housing_subsidy_amount: float = 0.0  # 万元
    childcare_subsidy_monthly: float = 0.0  # 元/月
    tax_deduction_monthly: float = 0.0  # 元/月
    has_childcare_policy: float = 0.0
    has_ivf_insurance: float = 0.0
    is_minority_favorable: float = 0.0
    policy_phase: float = 1.0  # default to 全面二孩

    def to_vector(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.maternity_leave_days,
            self.paternity_leave_days,
            self.parental_leave_days,
            self.marriage_leave_days,
            self.birth_subsidy_amount,
            self.housing_subsidy_amount,
            self.childcare_subsidy_monthly,
            self.tax_deduction_monthly,
            self.has_childcare_policy,
            self.has_ivf_insurance,
            self.is_minority_favorable,
            self.policy_phase
        ], dtype=np.float32)

    @property
    def dim(self) -> int:
        return 12


# City to province mapping (based on common Chinese city naming)
# Cities ending with 市 in the satellite data need to be mapped to provinces
CITY_TO_PROVINCE = {
    # 华北地区
    "北京市": "北京市",
    "天津市": "天津市",
    "石家庄市": "河北省", "唐山市": "河北省", "秦皇岛市": "河北省", "邯郸市": "河北省",
    "邢台市": "河北省", "保定市": "河北省", "张家口市": "河北省", "承德市": "河北省",
    "沧州市": "河北省", "廊坊市": "河北省", "衡水市": "河北省",
    "太原市": "山西省", "大同市": "山西省", "阳泉市": "山西省", "长治市": "山西省",
    "晋城市": "山西省", "朔州市": "山西省", "晋中市": "山西省", "运城市": "山西省",
    "忻州市": "山西省", "临汾市": "山西省", "吕梁市": "山西省",
    "呼和浩特市": "内蒙古自治区", "包头市": "内蒙古自治区", "乌海市": "内蒙古自治区",
    "赤峰市": "内蒙古自治区", "通辽市": "内蒙古自治区", "鄂尔多斯市": "内蒙古自治区",
    "呼伦贝尔市": "内蒙古自治区", "巴彦淖尔市": "内蒙古自治区", "乌兰察布市": "内蒙古自治区",
    # 东北地区
    "沈阳市": "辽宁省", "大连市": "辽宁省", "鞍山市": "辽宁省", "抚顺市": "辽宁省",
    "本溪市": "辽宁省", "丹东市": "辽宁省", "锦州市": "辽宁省", "营口市": "辽宁省",
    "阜新市": "辽宁省", "辽阳市": "辽宁省", "盘锦市": "辽宁省", "铁岭市": "辽宁省",
    "朝阳市": "辽宁省", "葫芦岛市": "辽宁省",
    "长春市": "吉林省", "吉林市": "吉林省", "四平市": "吉林省", "辽源市": "吉林省",
    "通化市": "吉林省", "白山市": "吉林省", "松原市": "吉林省", "白城市": "吉林省",
    "延边朝鲜族自治州": "吉林省",
    "哈尔滨市": "黑龙江省", "齐齐哈尔市": "黑龙江省", "鸡西市": "黑龙江省",
    "鹤岗市": "黑龙江省", "双鸭山市": "黑龙江省", "大庆市": "黑龙江省",
    "伊春市": "黑龙江省", "佳木斯市": "黑龙江省", "七台河市": "黑龙江省",
    "牡丹江市": "黑龙江省", "黑河市": "黑龙江省", "绥化市": "黑龙江省",
    # 华东地区
    "上海市": "上海市",
    "南京市": "江苏省", "无锡市": "江苏省", "徐州市": "江苏省", "常州市": "江苏省",
    "苏州市": "江苏省", "南通市": "江苏省", "连云港市": "江苏省", "淮安市": "江苏省",
    "盐城市": "江苏省", "扬州市": "江苏省", "镇江市": "江苏省", "泰州市": "江苏省",
    "宿迁市": "江苏省",
    "杭州市": "浙江省", "宁波市": "浙江省", "温州市": "浙江省", "嘉兴市": "浙江省",
    "湖州市": "浙江省", "绍兴市": "浙江省", "金华市": "浙江省", "衢州市": "浙江省",
    "舟山市": "浙江省", "台州市": "浙江省", "丽水市": "浙江省",
    "合肥市": "安徽省", "芜湖市": "安徽省", "蚌埠市": "安徽省", "淮南市": "安徽省",
    "马鞍山市": "安徽省", "淮北市": "安徽省", "铜陵市": "安徽省", "安庆市": "安徽省",
    "黄山市": "安徽省", "滁州市": "安徽省", "阜阳市": "安徽省", "宿州市": "安徽省",
    "六安市": "安徽省", "亳州市": "安徽省", "池州市": "安徽省", "宣城市": "安徽省",
    "福州市": "福建省", "厦门市": "福建省", "莆田市": "福建省", "三明市": "福建省",
    "泉州市": "福建省", "漳州市": "福建省", "南平市": "福建省", "龙岩市": "福建省",
    "宁德市": "福建省",
    "南昌市": "江西省", "景德镇市": "江西省", "萍乡市": "江西省", "九江市": "江西省",
    "新余市": "江西省", "鹰潭市": "江西省", "赣州市": "江西省", "吉安市": "江西省",
    "宜春市": "江西省", "抚州市": "江西省", "上饶市": "江西省",
    "济南市": "山东省", "青岛市": "山东省", "淄博市": "山东省", "枣庄市": "山东省",
    "东营市": "山东省", "烟台市": "山东省", "潍坊市": "山东省", "济宁市": "山东省",
    "泰安市": "山东省", "威海市": "山东省", "日照市": "山东省", "临沂市": "山东省",
    "德州市": "山东省", "聊城市": "山东省", "滨州市": "山东省", "菏泽市": "山东省",
    # 华中地区
    "郑州市": "河南省", "开封市": "河南省", "洛阳市": "河南省", "平顶山市": "河南省",
    "安阳市": "河南省", "鹤壁市": "河南省", "新乡市": "河南省", "焦作市": "河南省",
    "濮阳市": "河南省", "许昌市": "河南省", "漯河市": "河南省", "三门峡市": "河南省",
    "南阳市": "河南省", "商丘市": "河南省", "信阳市": "河南省", "周口市": "河南省",
    "驻马店市": "河南省",
    "武汉市": "湖北省", "黄石市": "湖北省", "十堰市": "湖北省", "宜昌市": "湖北省",
    "襄阳市": "湖北省", "鄂州市": "湖北省", "荆门市": "湖北省", "孝感市": "湖北省",
    "荆州市": "湖北省", "黄冈市": "湖北省", "咸宁市": "湖北省", "随州市": "湖北省",
    "恩施土家族苗族自治州": "湖北省",
    "长沙市": "湖南省", "株洲市": "湖南省", "湘潭市": "湖南省", "衡阳市": "湖南省",
    "邵阳市": "湖南省", "岳阳市": "湖南省", "常德市": "湖南省", "张家界市": "湖南省",
    "益阳市": "湖南省", "郴州市": "湖南省", "永州市": "湖南省", "怀化市": "湖南省",
    "娄底市": "湖南省", "湘西土家族苗族自治州": "湖南省",
    # 华南地区
    "广州市": "广东省", "韶关市": "广东省", "深圳市": "广东省", "珠海市": "广东省",
    "汕头市": "广东省", "佛山市": "广东省", "江门市": "广东省", "湛江市": "广东省",
    "茂名市": "广东省", "肇庆市": "广东省", "惠州市": "广东省", "梅州市": "广东省",
    "汕尾市": "广东省", "河源市": "广东省", "阳江市": "广东省", "清远市": "广东省",
    "东莞市": "广东省", "中山市": "广东省", "潮州市": "广东省", "揭阳市": "广东省",
    "云浮市": "广东省",
    "南宁市": "广西壮族自治区", "柳州市": "广西壮族自治区", "桂林市": "广西壮族自治区",
    "梧州市": "广西壮族自治区", "北海市": "广西壮族自治区", "防城港市": "广西壮族自治区",
    "钦州市": "广西壮族自治区", "贵港市": "广西壮族自治区", "玉林市": "广西壮族自治区",
    "百色市": "广西壮族自治区", "贺州市": "广西壮族自治区", "河池市": "广西壮族自治区",
    "来宾市": "广西壮族自治区", "崇左市": "广西壮族自治区",
    "海口市": "海南省", "三亚市": "海南省", "三沙市": "海南省", "儋州市": "海南省",
    # 西南地区
    "重庆市": "重庆市",
    "成都市": "四川省", "自贡市": "四川省", "攀枝花市": "四川省", "泸州市": "四川省",
    "德阳市": "四川省", "绵阳市": "四川省", "广元市": "四川省", "遂宁市": "四川省",
    "内江市": "四川省", "乐山市": "四川省", "南充市": "四川省", "眉山市": "四川省",
    "宜宾市": "四川省", "广安市": "四川省", "达州市": "四川省", "雅安市": "四川省",
    "巴中市": "四川省", "资阳市": "四川省",
    "贵阳市": "贵州省", "六盘水市": "贵州省", "遵义市": "贵州省", "安顺市": "贵州省",
    "毕节市": "贵州省", "铜仁市": "贵州省",
    "昆明市": "云南省", "曲靖市": "云南省", "玉溪市": "云南省", "保山市": "云南省",
    "昭通市": "云南省", "丽江市": "云南省", "普洱市": "云南省", "临沧市": "云南省",
    "拉萨市": "西藏自治区", "日喀则市": "西藏自治区", "昌都市": "西藏自治区",
    "林芝市": "西藏自治区", "山南市": "西藏自治区", "那曲市": "西藏自治区",
    # 西北地区
    "西安市": "陕西省", "铜川市": "陕西省", "宝鸡市": "陕西省", "咸阳市": "陕西省",
    "渭南市": "陕西省", "延安市": "陕西省", "汉中市": "陕西省", "榆林市": "陕西省",
    "安康市": "陕西省", "商洛市": "陕西省",
    "兰州市": "甘肃省", "嘉峪关市": "甘肃省", "金昌市": "甘肃省", "白银市": "甘肃省",
    "天水市": "甘肃省", "武威市": "甘肃省", "张掖市": "甘肃省", "平凉市": "甘肃省",
    "酒泉市": "甘肃省", "庆阳市": "甘肃省", "定西市": "甘肃省", "陇南市": "甘肃省",
    "西宁市": "青海省", "海东市": "青海省",
    "银川市": "宁夏回族自治区", "石嘴山市": "宁夏回族自治区", "吴忠市": "宁夏回族自治区",
    "固原市": "宁夏回族自治区", "中卫市": "宁夏回族自治区",
    "乌鲁木齐市": "新疆维吾尔自治区", "克拉玛依市": "新疆维吾尔自治区",
    "吐鲁番市": "新疆维吾尔自治区", "哈密市": "新疆维吾尔自治区",
    # 新疆其他地区和自治州
    "伊犁哈萨克自治州": "新疆维吾尔自治区", "塔城地区": "新疆维吾尔自治区",
    "阿勒泰地区": "新疆维吾尔自治区", "博尔塔拉蒙古自治州": "新疆维吾尔自治区",
    "昌吉回族自治州": "新疆维吾尔自治区", "巴音郭楞蒙古自治州": "新疆维吾尔自治区",
    "阿克苏地区": "新疆维吾尔自治区", "克孜勒苏柯尔克孜自治州": "新疆维吾尔自治区",
    "喀什地区": "新疆维吾尔自治区", "和田地区": "新疆维吾尔自治区",
    # 四川自治州
    "阿坝藏族羌族自治州": "四川省", "甘孜藏族自治州": "四川省",
    "凉山彝族自治州": "四川省",
    # 云南自治州
    "楚雄彝族自治州": "云南省", "红河哈尼族彝族自治州": "云南省",
    "文山壮族苗族自治州": "云南省", "西双版纳傣族自治州": "云南省",
    "大理白族自治州": "云南省", "德宏傣族景颇族自治州": "云南省",
    "怒江傈僳族自治州": "云南省",
    # 甘肃自治州
    "临夏回族自治州": "甘肃省", "甘南藏族自治州": "甘肃省",
}

# Minority autonomous regions (entitled to more favorable policies)
MINORITY_REGIONS = {
    "内蒙古自治区", "广西壮族自治区", "西藏自治区",
    "宁夏回族自治区", "新疆维吾尔自治区"
}


def parse_leave_days(value) -> float:
    """Parse leave days from various formats."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Handle ranges like "158-180" -> take average
        if "-" in value and not value.startswith("不少于"):
            parts = value.replace("天", "").split("-")
            try:
                nums = [float(p) for p in parts if p.strip().isdigit()]
                return sum(nums) / len(nums) if nums else 0.0
            except:
                pass
        # Handle "不少于128天" -> take the number
        match = re.search(r'(\d+)', value)
        if match:
            return float(match.group(1))
        # Handle "≥15天"
        if "≥" in value:
            match = re.search(r'(\d+)', value)
            if match:
                return float(match.group(1))
    return 0.0


def parse_subsidy_amount(value) -> float:
    """Parse subsidy amount to 万元."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        # Assume it's in 元, convert to 万元
        if value > 1000:  # likely in 元
            return value / 10000.0
        return value
    if isinstance(value, str):
        match = re.search(r'(\d+(?:\.\d+)?)', value)
        if match:
            amount = float(match.group(1))
            if "万" in value:
                return amount
            return amount / 10000.0
    return 0.0


class PolicyFeatureExtractor:
    """Extract structured features from fertility policy JSON data.

    IMPORTANT: Policy features use temporal lag to prevent data leakage.
    When predicting population growth for year Y, we use policy features from year Y-lag.
    This reflects the causal delay: policy -> decision -> pregnancy -> birth -> statistics.
    Default lag is 1 year.

    Supports two JSON formats:
    1. New format (fertility_policies_by_province_year.json): Flat structure with pre-computed
       province-year features. Preferred format with explicit temporal data for each year.
    2. Old format (fertility_policies.json): Hierarchical structure requiring complex parsing.
       Maintained for backward compatibility.
    """

    # 2021年前的旧假期标准（国家基准）- only used for old format fallback
    OLD_LEAVE_STANDARDS = {
        'maternity_leave_days': 128.0,   # 98天法定 + ~30天地方奖励
        'paternity_leave_days': 15.0,    # 各地10-15天不等
        'parental_leave_days': 0.0,      # 2021年前大多数地区无育儿假
        'marriage_leave_days': 10.0,     # 3天法定 + 地方奖励
    }

    def __init__(self, json_path: Optional[str] = None, policy_lag: int = 1):
        """
        Initialize the extractor.

        Args:
            json_path: Path to fertility policy JSON file. If None, uses default path.
                       Automatically detects format (new province-year format or old format).
            policy_lag: Number of years to lag policy features (default: 1).
                        This prevents data leakage by using past policy to predict current outcomes.
        """
        if json_path is None:
            # Prefer JSONL format (fastest), then new JSON format, then old format
            base_dir = Path(__file__).parent.parent / "data_local" / "Fertility_Policy"
            jsonl_path = base_dir / "fertility_policies_by_province_year.jsonl"
            new_format_path = base_dir / "fertility_policies_by_province_year.json"
            old_format_path = base_dir / "fertility_policies.json"

            if jsonl_path.exists():
                json_path = jsonl_path
            elif new_format_path.exists():
                json_path = new_format_path
            else:
                json_path = old_format_path

        self.json_path = Path(json_path)
        self.policy_lag = policy_lag
        self.policy_data = None
        self.policy_df = None  # DataFrame for JSONL format (fast lookup)
        self.province_features = {}  # province -> year -> PolicyFeatures
        self.national_features = {}  # year -> PolicyFeatures (national baseline)
        self.province_reform_years = {}  # province -> year when new leave standards took effect
        self.is_new_format = False  # Flag to track which format is being used
        self.is_jsonl_format = False  # Flag for JSONL format

        self._load_and_parse()

    def _load_and_parse(self):
        """Load policy data and parse all policy features.

        Supports three formats:
        1. JSONL format (fertility_policies_by_province_year.jsonl) - fastest, recommended
        2. New JSON format (fertility_policies_by_province_year.json) - nested province-year
        3. Old JSON format (fertility_policies.json) - complex nested hierarchy
        """
        # Detect format by file extension
        if self.json_path.suffix == '.jsonl':
            self.is_jsonl_format = True
            self._parse_jsonl_format()
        else:
            # JSON format
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.policy_data = json.load(f)

            # Detect format: new format has "policies" key with province data directly
            if "policies" in self.policy_data and "metadata" in self.policy_data:
                self.is_new_format = True
                self._parse_new_format()
            else:
                # Old format has "provincial_policies" with region hierarchy
                self.is_new_format = False
                self._parse_national_policies()
                self._parse_provincial_policies()

    def _parse_jsonl_format(self):
        """Parse JSONL format (fertility_policies_by_province_year.jsonl).

        This is the recommended format for training:
        - One record per line, each line is a complete province-year entry
        - Flat structure with all features directly accessible
        - Smaller file size (47% compression vs nested JSON)
        - Builds dict cache for O(1) query performance
        """
        # Load JSONL into DataFrame
        self.policy_df = pd.read_json(self.json_path, lines=True)

        # Feature columns for extraction
        self._feature_columns = [
            'maternity_leave_days', 'paternity_leave_days', 'parental_leave_days',
            'marriage_leave_days', 'birth_subsidy_amount', 'housing_subsidy_amount',
            'childcare_subsidy_monthly', 'tax_deduction_monthly',
            'has_childcare_policy', 'has_ivf_insurance', 'is_minority_favorable',
            'policy_phase'
        ]

        # Parse national baseline
        self._parse_national_policies()

        # Build province_features dict for O(1) lookup (fastest query)
        for _, row in self.policy_df.iterrows():
            province = row['province']
            year = int(row['year'])

            if province not in self.province_features:
                self.province_features[province] = {}

            features = PolicyFeatures(
                maternity_leave_days=float(row['maternity_leave_days']),
                paternity_leave_days=float(row['paternity_leave_days']),
                parental_leave_days=float(row['parental_leave_days']),
                marriage_leave_days=float(row['marriage_leave_days']),
                birth_subsidy_amount=float(row['birth_subsidy_amount']),
                housing_subsidy_amount=float(row['housing_subsidy_amount']),
                childcare_subsidy_monthly=float(row['childcare_subsidy_monthly']),
                tax_deduction_monthly=float(row['tax_deduction_monthly']),
                has_childcare_policy=1.0 if row['has_childcare_policy'] else 0.0,
                has_ivf_insurance=1.0 if row['has_ivf_insurance'] else 0.0,
                is_minority_favorable=1.0 if row['is_minority_favorable'] else 0.0,
                policy_phase=float(row['policy_phase'])
            )
            self.province_features[province][year] = features

            # Track reform year (when policy_phase transitions to 2 = 三孩政策)
            if row['policy_phase'] == 2:
                if province not in self.province_reform_years:
                    self.province_reform_years[province] = year

        # Clear DataFrame to free memory (dict is sufficient for queries)
        self.policy_df = None

    def _parse_new_format(self):
        """Parse the new province-year format (fertility_policies_by_province_year.json).

        This format has pre-computed features for each province-year combination,
        making parsing straightforward and eliminating complex temporal logic.
        """
        policies = self.policy_data.get("policies", {})

        # Parse national baseline from metadata or compute default
        self._parse_national_policies()

        for province_name, year_data in policies.items():
            self.province_features[province_name] = {}

            for year_str, policy_info in year_data.items():
                year = int(year_str)
                features = PolicyFeatures()

                # Extract leave standards
                leave_standards = policy_info.get("leave_standards", {})
                features.maternity_leave_days = float(leave_standards.get("maternity_leave_days", 128))
                features.paternity_leave_days = float(leave_standards.get("paternity_leave_days", 15))
                features.parental_leave_days = float(leave_standards.get("parental_leave_days", 0))
                features.marriage_leave_days = float(leave_standards.get("marriage_leave_days", 10))

                # Extract economic support
                economic = policy_info.get("economic_support", {})
                features.birth_subsidy_amount = float(economic.get("birth_subsidy_amount", 0))
                features.childcare_subsidy_monthly = float(economic.get("childcare_subsidy_monthly", 0))
                features.tax_deduction_monthly = float(economic.get("tax_deduction_monthly", 0))

                # Extract housing support
                housing = policy_info.get("housing_support", {})
                features.housing_subsidy_amount = float(housing.get("housing_subsidy_amount", 0))

                # Extract medical insurance (IVF)
                medical = policy_info.get("medical_insurance", {})
                features.has_ivf_insurance = 1.0 if medical.get("has_ivf_insurance", False) else 0.0

                # Extract childcare development
                childcare = policy_info.get("childcare_development", {})
                features.has_childcare_policy = 1.0 if childcare.get("has_childcare_policy", False) else 0.0

                # Extract minority favorable status
                features.is_minority_favorable = 1.0 if policy_info.get("is_minority_favorable", False) else 0.0

                # Extract policy phase
                features.policy_phase = float(policy_info.get("policy_phase", 1))

                self.province_features[province_name][year] = features

                # Track reform year (when policy_phase transitions to 2 = 三孩政策)
                if policy_info.get("policy_phase") == 2:
                    if province_name not in self.province_reform_years:
                        self.province_reform_years[province_name] = year

    def _parse_national_policies(self):
        """Parse national policy timeline for baseline features."""
        # National policy phases by year
        # 2013-2015: 单独二孩
        # 2016-2020: 全面二孩
        # 2021+: 三孩政策

        for year in range(2013, 2026):
            features = PolicyFeatures()

            if year <= 2015:
                features.policy_phase = POLICY_PHASES["单独二孩"]
                features.tax_deduction_monthly = 0.0
            elif year <= 2020:
                features.policy_phase = POLICY_PHASES["全面二孩"]
                features.tax_deduction_monthly = 0.0
            else:
                features.policy_phase = POLICY_PHASES["三孩政策"]
                # Tax deductions from 2022
                if year >= 2022:
                    features.tax_deduction_monthly = 1000.0  # 1000元/月
                if year >= 2023:
                    features.tax_deduction_monthly = 2000.0  # Increased to 2000元/月

            self.national_features[year] = features

    def _parse_provincial_policies(self):
        """Parse all provincial policies (old format)."""
        provincial_data = self.policy_data.get("provincial_policies", {})

        for region_name, region_data in provincial_data.items():
            for province_name, province_data in region_data.items():
                self._parse_single_province(province_name, province_data)

    def _parse_single_province(self, province_name: str, province_data: dict):
        """Parse policies for a single province with proper temporal handling (old format).

        Key fix: Uses historical standards before policy reform and new standards after.
        This prevents data leakage from using future policy information.
        """
        self.province_features[province_name] = {}

        # Determine if minority region
        is_minority = province_name in MINORITY_REGIONS

        # Get NEW leave standards (post-2021 reform, from "current_leave_standards")
        leave_standards = province_data.get("current_leave_standards", {})
        new_maternity_days = (
            parse_leave_days(leave_standards.get("maternity_leave_days")) or
            parse_leave_days(leave_standards.get("maternity_leave_days_first")) or
            parse_leave_days(leave_standards.get("maternity_leave_days_first_second")) or
            158.0  # post-2021 national default
        )
        new_paternity_days = parse_leave_days(leave_standards.get("paternity_leave_days")) or 15.0
        new_parental_days = parse_leave_days(leave_standards.get("parental_leave_days_per_year")) or 10.0
        new_marriage_days = parse_leave_days(leave_standards.get("marriage_leave_days")) or 10.0

        # Parse regulation revisions to determine policy timeline AND reform year
        regulations = province_data.get("regulation_revisions", [])
        policy_timeline = []  # (year, policy_phase)
        reform_year = 2021  # Default: assume reform happened in 2021

        for reg in regulations:
            date_str = reg.get("release_date", "")
            phase = reg.get("policy_phase", "")
            if date_str and phase in POLICY_PHASES:
                try:
                    year = int(date_str.split("-")[0])
                    policy_timeline.append((year, POLICY_PHASES[phase]))
                    # 三孩政策 phase indicates when new leave standards took effect
                    if phase == "三孩政策":
                        reform_year = year
                except:
                    pass

        policy_timeline.sort(key=lambda x: x[0])
        self.province_reform_years[province_name] = reform_year

        # Parse economic support with implementation year detection
        birth_subsidy = 0.0
        childcare_subsidy = 0.0
        has_childcare_policy = 0.0
        has_ivf_insurance = 0.0
        housing_subsidy = 0.0
        subsidy_impl_year = 2024  # Default: assume subsidy not yet implemented

        econ_support = province_data.get("economic_support", [])
        if isinstance(econ_support, list):
            for item in econ_support:
                if "育儿补贴" in item.get("subsidy_type", "") or "育儿补贴" in item.get("policy_name", ""):
                    amount = item.get("amount", 0)
                    if amount:
                        birth_subsidy = parse_subsidy_amount(amount)
                    # Try to extract implementation year from policy name or date
                    impl_date = item.get("implementation_date", "") or item.get("policy_date", "")
                    if impl_date:
                        try:
                            subsidy_impl_year = min(subsidy_impl_year, int(impl_date.split("-")[0]))
                        except:
                            pass
                if "托育" in item.get("policy_name", ""):
                    has_childcare_policy = 1.0

        # Parse housing support
        housing_support = province_data.get("housing_support", [])
        if isinstance(housing_support, list) and len(housing_support) > 0:
            for item in housing_support:
                content = item.get("main_content", "")
                match = re.search(r'(\d+)万', content)
                if match:
                    housing_subsidy = float(match.group(1))
                    break

        # Parse medical insurance (IVF coverage)
        ivf_impl_year = 2024  # Default: assume not yet implemented
        medical_policies = province_data.get("medical_insurance_policies", [])
        if isinstance(medical_policies, list):
            for item in medical_policies:
                if "辅助生殖" in item.get("main_content", "") or "试管" in item.get("main_content", ""):
                    has_ivf_insurance = 1.0
                    impl_date = item.get("implementation_date", "")
                    if impl_date:
                        try:
                            ivf_impl_year = int(impl_date.split("-")[0])
                        except:
                            # Most IVF policies started in 2023 (Beijing) or later
                            ivf_impl_year = 2023
                    else:
                        ivf_impl_year = 2023  # Conservative estimate
                    break

        # Parse childcare development
        childcare_dev = province_data.get("childcare_development", {})
        childcare_impl_year = 2022 if childcare_dev else 2024  # Most started around 2022

        # Create features for each year (2013-2025 to support full range)
        for year in range(2013, 2026):
            features = PolicyFeatures()

            # === LEAVE DAYS: Use OLD standards before reform, NEW standards after ===
            if year < reform_year:
                # Before reform: use old national standards
                features.maternity_leave_days = self.OLD_LEAVE_STANDARDS['maternity_leave_days']
                features.paternity_leave_days = self.OLD_LEAVE_STANDARDS['paternity_leave_days']
                features.parental_leave_days = self.OLD_LEAVE_STANDARDS['parental_leave_days']
                features.marriage_leave_days = self.OLD_LEAVE_STANDARDS['marriage_leave_days']
            else:
                # After reform: use new province-specific standards
                features.maternity_leave_days = new_maternity_days
                features.paternity_leave_days = new_paternity_days
                features.parental_leave_days = new_parental_days
                features.marriage_leave_days = new_marriage_days

            features.is_minority_favorable = 1.0 if is_minority else 0.0

            # === ECONOMIC FEATURES: Only available after their implementation year ===
            if year >= subsidy_impl_year:
                features.birth_subsidy_amount = birth_subsidy
            if year >= reform_year:  # Housing support generally tied to reform
                features.housing_subsidy_amount = housing_subsidy
            if year >= childcare_impl_year:
                features.has_childcare_policy = 1.0 if childcare_dev else 0.0
            if year >= ivf_impl_year:
                features.has_ivf_insurance = 1.0 if has_ivf_insurance else 0.0

            # Tax deductions (national policy, well-documented timeline)
            if year >= 2022:
                features.tax_deduction_monthly = 1000.0
            if year >= 2023:
                features.tax_deduction_monthly = 2000.0

            # Determine policy phase from timeline
            current_phase = POLICY_PHASES["全面二孩"]  # default for 2016+
            if year <= 2015:
                current_phase = POLICY_PHASES["单独二孩"]
            for timeline_year, phase in policy_timeline:
                if timeline_year <= year:
                    current_phase = phase
            features.policy_phase = current_phase

            self.province_features[province_name][year] = features

    def get_features(self, city: str, year: int, apply_lag: bool = True) -> np.ndarray:
        """
        Get policy features for a city-year pair.

        Args:
            city: City name (e.g., "北京市")
            year: Year (2018-2024) - the year for which we want to predict outcomes
            apply_lag: Whether to apply policy_lag (default: True).
                       When True, returns policy features from (year - policy_lag).
                       This prevents data leakage by using past policy to predict current outcomes.

        Returns:
            numpy array of shape (12,) with features
        """
        # Apply temporal lag to prevent data leakage
        # If predicting Y2022 outcome, use Y2021 policy (with lag=1)
        policy_year = year - self.policy_lag if apply_lag else year

        # Map city to province
        province = CITY_TO_PROVINCE.get(city, city)  # If not found, use city name directly

        # Use dict lookup (O(1), works for all formats)
        if province in self.province_features and policy_year in self.province_features[province]:
            features = self.province_features[province][policy_year]
        else:
            # Fallback to national baseline
            if policy_year in self.national_features:
                features = self.national_features[policy_year]
            else:
                features = PolicyFeatures()

        return features.to_vector()

    def get_normalized_features(self, city: str, year: int, apply_lag: bool = True) -> np.ndarray:
        """
        Get normalized policy features (0-1 range) for a city-year pair.

        Args:
            city: City name (e.g., "北京市")
            year: Year (2018-2024) - the year for which we want to predict outcomes
            apply_lag: Whether to apply policy_lag (default: True).
                       This prevents data leakage by using past policy to predict current outcomes.

        Normalization ranges:
        - maternity_leave_days: [98, 365] -> [0, 1]
        - paternity_leave_days: [0, 30] -> [0, 1]
        - parental_leave_days: [0, 20] -> [0, 1]
        - marriage_leave_days: [3, 30] -> [0, 1]
        - birth_subsidy_amount: [0, 10] 万元 -> [0, 1]
        - housing_subsidy_amount: [0, 100] 万元 -> [0, 1]
        - childcare_subsidy_monthly: [0, 2000] 元/月 -> [0, 1]
        - tax_deduction_monthly: [0, 2000] 元/月 -> [0, 1]
        - binary features: already 0 or 1
        - policy_phase: [0, 3] -> [0, 1]
        """
        raw_features = self.get_features(city, year, apply_lag=apply_lag)

        # Normalization ranges
        norm_ranges = [
            (98, 365),    # maternity_leave_days
            (0, 30),      # paternity_leave_days
            (0, 20),      # parental_leave_days
            (3, 30),      # marriage_leave_days
            (0, 10),      # birth_subsidy_amount (万元)
            (0, 100),     # housing_subsidy_amount (万元)
            (0, 2000),    # childcare_subsidy_monthly
            (0, 2000),    # tax_deduction_monthly
            (0, 1),       # has_childcare_policy (binary)
            (0, 1),       # has_ivf_insurance (binary)
            (0, 1),       # is_minority_favorable (binary)
            (0, 3),       # policy_phase
        ]

        normalized = np.zeros_like(raw_features)
        for i, (min_val, max_val) in enumerate(norm_ranges):
            if max_val > min_val:
                normalized[i] = (raw_features[i] - min_val) / (max_val - min_val)
                normalized[i] = np.clip(normalized[i], 0, 1)
            else:
                normalized[i] = raw_features[i]

        return normalized.astype(np.float32)

    @property
    def feature_dim(self) -> int:
        """Return the dimension of policy features."""
        return 12

    @property
    def feature_names(self) -> List[str]:
        """Return names of policy features."""
        return [
            "maternity_leave_days",
            "paternity_leave_days",
            "parental_leave_days",
            "marriage_leave_days",
            "birth_subsidy_amount",
            "housing_subsidy_amount",
            "childcare_subsidy_monthly",
            "tax_deduction_monthly",
            "has_childcare_policy",
            "has_ivf_insurance",
            "is_minority_favorable",
            "policy_phase"
        ]


# Global extractor instance (lazy initialization)
_extractor = None
_extractor_lag = None


def get_policy_extractor(policy_lag: int = 1) -> PolicyFeatureExtractor:
    """Get the global policy feature extractor instance.

    Args:
        policy_lag: Number of years to lag policy features (default: 1).
                    This is set once when the extractor is first created.
    """
    global _extractor, _extractor_lag
    if _extractor is None or _extractor_lag != policy_lag:
        _extractor = PolicyFeatureExtractor(policy_lag=policy_lag)
        _extractor_lag = policy_lag
    return _extractor


def get_policy_features(city: str, year: int, normalized: bool = True, apply_lag: bool = True) -> np.ndarray:
    """
    Convenience function to get policy features for a city-year pair.

    Args:
        city: City name
        year: Year for which we want to predict outcomes
        normalized: Whether to return normalized features
        apply_lag: Whether to apply policy_lag (default: True).
                   This prevents data leakage by using past policy to predict current outcomes.

    Returns:
        numpy array of shape (12,)
    """
    extractor = get_policy_extractor()
    if normalized:
        return extractor.get_normalized_features(city, year, apply_lag=apply_lag)
    return extractor.get_features(city, year, apply_lag=apply_lag)


if __name__ == "__main__":
    # Test the module
    print("=" * 60)
    print("Testing PolicyFeatureExtractor with Temporal Lag")
    print("=" * 60)

    extractor = PolicyFeatureExtractor(policy_lag=1)
    print(f"\nPolicy lag: {extractor.policy_lag} year(s)")
    print(f"Feature names: {extractor.feature_names}")
    print(f"Feature dimension: {extractor.feature_dim}")

    # Test temporal lag effect - show how features differ before/after reform
    print("\n" + "=" * 60)
    print("Testing temporal lag and historical standards")
    print("=" * 60)

    test_city = "北京市"
    print(f"\n{test_city} - comparing features across years:")
    print("(With lag=1: predicting Y2022 uses Y2021 policy, etc.)")

    for year in [2020, 2021, 2022, 2023]:
        print(f"\n  Predicting outcome for {year} (using policy from {year - extractor.policy_lag}):")
        raw = extractor.get_features(test_city, year, apply_lag=True)
        # Show key features
        print(f"    maternity_leave_days: {raw[0]:.0f}")
        print(f"    paternity_leave_days: {raw[1]:.0f}")
        print(f"    parental_leave_days:  {raw[2]:.0f}")
        print(f"    tax_deduction_monthly: {raw[7]:.0f}")
        print(f"    policy_phase:         {raw[11]:.0f}")

    # Compare with/without lag
    print("\n" + "=" * 60)
    print("Comparing WITH vs WITHOUT lag (potential leakage)")
    print("=" * 60)

    for year in [2021, 2022]:
        print(f"\n  Year {year}:")
        with_lag = extractor.get_features(test_city, year, apply_lag=True)
        without_lag = extractor.get_features(test_city, year, apply_lag=False)
        print(f"    WITH lag (uses {year-1} policy):    maternity={with_lag[0]:.0f}, paternity={with_lag[1]:.0f}")
        print(f"    WITHOUT lag (uses {year} policy): maternity={without_lag[0]:.0f}, paternity={without_lag[1]:.0f}")
        if with_lag[0] != without_lag[0]:
            print(f"    -> Difference detected! Lag prevents using future policy info.")

    # Test some cities
    print("\n" + "=" * 60)
    print("Testing various cities")
    print("=" * 60)

    test_cases = [
        ("北京市", 2022),
        ("上海市", 2023),
        ("西安市", 2024),
        ("呼和浩特市", 2022),  # Inner Mongolia (minority region)
        ("拉萨市", 2023),      # Tibet (minority region)
    ]

    for city, year in test_cases:
        print(f"\n{city} ({year}, using {year-1} policy):")
        raw = extractor.get_features(city, year)
        norm = extractor.get_normalized_features(city, year)

        for i, name in enumerate(extractor.feature_names):
            print(f"  {name}: {raw[i]:.2f} (normalized: {norm[i]:.4f})")

    print("\n" + "=" * 60)
    print("Testing unknown city fallback")
    print("=" * 60)
    unknown_features = extractor.get_features("未知城市", 2022)
    print(f"Unknown city features: {unknown_features}")

    print("\n" + "=" * 60)
    print("Province reform years (when new leave standards took effect)")
    print("=" * 60)
    for province, year in sorted(extractor.province_reform_years.items(), key=lambda x: x[1]):
        print(f"  {province}: {year}")
