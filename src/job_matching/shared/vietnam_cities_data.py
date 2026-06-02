
import re
from typing import List, Dict, Optional

VIETNAM_CITIES = {
    # North Region (24 provinces)
    'Hà Nội': {
        'lat': 21.0285, 'lon': 105.8542,
        'aliases': ['ha noi', 'hanoi', 'hn'],
        'region': 'north', 'tier': 1
    },
    'Hải Phòng': {
        'lat': 20.8449, 'lon': 106.6881,
        'aliases': ['hai phong', 'haiphong', 'hp'],
        'region': 'north', 'tier': 2
    },
    'Quảng Ninh': {
        'lat': 21.0064, 'lon': 107.2925,
        'aliases': ['quang ninh', 'ha long', 'halong'],
        'region': 'north', 'tier': 2
    },
    'Bắc Ninh': {
        'lat': 21.1214, 'lon': 106.1110,
        'aliases': ['bac ninh'],
        'region': 'north', 'tier': 3
    },
    'Hải Dương': {
        'lat': 20.9373, 'lon': 106.3145,
        'aliases': ['hai duong'],
        'region': 'north', 'tier': 3
    },
    'Hưng Yên': {
        'lat': 20.6464, 'lon': 106.0511,
        'aliases': ['hung yen'],
        'region': 'north', 'tier': 3
    },
    'Vĩnh Phúc': {
        'lat': 21.3609, 'lon': 105.5474,
        'aliases': ['vinh phuc'],
        'region': 'north', 'tier': 3
    },
    'Thái Nguyên': {
        'lat': 21.5671, 'lon': 105.8252,
        'aliases': ['thai nguyen'],
        'region': 'north', 'tier': 3
    },
    'Bắc Giang': {
        'lat': 21.2738, 'lon': 106.1946,
        'aliases': ['bac giang'],
        'region': 'north', 'tier': 3
    },
    'Lạng Sơn': {
        'lat': 21.8537, 'lon': 106.7615,
        'aliases': ['lang son'],
        'region': 'north', 'tier': 3
    },
    'Cao Bằng': {
        'lat': 22.6356, 'lon': 106.2522,
        'aliases': ['cao bang'],
        'region': 'north', 'tier': 3
    },
    'Lào Cai': {
        'lat': 22.4809, 'lon': 103.9750,
        'aliases': ['lao cai', 'sapa'],
        'region': 'north', 'tier': 3
    },
    'Điện Biên': {
        'lat': 21.3833, 'lon': 103.0167,
        'aliases': ['dien bien'],
        'region': 'north', 'tier': 3
    },
    'Lai Châu': {
        'lat': 22.3864, 'lon': 103.4702,
        'aliases': ['lai chau'],
        'region': 'north', 'tier': 3
    },
    'Sơn La': {
        'lat': 21.3256, 'lon': 103.9188,
        'aliases': ['son la'],
        'region': 'north', 'tier': 3
    },
    'Yên Bái': {
        'lat': 21.7167, 'lon': 104.8667,
        'aliases': ['yen bai'],
        'region': 'north', 'tier': 3
    },
    'Tuyên Quang': {
        'lat': 21.8236, 'lon': 105.2280,
        'aliases': ['tuyen quang'],
        'region': 'north', 'tier': 3
    },
    'Phú Thọ': {
        'lat': 21.4208, 'lon': 105.2045,
        'aliases': ['phu tho', 'viet tri'],
        'region': 'north', 'tier': 3
    },
    'Hà Giang': {
        'lat': 22.8025, 'lon': 104.9784,
        'aliases': ['ha giang'],
        'region': 'north', 'tier': 3
    },
    'Hà Nam': {
        'lat': 20.5385, 'lon': 105.9230,
        'aliases': ['ha nam', 'phu ly'],
        'region': 'north', 'tier': 3
    },
    'Nam Định': {
        'lat': 20.4388, 'lon': 106.1621,
        'aliases': ['nam dinh'],
        'region': 'north', 'tier': 3
    },
    'Thái Bình': {
        'lat': 20.4464, 'lon': 106.3365,
        'aliases': ['thai binh'],
        'region': 'north', 'tier': 3
    },
    'Ninh Bình': {
        'lat': 20.2506, 'lon': 105.9745,
        'aliases': ['ninh binh'],
        'region': 'north', 'tier': 3
    },
    'Hòa Bình': {
        'lat': 20.6861, 'lon': 105.3131,
        'aliases': ['hoa binh'],
        'region': 'north', 'tier': 3
    },

    # Central Region (19 provinces)
    'Đà Nẵng': {
        'lat': 16.0544, 'lon': 108.2022,
        'aliases': ['da nang', 'danang', 'dn'],
        'region': 'central', 'tier': 1
    },
    'Thừa Thiên Huế': {
        'lat': 16.4637, 'lon': 107.5909,
        'aliases': ['hue', 'thua thien hue'],
        'region': 'central', 'tier': 2
    },
    'Quảng Nam': {
        'lat': 15.5394, 'lon': 108.0191,
        'aliases': ['quang nam', 'hoi an'],
        'region': 'central', 'tier': 2
    },
    'Quảng Ngãi': {
        'lat': 15.1214, 'lon': 108.8044,
        'aliases': ['quang ngai'],
        'region': 'central', 'tier': 3
    },
    'Bình Định': {
        'lat': 13.7830, 'lon': 109.2196,
        'aliases': ['binh dinh', 'quy nhon'],
        'region': 'central', 'tier': 3
    },
    'Phú Yên': {
        'lat': 13.0881, 'lon': 109.0929,
        'aliases': ['phu yen', 'tuy hoa'],
        'region': 'central', 'tier': 3
    },
    'Khánh Hòa': {
        'lat': 12.2388, 'lon': 109.1967,
        'aliases': ['khanh hoa', 'nha trang'],
        'region': 'central', 'tier': 2
    },
    'Ninh Thuận': {
        'lat': 11.6739, 'lon': 108.8629,
        'aliases': ['ninh thuan', 'phan rang'],
        'region': 'central', 'tier': 3
    },
    'Bình Thuận': {
        'lat': 10.9273, 'lon': 108.1022,
        'aliases': ['binh thuan', 'phan thiet'],
        'region': 'central', 'tier': 3
    },
    'Kon Tum': {
        'lat': 14.3497, 'lon': 108.0004,
        'aliases': ['kon tum', 'kontum'],
        'region': 'central', 'tier': 3
    },
    'Gia Lai': {
        'lat': 13.9833, 'lon': 108.0000,
        'aliases': ['gia lai', 'pleiku'],
        'region': 'central', 'tier': 3
    },
    'Đắk Lắk': {
        'lat': 12.6667, 'lon': 108.0500,
        'aliases': ['dak lak', 'buon ma thuot'],
        'region': 'central', 'tier': 3
    },
    'Đắk Nông': {
        'lat': 12.2646, 'lon': 107.6098,
        'aliases': ['dak nong', 'gia nghia'],
        'region': 'central', 'tier': 3
    },
    'Lâm Đồng': {
        'lat': 11.9404, 'lon': 108.4583,
        'aliases': ['lam dong', 'da lat', 'dalat'],
        'region': 'central', 'tier': 2
    },
    'Quảng Bình': {
        'lat': 17.4676, 'lon': 106.6220,
        'aliases': ['quang binh', 'dong hoi'],
        'region': 'central', 'tier': 3
    },
    'Quảng Trị': {
        'lat': 16.7403, 'lon': 107.1854,
        'aliases': ['quang tri', 'dong ha'],
        'region': 'central', 'tier': 3
    },
    'Hà Tĩnh': {
        'lat': 18.3429, 'lon': 105.8879,
        'aliases': ['ha tinh'],
        'region': 'central', 'tier': 3
    },
    'Nghệ An': {
        'lat': 18.6792, 'lon': 105.6819,
        'aliases': ['nghe an', 'vinh'],
        'region': 'central', 'tier': 3
    },
    'Thanh Hóa': {
        'lat': 19.8000, 'lon': 105.7667,
        'aliases': ['thanh hoa'],
        'region': 'central', 'tier': 3
    },

    # South Region (20 provinces)
    'Hồ Chí Minh': {
        'lat': 10.8231, 'lon': 106.6297,
        'aliases': ['ho chi minh', 'hcm', 'sai gon', 'saigon', 'tp hcm'],
        'region': 'south', 'tier': 1
    },
    'Bình Dương': {
        'lat': 11.3254, 'lon': 106.4770,
        'aliases': ['binh duong', 'thu dau mot'],
        'region': 'south', 'tier': 2
    },
    'Đồng Nai': {
        'lat': 10.9465, 'lon': 106.8340,
        'aliases': ['dong nai', 'bien hoa'],
        'region': 'south', 'tier': 2
    },
    'Bà Rịa - Vũng Tàu': {
        'lat': 10.4114, 'lon': 107.1362,
        'aliases': ['vung tau', 'ba ria'],
        'region': 'south', 'tier': 2
    },
    'Cần Thơ': {
        'lat': 10.0452, 'lon': 105.7469,
        'aliases': ['can tho', 'cantho'],
        'region': 'south', 'tier': 2
    },
    'Long An': {
        'lat': 10.5356, 'lon': 106.4056,
        'aliases': ['long an', 'tan an'],
        'region': 'south', 'tier': 3
    },
    'Tiền Giang': {
        'lat': 10.3599, 'lon': 106.3601,
        'aliases': ['tien giang', 'my tho'],
        'region': 'south', 'tier': 3
    },
    'Bến Tre': {
        'lat': 10.2433, 'lon': 106.3757,
        'aliases': ['ben tre'],
        'region': 'south', 'tier': 3
    },
    'Trà Vinh': {
        'lat': 9.8128, 'lon': 106.2992,
        'aliases': ['tra vinh'],
        'region': 'south', 'tier': 3
    },
    'Vĩnh Long': {
        'lat': 10.2395, 'lon': 105.9572,
        'aliases': ['vinh long'],
        'region': 'south', 'tier': 3
    },
    'Đồng Tháp': {
        'lat': 10.4938, 'lon': 105.6881,
        'aliases': ['dong thap', 'cao lanh'],
        'region': 'south', 'tier': 3
    },
    'An Giang': {
        'lat': 10.3864, 'lon': 105.4359,
        'aliases': ['an giang', 'long xuyen', 'chau doc'],
        'region': 'south', 'tier': 3
    },
    'Kiên Giang': {
        'lat': 10.0125, 'lon': 105.0808,
        'aliases': ['kien giang', 'rach gia', 'phu quoc'],
        'region': 'south', 'tier': 3
    },
    'Hậu Giang': {
        'lat': 9.7577, 'lon': 105.6412,
        'aliases': ['hau giang', 'vi thanh'],
        'region': 'south', 'tier': 3
    },
    'Sóc Trăng': {
        'lat': 9.6024, 'lon': 105.9739,
        'aliases': ['soc trang'],
        'region': 'south', 'tier': 3
    },
    'Bạc Liêu': {
        'lat': 9.2515, 'lon': 105.7247,
        'aliases': ['bac lieu'],
        'region': 'south', 'tier': 3
    },
    'Cà Mau': {
        'lat': 9.1526, 'lon': 105.1960,
        'aliases': ['ca mau'],
        'region': 'south', 'tier': 3
    },
    'Tây Ninh': {
        'lat': 11.3351, 'lon': 106.1098,
        'aliases': ['tay ninh'],
        'region': 'south', 'tier': 3
    },
    'Bình Phước': {
        'lat': 11.7511, 'lon': 106.7234,
        'aliases': ['binh phuoc', 'dong xoai'],
        'region': 'south', 'tier': 3
    },
}

# Build fast lookup dict: alias -> city_data (O(1) instead of O(n))
_CITY_LOOKUP: Dict[str, Dict] = {}
for city_name, city_data in VIETNAM_CITIES.items():
    city_info = city_data.copy()
    city_info['name'] = city_name
    for alias in city_data['aliases']:
        _CITY_LOOKUP[alias.lower()] = city_info


def get_city_info(location_text: str) -> List[Dict]:
    if not location_text:
        return []

    location_lower = location_text.lower().strip()
    matched_cities = []

    # Address keywords that indicate it's NOT a city name
    address_keywords_before = [
        'phuong', 'quan', 'duong', 'pho', 'thon', 'ap', 'xa',
        'huyen', 'thi tran', 'khu pho', 'to', 'ngo', 'hem'
    ]

    address_keywords_after = [
        'phu',  # Điện Biên Phủ = street
        'plaza', 'tower', 'building', 'center', 'centre',
        'street', 'road', 'avenue'
    ]

    for alias, city_info in _CITY_LOOKUP.items():
        # Use word boundary for exact matching
        pattern = r'\b' + re.escape(alias) + r'\b'
        match = re.search(pattern, location_lower)

        if match:
            idx = match.start()
            prefix = location_lower[max(0, idx - 20):idx].strip()
            suffix = location_lower[idx + len(alias):min(len(location_lower), idx + len(alias) + 20)].strip()

            is_false_positive = False

            # Check keywords before alias
            for keyword in address_keywords_before:
                if prefix.endswith(keyword) or prefix.endswith(keyword + ' '):
                    is_false_positive = True
                    break

            # Check keywords after alias
            if not is_false_positive:
                for keyword in address_keywords_after:
                    if suffix.startswith(keyword) or suffix.startswith(' ' + keyword):
                        is_false_positive = True
                        break

            # Add if not false positive and not duplicate
            if not is_false_positive:
                city_name = city_info['name']
                if not any(city['name'] == city_name for city in matched_cities):
                    matched_cities.append({
                        'name': city_name,
                        'lat': city_info['lat'],
                        'lon': city_info['lon'],
                        'region': city_info['region'],
                        'tier': city_info['tier'],
                        'population': city_info['population']
                    })

    return matched_cities


def get_city_by_name(city_name: str) -> Optional[Dict]:
    return _CITY_LOOKUP.get(city_name.lower())
