import re
import unicodedata
from math import asin, cos, radians, sin, sqrt
from typing import Dict, List, Optional


# Vietnam has 34 province-level administrative units from Resolution
# 202/2025/QH15. Coordinates are representative administrative-center
# coordinates used for city/province-level matching, not legal boundaries.
ADMIN_UNITS_VERSION = "vn_province_level_34_2025"

VIETNAM_CITIES = {
    # Unchanged province-level units
    "Hà Nội": {
        "lat": 21.0285,
        "lon": 105.8542,
        "aliases": ["ha noi", "hanoi", "hn", "thanh pho ha noi", "tp ha noi"],
        "region": "north",
        "tier": 1,
    },
    "Huế": {
        "lat": 16.4637,
        "lon": 107.5909,
        "aliases": [
            "hue",
            "thua thien hue",
            "thua thien-hue",
            "thua thien - hue",
            "thanh pho hue",
            "tp hue",
        ],
        "region": "central",
        "tier": 1,
    },
    "Lai Châu": {
        "lat": 22.3864,
        "lon": 103.4702,
        "aliases": ["lai chau"],
        "region": "north",
        "tier": 3,
    },
    "Điện Biên": {
        "lat": 21.3833,
        "lon": 103.0167,
        "aliases": ["dien bien", "tp dien bien phu", "thanh pho dien bien phu"],
        "region": "north",
        "tier": 3,
    },
    "Sơn La": {
        "lat": 21.3256,
        "lon": 103.9188,
        "aliases": ["son la"],
        "region": "north",
        "tier": 3,
    },
    "Lạng Sơn": {
        "lat": 21.8537,
        "lon": 106.7615,
        "aliases": ["lang son"],
        "region": "north",
        "tier": 3,
    },
    "Quảng Ninh": {
        "lat": 21.0064,
        "lon": 107.2925,
        "aliases": ["quang ninh", "ha long", "halong", "mong cai", "cam pha"],
        "region": "north",
        "tier": 2,
    },
    "Thanh Hóa": {
        "lat": 19.8000,
        "lon": 105.7667,
        "aliases": ["thanh hoa"],
        "region": "central",
        "tier": 3,
    },
    "Nghệ An": {
        "lat": 18.6792,
        "lon": 105.6819,
        "aliases": ["nghe an", "vinh"],
        "region": "central",
        "tier": 3,
    },
    "Hà Tĩnh": {
        "lat": 18.3429,
        "lon": 105.8879,
        "aliases": ["ha tinh"],
        "region": "central",
        "tier": 3,
    },
    "Cao Bằng": {
        "lat": 22.6356,
        "lon": 106.2522,
        "aliases": ["cao bang"],
        "region": "north",
        "tier": 3,
    },

    # New units after merger/rearrangement. Legacy province names stay as
    # aliases so old crawled job data and CV text still match correctly.
    "Tuyên Quang": {
        "lat": 21.8236,
        "lon": 105.2280,
        "aliases": ["tuyen quang", "ha giang"],
        "region": "north",
        "tier": 3,
    },
    "Lào Cai": {
        "lat": 21.7167,
        "lon": 104.8667,
        "aliases": ["lao cai", "yen bai", "sa pa", "sapa"],
        "region": "north",
        "tier": 3,
    },
    "Thái Nguyên": {
        "lat": 21.5671,
        "lon": 105.8252,
        "aliases": ["thai nguyen", "bac kan", "backan", "bac kạn"],
        "region": "north",
        "tier": 3,
    },
    "Phú Thọ": {
        "lat": 21.4208,
        "lon": 105.2045,
        "aliases": [
            "phu tho",
            "viet tri",
            "vinh phuc",
            "hoa binh",
            "hoà bình",
        ],
        "region": "north",
        "tier": 3,
    },
    "Bắc Ninh": {
        "lat": 21.2738,
        "lon": 106.1946,
        "aliases": ["bac ninh", "bac giang"],
        "region": "north",
        "tier": 3,
    },
    "Hưng Yên": {
        "lat": 20.6464,
        "lon": 106.0511,
        "aliases": ["hung yen", "thai binh"],
        "region": "north",
        "tier": 3,
    },
    "Hải Phòng": {
        "lat": 20.8449,
        "lon": 106.6881,
        "aliases": [
            "hai phong",
            "haiphong",
            "hp",
            "thanh pho hai phong",
            "tp hai phong",
            "hai duong",
        ],
        "region": "north",
        "tier": 2,
    },
    "Ninh Bình": {
        "lat": 20.2506,
        "lon": 105.9745,
        "aliases": ["ninh binh", "ha nam", "nam dinh", "phu ly"],
        "region": "north",
        "tier": 3,
    },
    "Quảng Trị": {
        "lat": 17.4676,
        "lon": 106.6220,
        "aliases": ["quang tri", "dong ha", "quang binh", "dong hoi"],
        "region": "central",
        "tier": 3,
    },
    "Đà Nẵng": {
        "lat": 16.0544,
        "lon": 108.2022,
        "aliases": [
            "da nang",
            "danang",
            "dn",
            "thanh pho da nang",
            "tp da nang",
            "quang nam",
            "hoi an",
        ],
        "region": "central",
        "tier": 1,
    },
    "Quảng Ngãi": {
        "lat": 15.1214,
        "lon": 108.8044,
        "aliases": ["quang ngai", "kon tum", "kontum"],
        "region": "central",
        "tier": 3,
    },
    "Gia Lai": {
        "lat": 13.7830,
        "lon": 109.2196,
        "aliases": ["gia lai", "pleiku", "binh dinh", "quy nhon"],
        "region": "central",
        "tier": 3,
    },
    "Khánh Hòa": {
        "lat": 12.2388,
        "lon": 109.1967,
        "aliases": ["khanh hoa", "nha trang", "ninh thuan", "phan rang"],
        "region": "central",
        "tier": 2,
    },
    "Lâm Đồng": {
        "lat": 11.9404,
        "lon": 108.4583,
        "aliases": [
            "lam dong",
            "da lat",
            "dalat",
            "dak nong",
            "dac nong",
            "gia nghia",
            "binh thuan",
            "phan thiet",
        ],
        "region": "central",
        "tier": 2,
    },
    "Đắk Lắk": {
        "lat": 12.6667,
        "lon": 108.0500,
        "aliases": [
            "dak lak",
            "daklak",
            "dac lac",
            "dac lak",
            "buon ma thuot",
            "phu yen",
            "tuy hoa",
        ],
        "region": "central",
        "tier": 3,
    },
    "Hồ Chí Minh": {
        "lat": 10.8231,
        "lon": 106.6297,
        "aliases": [
            "ho chi minh",
            "hcm",
            "hcmc",
            "tp hcm",
            "tphcm",
            "sai gon",
            "saigon",
            "thanh pho ho chi minh",
            "ba ria - vung tau",
            "ba ria vung tau",
            "vung tau",
            "ba ria",
            "binh duong",
            "thu dau mot",
        ],
        "region": "south",
        "tier": 1,
    },
    "Đồng Nai": {
        "lat": 10.9465,
        "lon": 106.8340,
        "aliases": ["dong nai", "bien hoa", "binh phuoc", "dong xoai"],
        "region": "south",
        "tier": 2,
    },
    "Tây Ninh": {
        "lat": 10.5356,
        "lon": 106.4056,
        "aliases": ["tay ninh", "long an", "tan an"],
        "region": "south",
        "tier": 3,
    },
    "Cần Thơ": {
        "lat": 10.0452,
        "lon": 105.7469,
        "aliases": [
            "can tho",
            "cantho",
            "thanh pho can tho",
            "tp can tho",
            "hau giang",
            "vi thanh",
            "soc trang",
        ],
        "region": "south",
        "tier": 2,
    },
    "Vĩnh Long": {
        "lat": 10.2395,
        "lon": 105.9572,
        "aliases": ["vinh long", "ben tre", "tra vinh"],
        "region": "south",
        "tier": 3,
    },
    "Đồng Tháp": {
        "lat": 10.3599,
        "lon": 106.3601,
        "aliases": ["dong thap", "cao lanh", "tien giang", "my tho"],
        "region": "south",
        "tier": 3,
    },
    "Cà Mau": {
        "lat": 9.1526,
        "lon": 105.1960,
        "aliases": ["ca mau", "bac lieu"],
        "region": "south",
        "tier": 3,
    },
    "An Giang": {
        "lat": 10.0125,
        "lon": 105.0808,
        "aliases": [
            "an giang",
            "long xuyen",
            "chau doc",
            "kien giang",
            "rach gia",
            "phu quoc",
        ],
        "region": "south",
        "tier": 3,
    },
}


def _normalize_location_text(value: str) -> str:
    """Normalize Vietnamese location text for accent-insensitive matching."""
    normalized = unicodedata.normalize("NFD", str(value or "").lower())
    normalized = "".join(
        ch for ch in normalized if unicodedata.category(ch) != "Mn"
    )
    return normalized.replace("đ", "d")


def _build_city_lookup() -> Dict[str, Dict]:
    lookup: Dict[str, Dict] = {}
    for city_name, city_data in VIETNAM_CITIES.items():
        city_info = city_data.copy()
        city_info["name"] = city_name
        for alias in [city_name, *city_data["aliases"]]:
            lookup[_normalize_location_text(alias)] = city_info
    return lookup


_CITY_LOOKUP = _build_city_lookup()


def _haversine_km(origin: Dict, destination: Dict) -> float:
    lat1, lon1 = radians(origin["lat"]), radians(origin["lon"])
    lat2, lon2 = radians(destination["lat"]), radians(destination["lon"])
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    value = (
        sin(delta_lat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(delta_lon / 2) ** 2
    )
    return 6371.0 * 2 * asin(sqrt(value))


def _build_province_distance_graph() -> Dict[str, Dict[str, float]]:
    graph: Dict[str, Dict[str, float]] = {}
    for origin_name, origin in VIETNAM_CITIES.items():
        graph[origin_name] = {
            destination_name: _haversine_km(origin, destination)
            for destination_name, destination in VIETNAM_CITIES.items()
        }
    return graph


PROVINCE_DISTANCE_GRAPH = _build_province_distance_graph()


def get_city_info(location_text: str) -> List[Dict]:
    if not location_text:
        return []

    location_lower = _normalize_location_text(location_text).strip()
    matched_cities = []

    address_keywords_before = [
        "phuong",
        "quan",
        "duong",
        "pho",
        "thon",
        "ap",
        "xa",
        "huyen",
        "thi tran",
        "khu pho",
        "to",
        "ngo",
        "hem",
    ]

    address_keywords_after = [
        "phu",  # Dien Bien Phu as a street/place, not always the province.
        "plaza",
        "tower",
        "building",
        "center",
        "centre",
        "street",
        "road",
        "avenue",
    ]

    # Longer aliases first avoids short forms such as "dn" or "hp" winning
    # before full province/city names.
    aliases = sorted(_CITY_LOOKUP.items(), key=lambda item: len(item[0]), reverse=True)
    for alias, city_info in aliases:
        pattern = r"\b" + re.escape(alias) + r"\b"
        match = re.search(pattern, location_lower)

        if match:
            idx = match.start()
            prefix = location_lower[max(0, idx - 20):idx].strip()
            suffix = location_lower[
                idx + len(alias):min(len(location_lower), idx + len(alias) + 20)
            ].strip()

            is_false_positive = False

            for keyword in address_keywords_before:
                if keyword == "pho" and prefix.endswith("thanh pho"):
                    continue
                if prefix.endswith(keyword) or prefix.endswith(keyword + " "):
                    is_false_positive = True
                    break

            if not is_false_positive:
                for keyword in address_keywords_after:
                    if suffix.startswith(keyword) or suffix.startswith(" " + keyword):
                        is_false_positive = True
                        break

            if not is_false_positive:
                city_name = city_info["name"]
                if not any(city["name"] == city_name for city in matched_cities):
                    matched_cities.append(
                        {
                            "name": city_name,
                            "lat": city_info["lat"],
                            "lon": city_info["lon"],
                            "region": city_info["region"],
                            "tier": city_info["tier"],
                            "population": city_info.get("population"),
                        }
                    )

    return matched_cities


def get_city_by_name(city_name: str) -> Optional[Dict]:
    return _CITY_LOOKUP.get(_normalize_location_text(city_name))


def get_all_city_names() -> List[str]:
    return list(VIETNAM_CITIES.keys())


def sort_provinces_by_distance(origin_text: str, province_names: List[str]) -> List[str]:
    """Order canonical province names by distance from the first origin match."""
    origin_matches = get_city_info(origin_text)
    if not origin_matches:
        return list(dict.fromkeys(province_names))

    origin_name = origin_matches[0]["name"]
    canonical_names = []
    for name in province_names:
        city = get_city_by_name(name)
        if city and city["name"] not in canonical_names:
            canonical_names.append(city["name"])

    return sorted(
        canonical_names,
        key=lambda name: PROVINCE_DISTANCE_GRAPH[origin_name][name],
    )
