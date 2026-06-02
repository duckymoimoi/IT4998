"""
Geocode retry — Bổ sung tọa độ cho jobs chưa có lat/lng.

Chiến lược:
  1. Thử geocode full address
  2. Nếu fail → simplify (bỏ tầng/số nhà/ngõ, giữ quận/phường/TP)
  3. Nếu fail → dùng job_location (tên thành phố) làm fallback

Sử dụng:
  python src/geocode_retry.py --input src/topcv_pipeline_xxx.csv
  python src/geocode_retry.py --input src/topcv_pipeline_xxx.csv --dry-run
"""

import pandas as pd
import re
import sys
import time
import logging
import argparse
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [GEOCODE] %(levelname)-8s %(message)s'
)
logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "DoAnTotNghiep/1.0 (student project)"}

# Cache tránh gọi trùng
_geocode_cache = {}


def _nominatim_search(query, retries=2):
    """Gọi Nominatim, trả về (lat, lng) hoặc None"""
    if query in _geocode_cache:
        return _geocode_cache[query]

    for attempt in range(retries):
        try:
            resp = requests.get(NOMINATIM_URL, params={
                "q": query,
                "format": "json",
                "limit": 1,
                "countrycodes": "vn",
            }, headers=HEADERS, timeout=10)

            if resp.status_code == 200:
                results = resp.json()
                if results:
                    lat = float(results[0]["lat"])
                    lng = float(results[0]["lon"])
                    _geocode_cache[query] = (lat, lng)
                    return (lat, lng)

            _geocode_cache[query] = None
            return None

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                logger.debug(f"  Nominatim error: {e}")
                _geocode_cache[query] = None
                return None
        finally:
            time.sleep(1.1)  # Rate limit: 1 req/s


def simplify_address(addr):
    """
    Rút gọn địa chỉ VN để tăng khả năng match Nominatim.
    Bỏ: tầng, số nhà, ngõ, ngách, lô, khu
    Giữ: đường, phường, quận, thành phố
    """
    if not addr:
        return ""

    # Bỏ prefix thường gặp
    addr = re.sub(r'^(Tầng|Lầu|Phòng|P\.)\s*\d+[A-Za-z]?\s*,?\s*', '', addr, flags=re.IGNORECASE)
    addr = re.sub(r'(Tòa nhà|Building|Tower|Centre|Center)\s+[^,]+,?\s*', '', addr, flags=re.IGNORECASE)

    # Bỏ số nhà, ngõ, ngách
    addr = re.sub(r'Số\s+[\d/]+[A-Za-z]?\s*,?\s*', '', addr, flags=re.IGNORECASE)
    addr = re.sub(r'^\d+[\-/]?\d*[A-Za-z]?\s*,?\s*', '', addr)
    addr = re.sub(r'(Ngõ|Ngách|Hẻm|Lô|Khu)\s+\d+[A-Za-z]?\s*,?\s*', '', addr, flags=re.IGNORECASE)

    # Clean up
    addr = re.sub(r'\s*,\s*,', ',', addr)
    addr = re.sub(r'^\s*,\s*', '', addr)
    addr = addr.strip().strip(',').strip()

    return addr


def extract_city(addr, job_location=""):
    """Trích xuất tên thành phố từ address hoặc job_location"""
    # Từ address
    city_patterns = [
        r'(?:Thành phố|TP\.?|Tp\.?)\s+(Hồ Chí Minh|Hà Nội|Đà Nẵng|Hải Phòng|Cần Thơ)',
        r'(Hồ Chí Minh|Hà Nội|Đà Nẵng|Hải Phòng|Cần Thơ)',
        r'(Bình Dương|Đồng Nai|Bắc Ninh|Hưng Yên|Vĩnh Phúc|Quảng Ninh)',
    ]
    for text in [addr, job_location]:
        for pattern in city_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1) if m.lastindex else m.group(0)

    return job_location.strip() if job_location else ""


def geocode_job(row):
    """
    Geocode 1 job với 3 chiến lược fallback.
    Returns: (lat, lng, method) hoặc (None, None, None)
    """
    addr = str(row.get('company_address', '')).strip()
    job_loc = str(row.get('job_location', '')).strip()

    if addr in ('', 'nan'):
        # Fallback trực tiếp sang city
        city = extract_city('', job_loc)
        if city:
            result = _nominatim_search(city + ", Việt Nam")
            if result:
                return result[0], result[1], "city"
        return None, None, None

    # Strategy 1: Full address
    result = _nominatim_search(addr)
    if result:
        return result[0], result[1], "full"

    # Strategy 2: Simplified address
    simple = simplify_address(addr)
    if simple and simple != addr:
        result = _nominatim_search(simple)
        if result:
            return result[0], result[1], "simplified"

    # Strategy 3: City name only
    city = extract_city(addr, job_loc)
    if city:
        result = _nominatim_search(city + ", Việt Nam")
        if result:
            return result[0], result[1], "city"

    return None, None, None


def main():
    parser = argparse.ArgumentParser(description='Geocode retry for jobs without coordinates')
    parser.add_argument('--input', required=True, help='CSV file to process')
    parser.add_argument('--output', help='Output CSV (default: overwrite input)')
    parser.add_argument('--dry-run', action='store_true', help='Only show what would be geocoded')
    parser.add_argument('--limit', type=int, default=0, help='Max jobs to process (0=all)')
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding='utf-8-sig')
    output = args.output or args.input

    # Find rows without geocode
    mask = df['latitude'].isna() | (df['latitude'].astype(str).str.strip().isin(['', 'nan']))
    missing = df[mask]

    logger.info(f"Input: {args.input}")
    logger.info(f"Total: {len(df)} | Missing geocode: {len(missing)}")

    if args.dry_run:
        print(f"\nWould geocode {len(missing)} jobs. Sample addresses:")
        for i, (_, row) in enumerate(missing.head(20).iterrows()):
            addr = str(row.get('company_address', ''))[:60]
            simple = simplify_address(addr)[:60]
            print(f"  [{i+1}] {addr}")
            if simple != addr:
                print(f"       → {simple}")
        return

    stats = {"full": 0, "simplified": 0, "city": 0, "failed": 0}
    processed = 0

    for idx, row in missing.iterrows():
        if args.limit > 0 and processed >= args.limit:
            break

        title = str(row.get('title', ''))[:40]
        addr = str(row.get('company_address', ''))[:50]

        lat, lng, method = geocode_job(row)

        if lat is not None:
            df.at[idx, 'latitude'] = lat
            df.at[idx, 'longitude'] = lng
            stats[method] += 1
            logger.info(f"  [{processed+1}] ✓ {title} → {method} ({lat:.4f}, {lng:.4f})")
        else:
            stats["failed"] += 1
            logger.debug(f"  [{processed+1}] ✗ {title} ({addr})")

        processed += 1

        if processed % 50 == 0:
            logger.info(f"  Progress: {processed}/{len(missing)} | "
                        f"full={stats['full']} simplified={stats['simplified']} "
                        f"city={stats['city']} failed={stats['failed']}")

    # Save
    df.to_csv(output, index=False, encoding='utf-8-sig')

    total_success = stats['full'] + stats['simplified'] + stats['city']
    total_geo = df['latitude'].apply(lambda x: str(x).strip() not in ['', 'nan']).sum()

    logger.info("=" * 60)
    logger.info("  GEOCODE RETRY COMPLETE")
    logger.info(f"  Processed: {processed}")
    logger.info(f"  Success: {total_success} (full={stats['full']}, simplified={stats['simplified']}, city={stats['city']})")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Total geocoded: {total_geo}/{len(df)} ({total_geo/len(df)*100:.1f}%)")
    logger.info(f"  Output: {output}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
