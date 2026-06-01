"""
Crawl thêm ~10 jobs senior (4+ năm) cho mỗi ngành.
Sử dụng lại logic từ crawl_balanced_jobs.py nhưng chỉ crawl tầng senior.

Sử dụng:
    python src/crawl_senior_supplement.py --dry-run      # Xem URLs trước
    python src/crawl_senior_supplement.py                # Crawl
    python src/crawl_senior_supplement.py --categories it telesales  # Chỉ crawl 1 số ngành
"""
import sys
import os
import time
import csv
import logging
import argparse
import random
import hashlib
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crawl_topcv import (
    setup_driver, extract_job_simple, random_delay,
    is_valid_job_url
)
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ===== Constants (from crawl_balanced_jobs) =====
CATEGORIES = {
    'nhan-vien-kinh-doanh': {'name': 'Nhân viên kinh doanh', 'base_url': 'https://www.topcv.vn/tim-viec-lam-nhan-vien-kinh-doanh'},
    'ke-toan': {'name': 'Kế toán', 'base_url': 'https://www.topcv.vn/tim-viec-lam-ke-toan'},
    'marketing': {'name': 'Marketing', 'base_url': 'https://www.topcv.vn/tim-viec-lam-marketing'},
    'hanh-chinh-nhan-su': {'name': 'Hành chính nhân sự', 'base_url': 'https://www.topcv.vn/tim-viec-lam-hanh-chinh-nhan-su'},
    'cham-soc-khach-hang': {'name': 'Chăm sóc khách hàng', 'base_url': 'https://www.topcv.vn/tim-viec-lam-nhan-vien-cham-soc-khach-hang'},
    'ngan-hang': {'name': 'Ngân hàng', 'base_url': 'https://www.topcv.vn/tim-viec-lam-ngan-hang'},
    'it': {'name': 'IT', 'base_url': 'https://www.topcv.vn/viec-lam-it'},
    'lao-dong-pho-thong': {'name': 'Lao động phổ thông', 'base_url': 'https://www.topcv.vn/tim-viec-lam-lao-dong-pho-thong-cr1042'},
    'ky-su-xay-dung': {'name': 'Kỹ sư xây dựng', 'base_url': 'https://www.topcv.vn/tim-viec-lam-ky-su-xay-dung'},
    'thiet-ke': {'name': 'Thiết kế đồ họa', 'base_url': 'https://www.topcv.vn/tim-viec-lam-thiet-ke-do-hoa-designer'},
    'bat-dong-san': {'name': 'Bất động sản', 'base_url': 'https://www.topcv.vn/tim-viec-lam-bat-dong-san'},
    'giao-duc': {'name': 'Giáo dục', 'base_url': 'https://www.topcv.vn/tim-viec-lam-giao-duc'},
    'telesales': {'name': 'Telesales', 'base_url': 'https://www.topcv.vn/tim-viec-lam-nhan-vien-telesales'},
}

EXP_LABELS = {
    1: 'Không yêu cầu', 2: 'Dưới 1 năm', 3: '1 năm', 4: '2 năm',
    5: '3 năm', 6: '4 năm', 7: '5 năm', 8: 'Trên 5 năm',
}


def build_url(base_url, exp_level, page=1):
    params = f"?exp={exp_level}&sort=up_top&type_keyword=1&sba=1&saturday_status=0"
    if page > 1:
        params += f"&page={page}"
    return base_url + params


def append_jobs_to_csv(new_jobs, output_file):
    if not new_jobs:
        return
    file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0
    fieldnames = list(new_jobs[0].keys())
    with open(output_file, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_jobs)


def get_job_urls_from_page_simple(driver, page_url, wait_time=8):
    """Thu thập URLs từ 1 trang listing TopCV."""
    urls = []
    try:
        driver.get(page_url)
        try:
            WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'h3[class*="title"] a'))
            )
        except TimeoutException:
            return urls
        
        time.sleep(1.5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        job_links = soup.select('h3[class*="title"] a')
        
        for link in job_links:
            href = link.get('href', '')
            if href.startswith('/'):
                href = 'https://www.topcv.vn' + href
            if is_valid_job_url(href):
                urls.append(href)
    except Exception as e:
        logger.warning(f"Error getting URLs from {page_url}: {e}")
    
    return urls


def crawl_category_jobs(category_info, num_threads=2):
    """Crawl chi tiết jobs cho 1 category."""
    cat_name = category_info['category_name']
    urls = category_info['urls']
    
    if not urls:
        return []
    
    all_jobs = []
    jobs_lock = threading.Lock()
    
    def _worker(url_chunk, worker_id):
        local_jobs = []
        driver = None
        try:
            driver = setup_driver()
            driver.get("https://www.topcv.vn/")
            random_delay(2, 4)
            
            for i, url in enumerate(url_chunk):
                if i > 0 and i % 15 == 0:
                    try:
                        driver.quit()
                    except:
                        pass
                    random_delay(3, 5)
                    driver = setup_driver()
                    driver.get("https://www.topcv.vn/")
                    random_delay(2, 3)
                
                logger.info(f"  [W{worker_id}] [{i+1}/{len(url_chunk)}] {url[:70]}...")
                job = extract_job_simple(driver, url)
                if job:
                    local_jobs.append(job)
                    random_delay()
                else:
                    random_delay(1, 2)
        except Exception as e:
            logger.error(f"  [W{worker_id}] Error: {e}")
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
        
        with jobs_lock:
            all_jobs.extend(local_jobs)
        return len(local_jobs)
    
    # Split URLs
    chunks = [[] for _ in range(num_threads)]
    for i, url in enumerate(urls):
        chunks[i % num_threads].append(url)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for wid, chunk in enumerate(chunks):
            if chunk:
                futures.append(executor.submit(_worker, chunk, wid + 1))
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    return all_jobs

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    handlers=[
        logging.FileHandler('src/crawl_senior.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Senior: exp=6 (4 năm), exp=7 (5 năm), exp=8 (Trên 5 năm)
SENIOR_EXP_LEVELS = [6, 7, 8]
TARGET_PER_CATEGORY = 15  # Crawl 15, lấy 10 tốt nhất


def load_existing_hashes(*csv_files):
    """Load content_hash từ tất cả CSV files hiện có để loại trùng."""
    hashes = set()
    for f in csv_files:
        if not os.path.exists(f):
            continue
        try:
            with open(f, 'r', encoding='utf-8-sig') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    h = row.get('content_hash', '')
                    if h:
                        hashes.add(h)
        except Exception as e:
            logger.warning(f"Cannot read {f}: {e}")
    return hashes


def collect_senior_urls(driver, base_url, cat_name, max_pages=4):
    """Thu thập URLs cho tầng senior của 1 ngành (with retry on error)."""
    all_urls = set()
    
    for exp_level in SENIOR_EXP_LEVELS:
        exp_label = EXP_LABELS[exp_level]
        
        for page_num in range(1, max_pages + 1):
            page_url = build_url(base_url, exp_level, page_num)
            logger.info(f"  [{cat_name}] exp={exp_level} ({exp_label}) trang {page_num}")
            
            wait_time = 8 if page_num == 1 else 6
            
            # Retry up to 2 times on connection errors
            page_urls = []
            for attempt in range(3):
                try:
                    page_urls = get_job_urls_from_page_simple(driver, page_url, wait_time=wait_time)
                    break
                except Exception as e:
                    logger.warning(f"    Attempt {attempt+1} failed: {e}")
                    if attempt < 2:
                        time.sleep(5 + attempt * 5)
                        try:
                            driver.quit()
                        except:
                            pass
                        driver = setup_driver()
                        time.sleep(2)
            
            if len(page_urls) == 0:
                logger.info(f"    Trang trống → chuyển exp level tiếp theo")
                break
            
            new_urls = set(page_urls) - all_urls
            all_urls.update(page_urls)
            logger.info(f"    +{len(new_urls)} mới → tổng {len(all_urls)}")
            
            time.sleep(1.5)
    
    return list(all_urls), driver


def main():
    parser = argparse.ArgumentParser(description='Crawl thêm jobs senior cho mỗi ngành')
    parser.add_argument('--categories', nargs='+', default=None,
                        help='Category keys (mặc định: tất cả 13 ngành)')
    parser.add_argument('--max-pages', type=int, default=4,
                        help='Số trang tối đa mỗi exp level (mặc định: 4)')
    parser.add_argument('--threads', type=int, default=2,
                        help='Số threads crawl (mặc định: 2)')
    parser.add_argument('--target', type=int, default=15,
                        help='Số jobs cần crawl mỗi ngành (mặc định: 15)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Chỉ thu thập URLs, không crawl chi tiết')
    args = parser.parse_args()
    
    # Chọn categories
    if args.categories:
        selected_keys = [k for k in args.categories if k in CATEGORIES]
        invalid = [k for k in args.categories if k not in CATEGORIES]
        if invalid:
            print(f"Category không hợp lệ: {invalid}")
            print(f"Có thể chọn: {list(CATEGORIES.keys())}")
            return
    else:
        selected_keys = list(CATEGORIES.keys())
    
    # Load hashes hiện có để loại trùng
    existing_files = [
        'data/topcv_balanced_650_final.csv',
        'src/topcv_pipeline_20260424_223554.csv',
        'src/topcv_pipeline_20260427_170448.csv',
    ]
    existing_hashes = load_existing_hashes(*existing_files)
    logger.info(f"Loaded {len(existing_hashes)} existing hashes for dedup")
    
    print("=" * 60)
    print("CRAWL THÊM JOBS SENIOR (4+ NĂM)")
    print("=" * 60)
    print(f"Ngành:        {len(selected_keys)}")
    print(f"Target/ngành: {args.target}")
    print(f"Exp levels:   4 năm, 5 năm, Trên 5 năm")
    print(f"Max pages:    {args.max_pages}")
    print(f"Existing:     {len(existing_hashes)} jobs (dedup)")
    print("=" * 60)
    
    # Bước 1: Thu thập URLs
    print("\n[1/2] Thu thập URLs senior...")
    all_urls = {}
    
    for cat_key in selected_keys:
        cat = CATEGORIES[cat_key]
        cat_name = cat['name']
        base_url = cat['base_url']
        
        driver = setup_driver()
        try:
            urls, driver = collect_senior_urls(driver, base_url, cat_name, args.max_pages)
            all_urls[cat_key] = urls
            logger.info(f"  [{cat_name}] Tìm được {len(urls)} URLs senior")
        except Exception as e:
            logger.error(f"  [{cat_name}] URL collection failed: {e}")
            all_urls[cat_key] = []
        finally:
            try:
                driver.quit()
            except:
                pass
        
        time.sleep(3)
    
    # In tổng kết URLs
    print(f"\n{'Category':25s} | {'URLs':>6}")
    print("-" * 40)
    total_urls = 0
    for cat_key in selected_keys:
        cat_name = CATEGORIES[cat_key]['name']
        n = len(all_urls.get(cat_key, []))
        total_urls += n
        print(f"  {cat_name:23s} | {n:>6}")
    print(f"  {'TỔNG':23s} | {total_urls:>6}")
    
    if args.dry_run:
        print("\n[DRY-RUN] Dừng. Chạy lại không có --dry-run để crawl chi tiết.")
        return
    
    # Bước 2: Crawl chi tiết
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'src/topcv_senior_supplement_{timestamp}.csv'
    
    print(f"\n[2/2] Crawl chi tiết...")
    print(f"  Output: {output_file}")
    
    total_crawled = 0
    total_new = 0
    
    for cat_key in selected_keys:
        cat = CATEGORIES[cat_key]
        cat_name = cat['name']
        urls = all_urls.get(cat_key, [])
        
        if not urls:
            continue
        
        # Shuffle và lấy tối đa
        random.shuffle(urls)
        urls_to_crawl = urls[:args.target * 2]  # Crawl gấp đôi target để dự phòng
        
        logger.info(f"[{cat_name}] Crawl {len(urls_to_crawl)} URLs...")
        
        result = {
            'category_key': cat_key,
            'category_name': cat_name,
            'urls': urls_to_crawl
        }
        jobs = crawl_category_jobs(result, num_threads=args.threads)
        
        # Filter trùng
        new_jobs = []
        for job in jobs:
            job['category'] = cat_name
            h = job.get('content_hash', '')
            if h and h not in existing_hashes:
                new_jobs.append(job)
                existing_hashes.add(h)
        
        if new_jobs:
            append_jobs_to_csv(new_jobs, output_file)
        
        total_crawled += len(jobs)
        total_new += len(new_jobs)
        logger.info(f"[{cat_name}] Crawled: {len(jobs)}, New (unique): {len(new_jobs)}")
        
        time.sleep(3)
    
    print(f"\n{'='*60}")
    print(f"HOÀN THÀNH!")
    print(f"{'='*60}")
    print(f"  Crawled:    {total_crawled}")
    print(f"  New unique: {total_new}")
    print(f"  Output:     {output_file}")


if __name__ == '__main__':
    main()
