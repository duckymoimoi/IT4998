"""
Crawl bộ 1950 jobs cân bằng từ TopCV: 13 ngành x 150 jobs/ngành x 3 tầng kinh nghiệm.

Mapping exp param TopCV:
  exp=1: Không yêu cầu (KYC)
  exp=2: Dưới 1 năm
  exp=3: 1 năm
  exp=4: 2 năm
  exp=5: 3 năm
  exp=6: 4 năm
  exp=7: 5 năm
  exp=8: Trên 5 năm

3 tầng kinh nghiệm (50 jobs/tầng/ngành):
  Junior  (exp=1,2)  : KYC + Dưới 1 năm  → 50 jobs
  Mid     (exp=3,4,5): 1-3 năm           → 50 jobs
  Senior  (exp=6,7,8): 4+ năm            → 50 jobs

Sử dụng:
    python src/crawl_balanced_jobs.py --dry-run           # Xem URLs, không crawl
    python src/crawl_balanced_jobs.py                     # Crawl toàn bộ
    python src/crawl_balanced_jobs.py --categories it     # Crawl 1 ngành
    python src/crawl_balanced_jobs.py --max-pages 5       # Tăng số trang
"""
import os
import time
import csv
import json
import logging
import argparse
import random
import re
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from job_matching.crawling.crawl_topcv import setup_driver, extract_job_simple, random_delay
from job_matching.crawling.crawl_senior_supplement import get_job_urls_from_page_simple
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from job_matching.scoring.llm_cleaner import LLMCleaner

# Khởi tạo LLM Cleaner toàn cục để sử dụng trong các worker
llm_cleaner = LLMCleaner()

# ============= CẤU HÌNH =============

# 3 tầng kinh nghiệm, mỗi tầng gồm các exp levels và target số lượng
TIERS = {
    'junior': {
        'label': 'Junior (0-1 năm)',
        'exp_levels': [1, 2],
        'target_per_category': 50,
    },
    'mid': {
        'label': 'Mid (1-3 năm)',
        'exp_levels': [3, 4, 5],
        'target_per_category': 50,
    },
    'senior': {
        'label': 'Senior (3-5+ năm)',
        'exp_levels': [6, 7, 8],
        'target_per_category': 50,
    },
}

EXP_LABELS = {
    1: 'Không yêu cầu',
    2: 'Dưới 1 năm',
    3: '1 năm',
    4: '2 năm',
    5: '3 năm',
    6: '4 năm',
    7: '5 năm',
    8: 'Trên 5 năm',
}

# 13 ngành (loại bỏ "Senior" vì không phải ngành nghề, chỉ là mức kinh nghiệm)
CATEGORIES = {
    'nhan-vien-kinh-doanh': {
        'name': 'Nhân viên kinh doanh',
        'base_url': 'https://www.topcv.vn/tim-viec-lam-nhan-vien-kinh-doanh',
    },
    'ke-toan': {
        'name': 'Kế toán',
        'base_url': 'https://www.topcv.vn/tim-viec-lam-ke-toan',
    },
    'marketing': {
        'name': 'Marketing',
        'base_url': 'https://www.topcv.vn/tim-viec-lam-marketing',
    },
    'hanh-chinh-nhan-su': {
        'name': 'Hành chính nhân sự',
        'base_url': 'https://www.topcv.vn/tim-viec-lam-hanh-chinh-nhan-su',
    },
    'cham-soc-khach-hang': {
        'name': 'Chăm sóc khách hàng',
        'base_url': 'https://www.topcv.vn/tim-viec-lam-nhan-vien-cham-soc-khach-hang',
    },
    'ngan-hang': {
        'name': 'Ngân hàng',
        'base_url': 'https://www.topcv.vn/tim-viec-lam-ngan-hang',
    },
    'it': {
        'name': 'IT',
        'base_url': 'https://www.topcv.vn/tim-viec-lam-cong-nghe-thong-tin-cr257',
        'category_family': 'r257',
    },
    'lao-dong-pho-thong': {
        'name': 'Lao động phổ thông',
        'base_url': 'https://www.topcv.vn/tim-viec-lam-lao-dong-pho-thong-cr1042',
    },
    'ky-su-xay-dung': {
        'name': 'Kỹ sư xây dựng',
        'base_url': 'https://www.topcv.vn/tim-viec-lam-ky-su-xay-dung',
    },
    'thiet-ke': {
        'name': 'Thiết kế đồ họa',
        'base_url': 'https://www.topcv.vn/tim-viec-lam-thiet-ke-do-hoa-designer',
    },
    'bat-dong-san': {
        'name': 'Bất động sản',
        'base_url': 'https://www.topcv.vn/tim-viec-lam-bat-dong-san',
    },
    'giao-duc': {
        'name': 'Giáo dục',
        'base_url': 'https://www.topcv.vn/tim-viec-lam-giao-duc',
    },
    'telesales': {
        'name': 'Telesales',
        'base_url': 'https://www.topcv.vn/tim-viec-lam-nhan-vien-telesales',
    },
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    handlers=[
        logging.FileHandler('src/crawl_balanced.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def canonicalize_job_url(url):
    """Normalize TopCV job URL by dropping tracking query parameters."""
    if not url:
        return ''
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))


def get_job_dedup_key(job):
    """Prefer stable content_hash, fallback to canonical job URL."""
    return job.get('content_hash') or canonicalize_job_url(job.get('url', ''))


def build_url(base_url, exp_level, page=1):
    """Xây URL với filter kinh nghiệm"""
    params = f"?exp={exp_level}&type_keyword=1&sba=1&saturday_status=0"
    for cat in CATEGORIES.values():
        if cat.get('base_url') == base_url and cat.get('category_family'):
            params += f"&category_family={cat['category_family']}"
            break
    if page > 1:
        params += f"&page={page}"
    return base_url + params


def collect_urls_for_tier(driver, base_url, cat_name, tier_key, max_pages=3):
    """
    Thu thập URLs cho 1 ngành + 1 tầng kinh nghiệm.
    Luôn crawl TẤT CẢ exp levels để đảm bảo đa dạng.
    Lấy tất cả URLs (dự phòng lỗi), giới hạn crawl chi tiết ở bước sau.
    """
    tier = TIERS[tier_key]
    exp_levels = tier['exp_levels']

    urls_by_exp = {exp_level: [] for exp_level in exp_levels}
    all_urls_set = set()

    for exp_level in exp_levels:
        exp_label = EXP_LABELS[exp_level]

        for page_num in range(1, max_pages + 1):
            page_url = build_url(base_url, exp_level, page_num)
            logger.info(f"  [{cat_name}] {tier_key} exp={exp_level} ({exp_label}) trang {page_num}")

            wait_time = 8 if page_num == 1 else 6
            page_urls = get_job_urls_from_page_simple(driver, page_url, wait_time=wait_time)

            if len(page_urls) == 0:
                logger.info(f"    Trang trống → chuyển exp level tiếp theo")
                break

            for u in page_urls:
                if u not in all_urls_set:
                    all_urls_set.add(u)
                    urls_by_exp[exp_level].append(u)
                    
            logger.info(f"    + mớí → tổng exp={exp_level} có {len(urls_by_exp[exp_level])} URLs")
            time.sleep(1.5)

    # Interleave (trộn xen kẽ đều đặn) các URLs từ các exp_levels khác nhau
    import itertools
    interleaved_urls = []
    for items in itertools.zip_longest(*urls_by_exp.values()):
        for item in items:
            if item is not None and item not in interleaved_urls:
                interleaved_urls.append(item)
                
    return interleaved_urls


def collect_urls_round_robin_for_category(
    driver,
    base_url,
    cat_name,
    tier_remaining,
    max_pages=3,
    existing_urls=None,
    wait_time=12,
    delay_min=8,
    delay_max=18,
):
    """
    Thu thap URL theo vong cho ca 3 tier trong cung mot category.

    Cach nay tranh viec gom het URL cua junior truoc, bi block, roi mid/senior
    khong con co hoi. Thu tu moi vong:
      page 1: junior exp..., mid exp..., senior exp...
      page 2: junior exp..., mid exp..., senior exp...
    """
    existing_urls = existing_urls or set()
    seen_urls = {canonicalize_job_url(url) for url in existing_urls if url}
    collected_urls = []
    url_summary = defaultdict(int)
    empty_exp_pages = set()
    empty_rounds = 0

    active_tiers = [tier_key for tier_key in TIERS if tier_remaining.get(tier_key, 0) > 0]
    logger.info(
        f"  [{cat_name}] Thu thap URL round-robin cho tiers con thieu: "
        + ", ".join(f"{tier}={tier_remaining.get(tier, 0)}" for tier in active_tiers)
    )

    for page_num in range(1, max_pages + 1):
        logger.info(f"  [{cat_name}] === Vong URL page={page_num}/{max_pages} ===")
        new_in_round = 0

        for tier_key in active_tiers:
            for exp_level in TIERS[tier_key]['exp_levels']:
                exp_page_key = (exp_level, page_num)
                if exp_page_key in empty_exp_pages:
                    continue

                exp_label = EXP_LABELS[exp_level]
                page_url = build_url(base_url, exp_level, page_num)
                logger.info(f"  [{cat_name}] {tier_key} exp={exp_level} ({exp_label}) trang {page_num}")

                page_urls = get_job_urls_from_page_simple(driver, page_url, wait_time=wait_time)
                if not page_urls:
                    empty_exp_pages.add(exp_page_key)
                    logger.info("    Trang trong hoac bi chan -> tiep tuc nhom tiep theo")
                    random_delay(delay_min, delay_max)
                    continue

                found = len(page_urls)
                added = 0
                for url in page_urls:
                    canonical_url = canonicalize_job_url(url)
                    if canonical_url in seen_urls:
                        continue
                    seen_urls.add(canonical_url)
                    collected_urls.append(url)
                    url_summary[tier_key] += 1
                    added += 1

                new_in_round += added
                duplicate = found - added
                logger.info(
                    f"    found={found}, new={added}, duplicate={duplicate} "
                    f"-> tong tam thoi {len(collected_urls)} URLs"
                )
                random_delay(delay_min, delay_max)

        if new_in_round == 0:
            empty_rounds += 1
            logger.warning(f"  [{cat_name}] Vong page={page_num} khong co URL moi ({empty_rounds}/2)")
            if empty_rounds >= 2:
                logger.warning(f"  [{cat_name}] Dung thu URL som vi 2 vong lien tiep khong co URL moi.")
                break
        else:
            empty_rounds = 0

    return collected_urls, url_summary


def classify_job_tier(job):
    """Phân loại job vào tier dựa trên trường experience"""
    exp = job.get('experience', '').lower()

    if 'không yêu cầu' in exp or 'chưa' in exp or exp == '':
        return 'junior'
    elif 'dưới 1' in exp or 'under' in exp:
        return 'junior'
    elif '1 năm' in exp and '10' not in exp:
        return 'mid'
    elif '2 năm' in exp:
        return 'mid'
    elif '3 năm' in exp:
        return 'mid'
    elif '4 năm' in exp or '5 năm' in exp or 'trên 5' in exp or 'over' in exp:
        return 'senior'
    else:
        # Fallback: thử parse số
        import re
        nums = re.findall(r'(\d+)', exp)
        if nums:
            y = int(nums[0])
            if y <= 1:
                return 'junior'
            elif y <= 3:
                return 'mid'
            else:
                return 'senior'
        return 'junior'  # default


def get_job_keywords(job):
    """Trích xuất keywords từ job để so sánh diversity"""
    parts = []
    for field in ['title', 'technical_skills', 'specializations', 'requirements_tags']:
        val = job.get(field, '')
        if val:
            parts.append(str(val).lower())
    text = ' '.join(parts)
    # Tách thành set từ
    words = set(re.split(r'[,\s/\-\(\)]+', text))
    words.discard('')
    return words


def jaccard_distance(set_a, set_b):
    """Khoảng cách Jaccard (0 = giống hệt, 1 = hoàn toàn khác)"""
    if not set_a and not set_b:
        return 0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0
    return 1 - (intersection / union)


def select_diverse_jobs(jobs, target, min_diversity=0.3):
    """
    Chọn jobs đa dạng nhất bằng greedy selection.
    Mỗi bước chọn job có khoảng cách lớn nhất so với các job đã chọn.

    Args:
        jobs: list of job dicts
        target: số jobs cần chọn
        min_diversity: ngưỡng Jaccard tối thiểu (0.3 = ít nhất 30% khác biệt)

    Returns:
        list of selected jobs
    """
    if len(jobs) <= target:
        return jobs

    # Tính keywords cho mỗi job
    job_keywords = [(job, get_job_keywords(job)) for job in jobs]
    random.shuffle(job_keywords)

    # Greedy: bắt đầu với job đầu tiên
    selected = [job_keywords[0]]
    remaining = job_keywords[1:]

    while len(selected) < target and remaining:
        best_job = None
        best_score = -1
        best_idx = -1

        for i, (job, kw) in enumerate(remaining):
            # Tính min distance tới tất cả jobs đã chọn
            min_dist = min(jaccard_distance(kw, sel_kw) for _, sel_kw in selected)

            if min_dist > best_score:
                best_score = min_dist
                best_job = (job, kw)
                best_idx = i

        # Nếu job tốt nhất vẫn quá giống (< min_diversity), vẫn lấy
        if best_job:
            selected.append(best_job)
            remaining.pop(best_idx)

    return [job for job, kw in selected]


def is_valid_job_data(job_data):
    """Kiểm tra xem job có đủ các trường dữ liệu chuẩn từ HTML parsing hay không."""
    if not job_data:
        return False
        
    title = str(job_data.get('title', '')).strip()
    if not title:
        return False
        
    overview = str(job_data.get('overview', ''))
    job_details = str(job_data.get('job_details', ''))
    
    if len(overview) < 50 or len(job_details) < 50:
        return False
        
    # Trích xuất tạm experience để classify
    import re
    exp_match = re.search(r'Kinh nghiệm[:\s]*\n?\s*(.+?)(?:\n|$)', overview)
    if exp_match:
        job_data['experience'] = exp_match.group(1).strip()
    else:
        # Không tìm thấy thông tin kinh nghiệm -> loại
        return False
        
    # Check JD sections
    def trich_section(text, tieu_de):
        pattern = r'===\s*' + re.escape(tieu_de) + r'\s*===\s*\n(.*?)(?:\n===|$)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ''
        
    jd = trich_section(job_details, 'MÔ TẢ CÔNG VIỆC')
    req = trich_section(job_details, 'YÊU CẦU ỨNG VIÊN')
    
    # Độ dài tối thiểu cho mô tả công việc (tránh các job crawl bị lỗi layout)
    if len(jd) < 30 or len(req) < 10:
        return False
        
    return True


def crawl_tier_with_target(urls, cat_key, cat_name, tier_key, target, num_threads, incremental_file=None):
    """Crawl jobs for a specific tier until the target is reached."""
    if not urls:
        return []
        
    all_jobs = []
    jobs_lock = threading.Lock()
    
    def _worker(url_chunk, worker_id):
        local_jobs = []
        driver = None
        try:
            driver = setup_driver()
            try:
                driver.get("https://www.topcv.vn/")
                random_delay(2, 4)
            except:
                pass
                
            for i, url in enumerate(url_chunk):
                with jobs_lock:
                    if len(all_jobs) >= target:
                        break  # Đã đủ target thì dừng
                        
                # Restart browser định kỳ để tránh memory leak
                if i > 0 and i % 15 == 0:
                    try: driver.quit()
                    except: pass
                    random_delay(3, 5)
                    driver = setup_driver()
                    try:
                        driver.get("https://www.topcv.vn/")
                        random_delay(2, 3)
                    except: pass
                
                logger.info(f"  [W{worker_id}] [{cat_name} - {tier_key}] Crawling {url[:60]}...")
                job = extract_job_simple(driver, url)
                
                # CHỈ CHẤP NHẬN JOB HỢP LỆ (ĐẦY ĐỦ THÔNG TIN CƠ BẢN)
                if is_valid_job_data(job):
                    # Tiến hành Clean bằng LLM ngay lập tức
                    logger.info(f"  [W{worker_id}] Đang làm sạch bằng LLM: {url[:60]}")
                    cleaned_job = llm_cleaner.clean_job(job)
                    
                    if cleaned_job:
                        # Kiểm tra xem job SAU KHI CLEAN có thực sự thuộc tier này không
                        actual_tier = classify_job_tier(cleaned_job)
                        if actual_tier == tier_key:
                            with jobs_lock:
                                if len(all_jobs) < target:
                                    cleaned_job['category_key'] = cat_key
                                    cleaned_job['category'] = cat_name
                                    
                                    all_jobs.append(cleaned_job)
                                    local_jobs.append(cleaned_job)
                                    # LƯU NGAY LẬP TỨC MỖI KHI CÓ JOB HỢP LỆ
                                    if incremental_file:
                                        append_jobs_to_csv([cleaned_job], incremental_file)
                                    logger.info(f"  [W{worker_id}] --> Hợp lệ & Đã Clean! ({len(all_jobs)}/{target} jobs cho {tier_key})")
                                else:
                                    break
                        else:
                            logger.info(f"  [W{worker_id}] --> Bỏ qua: Job thuộc tầng {actual_tier}, không phải {tier_key}")
                        random_delay(1, 2)
                    else:
                        logger.warning(f"  [W{worker_id}] --> Bỏ qua: LLM clean thất bại")
                        random_delay(1, 2)
                else:
                    logger.info(f"  [W{worker_id}] --> Bỏ qua: Dữ liệu raw không đầy đủ hoặc bị lỗi format")
                    random_delay(1, 2)
                    
        except Exception as e:
            logger.error(f"  [W{worker_id}] Error: {e}")
        finally:
            if driver:
                try: driver.quit()
                except: pass
                
        return len(local_jobs)

    # Chia URLs cho các worker
    chunks = [[] for _ in range(num_threads)]
    for i, url in enumerate(urls):
        chunks[i % num_threads].append(url)
        
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for wid, chunk in enumerate(chunks):
            if chunk:
                futures.append(executor.submit(_worker, chunk, wid + 1))
        for f in as_completed(futures):
            try: f.result()
            except: pass
            
    return all_jobs


def crawl_urls_fill_targets(
    urls,
    cat_key,
    cat_name,
    tier_remaining,
    num_threads,
    incremental_file=None,
    existing_dedup_keys=None,
    existing_canonical_urls=None,
    delay_min=4,
    delay_max=9,
):
    """
    Crawl danh sach URL chung va dua job vao tier that sau khi doc JD.

    Khac voi crawl_tier_with_target(), ham nay khong bo job chi vi URL den tu
    filter kinh nghiem khac. Neu job thuc te thuoc tier nao con thieu thi van
    duoc giu lai.
    """
    if not urls:
        return []

    all_jobs = []
    jobs_lock = threading.Lock()
    remaining = dict(tier_remaining)
    seen_dedup_keys = set(existing_dedup_keys or set())
    seen_canonical_urls = set(existing_canonical_urls or set())

    def _has_remaining():
        return any(v > 0 for v in remaining.values())

    def _worker(url_chunk, worker_id):
        local_jobs = []
        driver = None
        try:
            driver = setup_driver()
            try:
                driver.get("https://www.topcv.vn/")
                random_delay(delay_min, delay_max)
            except:
                pass

            for i, url in enumerate(url_chunk):
                with jobs_lock:
                    if not _has_remaining():
                        break

                if i > 0 and i % 10 == 0:
                    try:
                        driver.quit()
                    except:
                        pass
                    random_delay(delay_min, delay_max)
                    driver = setup_driver()
                    try:
                        driver.get("https://www.topcv.vn/")
                        random_delay(delay_min, delay_max)
                    except:
                        pass

                logger.info(f"  [W{worker_id}] [{cat_name}] Crawling {url[:70]}...")
                job = extract_job_simple(driver, url)

                if not is_valid_job_data(job):
                    logger.info(f"  [W{worker_id}] --> Bo qua: du lieu raw khong day du hoac loi format")
                    random_delay(delay_min, delay_max)
                    continue

                logger.info(f"  [W{worker_id}] Dang lam sach bang LLM: {url[:70]}")
                cleaned_job = llm_cleaner.clean_job(job)
                if not cleaned_job:
                    logger.warning(f"  [W{worker_id}] --> Bo qua: LLM clean that bai")
                    random_delay(delay_min, delay_max)
                    continue

                actual_tier = classify_job_tier(cleaned_job)
                dedup_key = get_job_dedup_key(cleaned_job)
                canonical_url = canonicalize_job_url(cleaned_job.get('url', ''))
                with jobs_lock:
                    if dedup_key in seen_dedup_keys or canonical_url in seen_canonical_urls:
                        logger.info(f"  [W{worker_id}] --> Bo qua: job trung lap")
                    elif remaining.get(actual_tier, 0) <= 0:
                        logger.info(f"  [W{worker_id}] --> Bo qua: tier {actual_tier} da du")
                    else:
                        cleaned_job['category_key'] = cat_key
                        cleaned_job['category'] = cat_name

                        all_jobs.append(cleaned_job)
                        local_jobs.append(cleaned_job)
                        remaining[actual_tier] -= 1
                        if dedup_key:
                            seen_dedup_keys.add(dedup_key)
                        if canonical_url:
                            seen_canonical_urls.add(canonical_url)

                        if incremental_file:
                            append_jobs_to_csv([cleaned_job], incremental_file)

                        accepted = sum(tier_remaining.values()) - sum(max(v, 0) for v in remaining.values())
                        logger.info(
                            f"  [W{worker_id}] --> Hop le tier={actual_tier}. "
                            f"Con thieu: junior={remaining.get('junior', 0)}, "
                            f"mid={remaining.get('mid', 0)}, senior={remaining.get('senior', 0)} "
                            f"(accepted={accepted})"
                        )

                random_delay(delay_min, delay_max)

        except Exception as e:
            logger.error(f"  [W{worker_id}] Error: {e}")
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

        return len(local_jobs)

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
            except:
                pass

    return all_jobs


def save_to_csv(all_jobs, output_file):
    """Luu ket qua ra CSV (ghi de toan bo file)"""
    if not all_jobs:
        logger.warning("Khong co jobs de luu!")
        return

    fieldnames = list(all_jobs[0].keys())
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_jobs)

    logger.info(f"Da luu {len(all_jobs)} jobs -> {output_file}")


def append_jobs_to_csv(new_jobs, output_file):
    """
    Append thêm jobs vào file CSV đã có.
    Nếu file chưa tồn tại → tạo mới với header.
    Nếu file đã có → chỉ append rows (không ghi lại header).
    """
    if not new_jobs:
        return

    file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0
    fieldnames = list(new_jobs[0].keys())

    with open(output_file, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_jobs)

    logger.info(f"Da append {len(new_jobs)} jobs -> {output_file} (tong file)")


def load_existing_jobs(csv_file):
    """
    Load jobs đã crawl từ file CSV (dùng để resume/kiểm tra trùng lặp).
    Trả về list of dicts.
    """
    if not os.path.exists(csv_file):
        return []
    try:
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            jobs = list(reader)
        logger.info(f"Da load {len(jobs)} jobs tu {csv_file}")
        return jobs
    except Exception as e:
        logger.warning(f"Khong doc duoc {csv_file}: {e}")
        return []


def print_summary(summary):
    """In bang tong hop"""
    print("\n" + "=" * 80)
    print(f"{'NGANH':<30} {'Junior(0-1)':>12} {'Mid(1-3)':>12} {'Senior(3-5+)':>12} {'TONG':>10}")
    print("-" * 80)

    total_junior = total_mid = total_senior = 0

    for cat_key in sorted(CATEGORIES.keys()):
        if cat_key not in summary:
            continue
        cat_name = CATEGORIES[cat_key]['name']
        j = summary[cat_key].get('junior', 0)
        m = summary[cat_key].get('mid', 0)
        s = summary[cat_key].get('senior', 0)
        t = j + m + s
        total_junior += j
        total_mid += m
        total_senior += s
        print(f"  {cat_name:<28} {j:>12} {m:>12} {s:>12} {t:>10}")

    total = total_junior + total_mid + total_senior
    print("-" * 80)
    print(f"  {'TONG':<28} {total_junior:>12} {total_mid:>12} {total_senior:>12} {total:>10}")
    print(f"  {'MUC TIEU':<28} {'650':>12} {'650':>12} {'650':>12} {'1950':>10}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Crawl 700 jobs can bang tu TopCV')
    parser.add_argument('--categories', nargs='+', default=None,
                        help='Category keys (mac dinh: tat ca 14 nganh)')
    parser.add_argument('--tiers', nargs='+', choices=list(TIERS.keys()), default=None,
                        help='Chi crawl mot so tier, vi du: --tiers senior')
    parser.add_argument('--max-pages', type=int, default=2,
                        help='So trang toi da moi exp level (mac dinh: 2)')
    parser.add_argument('--threads', type=int, default=2,
                        help='So threads crawl (mac dinh: 2)')
    parser.add_argument('--target-per-tier', type=int, default=None,
                        help='So job muc tieu moi tier moi nganh; mac dinh dung cau hinh TIERS')
    parser.add_argument('--url-delay-min', type=float, default=8,
                        help='Delay toi thieu giua cac trang listing khi gom URL (mac dinh: 8s)')
    parser.add_argument('--url-delay-max', type=float, default=18,
                        help='Delay toi da giua cac trang listing khi gom URL (mac dinh: 18s)')
    parser.add_argument('--crawl-delay-min', type=float, default=4,
                        help='Delay toi thieu giua cac trang chi tiet job (mac dinh: 4s)')
    parser.add_argument('--crawl-delay-max', type=float, default=9,
                        help='Delay toi da giua cac trang chi tiet job (mac dinh: 9s)')
    parser.add_argument('--page-wait', type=float, default=12,
                        help='Thoi gian cho selector tren trang listing (mac dinh: 12s)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Chi thu thap URLs, khong crawl chi tiet')
    parser.add_argument('--output', type=str, default=None,
                        help='File output CSV')
    parser.add_argument('--exclude-csv', nargs='*', default=[],
                        help='CSV files chi dung de loai trung URL/content_hash, khong tinh vao target resume')
    args = parser.parse_args()

    if args.target_per_tier is not None:
        for tier in TIERS.values():
            tier['target_per_category'] = args.target_per_tier
    selected_tiers = args.tiers or list(TIERS.keys())

    # Chon categories
    if args.categories:
        selected_keys = [k for k in args.categories if k in CATEGORIES]
        invalid = [k for k in args.categories if k not in CATEGORIES]
        if invalid:
            print(f"Category khong hop le: {invalid}")
            print(f"Co the chon: {list(CATEGORIES.keys())}")
            return
    else:
        selected_keys = list(CATEGORIES.keys())

    target_total = sum(t['target_per_category'] for t in TIERS.values()) * len(selected_keys)

    print("=" * 60)
    print("CRAWL BO JOBS CAN BANG TU TOPCV")
    print("=" * 60)
    print(f"Nganh:        {len(selected_keys)} / {len(CATEGORIES)}")
    print(f"Tang:         3 (Junior 0-1 / Mid 1-3 / Senior 3-5+)")
    print(f"Jobs/nganh:   150 (50+50+50)")
    print(f"Muc tieu:     {target_total} jobs")
    print(f"Max pages:    {args.max_pages}")
    print(f"Dry-run:      {args.dry_run}")
    if args.exclude_csv:
        print(f"Exclude CSV:   {', '.join(args.exclude_csv)}")
    print("=" * 60)

    # Tạo hoặc tìm file incremental để lưu dần
    data_dir = str(Path(__file__).resolve().parents[3] / 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    if args.output:
        incremental_file = args.output
    else:
        import glob
        existing_files = glob.glob(os.path.join(data_dir, 'topcv_balanced_1950_*.csv'))
        if existing_files:
            incremental_file = max(existing_files, key=os.path.getmtime)
            print(f"[INFO] Tự động tiếp tục từ file gần nhất: {os.path.basename(incremental_file)}")
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            incremental_file = os.path.join(data_dir, f'topcv_balanced_1950_{timestamp}.csv')

    print(f"\n[1/2] Bắt đầu Crawl đan xen (Thu thập URLs -> Crawl chi tiết ngay lập tức)")
    print(f"  File incremental: {incremental_file}\n")

    all_jobs = []
    existing_urls = set()
    existing_canonical_urls = set()
    existing_dedup_keys = set()
    category_tier_counts = defaultdict(lambda: defaultdict(int))
    url_summary = defaultdict(lambda: defaultdict(int))

    # Load jobs đã có trong file incremental (nếu resume)
    if os.path.exists(incremental_file):
        existing_jobs = load_existing_jobs(incremental_file)
        all_jobs.extend(existing_jobs)
        existing_urls = set(j.get('url', '') for j in existing_jobs if j.get('url'))
        existing_canonical_urls = {canonicalize_job_url(url) for url in existing_urls if url}
        existing_dedup_keys = {get_job_dedup_key(j) for j in existing_jobs if get_job_dedup_key(j)}
        
        for job in existing_jobs:
            cat = job.get('category_key')
            if not cat:
                cat_name = job.get('category', '')
                for k, v in CATEGORIES.items():
                    if v['name'] == cat_name:
                        cat = k
                        break
            tier = classify_job_tier(job)
            if cat:
                category_tier_counts[cat][tier] += 1
                
        if existing_jobs:
            logger.info(f"Resume: Đã load {len(existing_jobs)} jobs từ lần chạy trước.")

    # Load corpus cu chi de tranh crawl trung khi bo sung.
    # Cac file nay khong duoc tinh vao target resume cua output hien tai.
    excluded_count = 0
    for exclude_file in args.exclude_csv:
        excluded_jobs = load_existing_jobs(exclude_file)
        excluded_count += len(excluded_jobs)
        for job in excluded_jobs:
            url = job.get('url', '')
            if url:
                existing_urls.add(url)
                existing_canonical_urls.add(canonicalize_job_url(url))
            dedup_key = get_job_dedup_key(job)
            if dedup_key:
                existing_dedup_keys.add(dedup_key)
    if excluded_count:
        logger.info(f"Exclude: Da load {excluded_count} jobs tu {len(args.exclude_csv)} CSV de tranh crawl trung corpus cu.")

    total_urls = 0

    for cat_key in selected_keys:
        cat = CATEGORIES[cat_key]
        cat_name = cat['name']
        base_url = cat['base_url']

        logger.info(f"\n{'='*60}\n[{cat_name}] Bat dau xu ly round-robin URL -> crawl chi tiet\n{'='*60}")

        tier_remaining = {}
        for tier_key in selected_tiers:
            target = TIERS[tier_key]['target_per_category']
            current_count = category_tier_counts[cat_key][tier_key]
            remaining = target - current_count

            if remaining <= 0:
                logger.info(f"  [{cat_name}] {tier_key}: Da du {target} jobs (hien co {current_count}), bo qua.")
            else:
                logger.info(f"  [{cat_name}] {tier_key}: Can crawl them {remaining} jobs (da co {current_count}).")
                tier_remaining[tier_key] = remaining

        if not tier_remaining:
            continue

        logger.info(f"  [{cat_name}] Khoi dong browser thu URL theo vong junior/mid/senior...")
        driver = setup_driver()
        try:
            candidate_urls, cat_url_summary = collect_urls_round_robin_for_category(
                driver=driver,
                base_url=base_url,
                cat_name=cat_name,
                tier_remaining=tier_remaining,
                max_pages=args.max_pages,
                existing_urls=existing_urls,
                wait_time=args.page_wait,
                delay_min=args.url_delay_min,
                delay_max=args.url_delay_max,
            )
        finally:
            try:
                driver.quit()
            except:
                pass

        for tier_key, count in cat_url_summary.items():
            url_summary[cat_key][tier_key] += count
            total_urls += count

        if not candidate_urls:
            logger.warning(f"  [{cat_name}]: Khong thu thap duoc URL moi nao (co the do limit/block).")
            continue

        logger.info(f"  [{cat_name}]: Thu duoc {len(candidate_urls)} URL moi theo round-robin.")

        if args.dry_run:
            logger.info("  [DRY-RUN] Bo qua buoc crawl chi tiet.")
            continue

        logger.info(f"  [{cat_name}]: Bat dau crawl chi tiet {len(candidate_urls)} URLs voi {args.threads} threads...")
        jobs = crawl_urls_fill_targets(
            urls=candidate_urls,
            cat_key=cat_key,
            cat_name=cat_name,
            tier_remaining=tier_remaining,
            num_threads=args.threads,
            incremental_file=incremental_file,
            existing_dedup_keys=existing_dedup_keys,
            existing_canonical_urls=existing_canonical_urls,
            delay_min=args.crawl_delay_min,
            delay_max=args.crawl_delay_max,
        )

        for job in jobs:
            tier = classify_job_tier(job)
            category_tier_counts[cat_key][tier] += 1
            url = job.get('url', '')
            existing_urls.add(url)
            existing_canonical_urls.add(canonicalize_job_url(url))
            dedup_key = get_job_dedup_key(job)
            if dedup_key:
                existing_dedup_keys.add(dedup_key)

        all_jobs.extend(jobs)
        continue

        logger.info(f"\n{'='*60}\n[{cat_name}] Bắt đầu xử lý (Thu thập URLs -> Crawl chi tiết)\n{'='*60}")

        for tier_key in selected_tiers:
            target = TIERS[tier_key]['target_per_category']
            current_count = category_tier_counts[cat_key][tier_key]
            remaining = target - current_count
            
            if remaining <= 0:
                logger.info(f"  [{cat_name}] {tier_key}: Đã đủ {target} jobs (hiện có {current_count}), bỏ qua.")
                continue

            logger.info(f"  [{cat_name}] {tier_key}: Cần crawl thêm {remaining} jobs (đã có {current_count}). Khởi động browser tìm URLs...")

            # 1. Thu thập URLs cho riêng tầng này
            driver = setup_driver()
            try:
                tier_urls = collect_urls_for_tier(driver, base_url, cat_name, tier_key, args.max_pages)
            finally:
                try: driver.quit()
                except: pass
                
            if not tier_urls:
                logger.warning(f"  [{cat_name}] {tier_key}: Không thu thập được URL nào (có thể do limit)!")
                continue

            url_summary[cat_key][tier_key] = len(tier_urls)
            total_urls += len(tier_urls)

            if args.dry_run:
                logger.info(f"  [DRY-RUN] Thu thập được {len(tier_urls)} URLs. Bỏ qua bước crawl chi tiết.")
                continue
                
            # Lọc URLs đã crawl mà VẪN GIỮ NGUYÊN thứ tự xen kẽ (interleaved)
            tier_urls = [u for u in tier_urls if u not in existing_urls]
            
            if not tier_urls:
                logger.warning(f"  [{cat_name}] {tier_key}: Các URLs lấy được đều đã crawl trước đó!")
                continue
                
            # 2. Crawl chi tiết ngay lập tức
            logger.info(f"  [{cat_name}] {tier_key}: Bắt đầu crawl chi tiết {len(tier_urls)} URLs với {args.threads} threads...")
            jobs = crawl_tier_with_target(tier_urls, cat_key, cat_name, tier_key, remaining, args.threads, incremental_file)
            
            # Cập nhật cache
            for job in jobs:
                category_tier_counts[cat_key][tier_key] += 1
                existing_urls.add(job.get('url', ''))
            
            all_jobs.extend(jobs)
            
    print_summary(url_summary)
    print(f"\n  Tong URLs thu thap: {total_urls}")
    print(f"  Tong jobs crawl duoc: {len(all_jobs)}")
    print(f"  Du lieu da luu tai: {incremental_file}")

    # Thống kê kết quả
    selection_summary = defaultdict(lambda: defaultdict(int))
    for job in all_jobs:
        cat_key = job.get('category_key')
        if not cat_key:
            cat_name = job.get('category', '')
            for k, v in CATEGORIES.items():
                if v['name'] == cat_name:
                    cat_key = k
                    break
        tier = classify_job_tier(job)
        if cat_key:
            selection_summary[cat_key][tier] += 1

    print_summary(selection_summary)

    # Stats cuoi
    from collections import Counter
    cat_counts = Counter(j.get('category', '?') for j in all_jobs)
    print("\n" + "=" * 60)
    print("KET QUA")
    print("=" * 60)
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat:<30} {cnt:>4} jobs")
    print(f"  {'TONG':<30} {len(all_jobs):>4} jobs (crawled)")
    print(f"\nFile chinh:   {incremental_file}")


if __name__ == '__main__':
    main()
