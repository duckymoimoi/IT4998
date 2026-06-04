"""
Scheduler: Crawl → LLM Clean → Geocode → Upsert ES (all-in-one pipeline)

Pipeline:
    Crawl workers (3 threads) → Queue → Clean worker (1 thread) → Geocode → Append CSV → Upsert ES

Sử dụng:
    # Chạy 1 lần (local)
    python scheduler.py --once --pages 3

    # Chạy qua Docker
    docker compose --profile crawl run scheduler

    # Chạy theo chu kỳ
    python scheduler.py --interval 60 --pages 5

    # Chỉ upsert file CSV có sẵn
    python scheduler.py --upsert-file topcv_jobs_cleaned.csv

    # Chỉ kiểm tra expired
    python scheduler.py --check-expired-only
"""

import sys
import os
import json
import logging
import hashlib
import time
import signal
import argparse
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import csv
import re
import requests as http_requests

# Suppress UC destructor error on Windows (WinError 6: handle invalid)
try:
    import undetected_chromedriver as _uc
    _orig_del = _uc.Chrome.__del__
    def _safe_del(self):
        try:
            _orig_del(self)
        except OSError:
            pass
    _uc.Chrome.__del__ = _safe_del
except Exception:
    pass

# ============= PATHS =============
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
LOG_DIR = SRC_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============= LOGGING =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [SCHEDULER] %(levelname)-8s %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'scheduler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============= CẤU HÌNH =============
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "topcv_jobs")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
CRAWL_HISTORY_FILE = LOG_DIR / "crawl_history.json"

# Nominatim geocoding (OSM — miễn phí)
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_geo_cache = {}


def _nominatim_geocode(address):
    """Geocode address -> {lat, lng} via OSM Nominatim. Cached."""
    if not address or address == 'nan':
        return None
    if address in _geo_cache:
        return _geo_cache[address]

    try:
        time.sleep(1.1)  # Respect Nominatim rate limit (1 req/s)
        resp = http_requests.get(
            _NOMINATIM_URL,
            params={"q": address, "format": "json", "limit": 1, "countrycodes": "vn"},
            headers={"User-Agent": "HUST-JobMatching-Thesis/1.0"},
            timeout=10,
        )
        if resp.status_code == 200 and resp.json():
            result = resp.json()[0]
            coords = {"lat": float(result["lat"]), "lng": float(result["lon"])}
            _geo_cache[address] = coords
            return coords
    except Exception as e:
        logger.warning(f"Nominatim geocode error for '{address[:50]}': {e}")

    _geo_cache[address] = None
    return None


# ============= CSV FIELDS =============
CSV_FIELDNAMES = [
    'title', 'url', 'company', 'company_address',
    'company_size', 'company_field',
    'job_salary', 'salary_min', 'salary_max', 'salary_type',
    'salary_note', 'has_commission',
    'job_location', 'experience',
    'education_level', 'education_field',
    'technical_skills', 'soft_skills', 'languages', 'certificates',
    'gender_requirement',
    'job_description', 'job_requirements', 'job_benefits', 'working_time',
    'latitude', 'longitude',
    'overview', 'content_hash', 'deadline', 'is_expired',
    'crawled_date', 'category',
]

# Trang listing tổng dùng cho crawl production.
# TopCV giữ phân trang dạng /viec-lam-tot-nhat?page=N.
PRODUCTION_LISTING_URL = "https://www.topcv.vn/viec-lam-tot-nhat"

# 13 ngành nghề — URLs listing theo category từ TopCV.
# Chỉ dùng khi cần crawl theo ngành, ví dụ phục vụ thực nghiệm/cân bằng dữ liệu.
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

# Lock để tránh race condition khi init UC chromedriver
_driver_init_lock = threading.Lock()


class CrawlScheduler:
    """Scheduler: Crawl → Clean → Geocode → Upsert ES"""

    def __init__(self, es_host=ES_HOST, index_name=ES_INDEX, use_embeddings=True, skip_es=False):
        self.es_host = es_host
        self.index_name = index_name
        self.running = True
        self.use_embeddings = use_embeddings
        self.embedding_service = None

        # Connect to ES
        if skip_es:
            logger.info("[SKIP] ES disabled (--no-es)")
            self.es = None
        else:
            try:
                self.es = Elasticsearch([es_host])
                if self.es.ping():
                    logger.info(f"[OK] ES connected: {es_host}")
                else:
                    logger.warning("[WARN] ES unavailable — CSV only mode")
                    self.es = None
            except Exception as e:
                logger.warning(f"[WARN] ES error: {e}")
                self.es = None

        # Init LLM Cleaner
        self.cleaner = None
        try:
            from job_matching.scoring.llm_cleaner import LLMCleaner
            self.cleaner = LLMCleaner(ollama_url=OLLAMA_URL)
            if self.cleaner._kiem_tra_ollama():
                logger.info("[OK] LLM Cleaner (Ollama) ready")
            else:
                logger.warning("[WARN] Ollama unavailable — raw data only")
                self.cleaner = None
        except Exception as e:
            logger.warning(f"[WARN] LLM Cleaner init error: {e}")
            self.cleaner = None

        # Init embedding service
        if self.use_embeddings:
            try:
                from job_matching.retrieval.embedding_service import get_embedding_service
                self.embedding_service = get_embedding_service()
                logger.info("[OK] Embedding service (bge-m3) ready")
            except Exception as e:
                logger.warning(f"[WARN] Embedding service unavailable: {e}")
                self.use_embeddings = False

    # ============= HISTORY =============
    def _load_history(self):
        if CRAWL_HISTORY_FILE.exists():
            with open(CRAWL_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"runs": []}

    def _save_history(self, history):
        with open(CRAWL_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def _log_run(self, stats):
        history = self._load_history()
        history["runs"].append({"timestamp": datetime.now().isoformat(), **stats})
        history["runs"] = history["runs"][-100:]
        self._save_history(history)

    # ============= GEOCODE =============
    def _geocode_job(self, job_data):
        """Geocode company_address → lat/lng"""
        address = job_data.get('company_address', '')
        if not address or address == 'nan':
            job_data['latitude'] = ''
            job_data['longitude'] = ''
            return job_data

        # Nếu nhiều chi nhánh (pipe-separated), lấy chi nhánh đầu
        first_addr = address.split(' | ')[0].strip() if ' | ' in address else address

        coords = _nominatim_geocode(first_addr)
        if coords:
            job_data['latitude'] = coords['lat']
            job_data['longitude'] = coords['lng']
            logger.debug(f"  Geocoded: {first_addr[:40]} → {coords['lat']:.4f}, {coords['lng']:.4f}")
        else:
            job_data['latitude'] = ''
            job_data['longitude'] = ''

        return job_data

    # ============= CLEAN + GEOCODE 1 JOB =============
    def _clean_and_geocode(self, raw_job):
        """LLM clean 1 job + geocode → cleaned dict"""
        if self.cleaner:
            cleaned = self.cleaner.clean_job(raw_job)
            if cleaned:
                cleaned = self._geocode_job(cleaned)
                return cleaned

        # Fallback: no clean, just pass through + geocode
        raw_job = self._geocode_job(raw_job)
        return raw_job

    # ============= COLLECT LISTING URLS =============
    def _build_listing_url(self, base_url, page_num):
        if page_num <= 1:
            return base_url
        separator = "&" if "?" in base_url else "?"
        return f"{base_url}{separator}page={page_num}"

    def _extract_listing_urls(self, page_source, seen_urls):
        """Extract valid TopCV job URLs from one listing page.

        Return both new URLs and the number of valid job links found. A page can
        contain jobs but still add zero new URLs if all of them were seen before.
        """
        from bs4 import BeautifulSoup
        from job_matching.crawling.crawl_topcv import is_valid_job_url

        soup = BeautifulSoup(page_source, "html.parser")

        # Main selector used by TopCV listing cards. The fallback catches minor
        # DOM changes where title links are no longer wrapped by h3.
        links = list(soup.select('h3[class*="title"] a[href]'))
        if not links:
            links = list(soup.select('a[href*="/viec-lam/"]'))

        page_urls = []
        valid_link_count = 0
        for link in links:
            href = link.get("href", "")
            if href.startswith("/"):
                href = "https://www.topcv.vn" + href
            href = href.split("?")[0]
            if not is_valid_job_url(href):
                continue
            valid_link_count += 1
            if href not in seen_urls:
                seen_urls.add(href)
                page_urls.append(href)
        return page_urls, valid_link_count

    def _load_listing_page(self, driver, page_url, selector, wait_seconds=20, retries=2):
        """Load one listing page with retry; return page source or None."""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        for attempt in range(1, retries + 1):
            try:
                driver.get(page_url)
                time.sleep(3)
                WebDriverWait(driver, wait_seconds).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                return driver.page_source
            except Exception as exc:
                logger.warning(
                    f"  Listing page load failed attempt {attempt}/{retries}: {page_url} ({exc})"
                )
                if attempt < retries:
                    time.sleep(5 * attempt)
                    try:
                        driver.refresh()
                    except Exception:
                        pass
        return None

    def _collect_general_urls(self, pages=3, base_url=PRODUCTION_LISTING_URL):
        """
        Thu thap URLs tu trang viec lam tong cho production.
        Returns: list of (url, category_name)
        """
        from job_matching.crawling.crawl_topcv import setup_driver

        url_category_pairs = []
        seen_urls = set()
        consecutive_failed_pages = 0
        consecutive_empty_pages = 0
        selector = 'h3[class*="title"] a, a[href*="/viec-lam/"]'

        with _driver_init_lock:
            driver = setup_driver()

        try:
            logger.info(f"  [GENERAL] Collecting URLs from {base_url}")
            for page_num in range(1, pages + 1):
                page_url = self._build_listing_url(base_url, page_num)
                page_source = self._load_listing_page(driver, page_url, selector)

                if not page_source:
                    consecutive_failed_pages += 1
                    logger.warning(f"  [GENERAL] Page {page_num} failed; skip ({consecutive_failed_pages}/3)")
                    if consecutive_failed_pages >= 3:
                        logger.warning("  [GENERAL] Stop collecting after 3 consecutive failed pages.")
                        break
                    continue

                consecutive_failed_pages = 0
                page_urls, valid_link_count = self._extract_listing_urls(page_source, seen_urls)
                for href in page_urls:
                    url_category_pairs.append((href, "Tong hop"))

                logger.info(
                    f"  [GENERAL] Page {page_num}: links={valid_link_count}, "
                    f"new={len(page_urls)} URLs (total: {len(url_category_pairs)})"
                )

                if valid_link_count == 0:
                    consecutive_empty_pages += 1
                    logger.warning(f"  [GENERAL] Page {page_num} has no valid job links ({consecutive_empty_pages}/3)")
                    if consecutive_empty_pages >= 3:
                        logger.warning("  [GENERAL] Stop collecting after 3 consecutive empty pages.")
                        break
                else:
                    consecutive_empty_pages = 0

                time.sleep(1)
        finally:
            driver.quit()

        return url_category_pairs

    def _collect_category_urls(self, pages=3, categories=None):
        """
        Thu thap URLs theo tung category.
        Returns: list of (url, category_name)
        """
        from job_matching.crawling.crawl_topcv import setup_driver

        selected = categories or list(CATEGORIES.keys())
        url_category_pairs = []  # [(url, category_name), ...]
        seen_urls = set()
        selector = 'h3[class*="title"] a, a[href*="/viec-lam/"]'

        with _driver_init_lock:
            driver = setup_driver()

        try:
            for cat_key in selected:
                cat = CATEGORIES.get(cat_key)
                if not cat:
                    logger.warning(f"Unknown category: {cat_key}")
                    continue

                cat_name = cat['name']
                base_url = cat['base_url']
                logger.info(f"  [{cat_name}] Collecting URLs...")
                consecutive_failed_pages = 0
                consecutive_empty_pages = 0

                for page_num in range(1, pages + 1):
                    page_url = self._build_listing_url(base_url, page_num)
                    page_source = self._load_listing_page(driver, page_url, selector)

                    if not page_source:
                        consecutive_failed_pages += 1
                        logger.warning(f"  [{cat_name}] Page {page_num} failed; skip ({consecutive_failed_pages}/3)")
                        if consecutive_failed_pages >= 3:
                            logger.warning(f"  [{cat_name}] Stop collecting after 3 consecutive failed pages.")
                            break
                        continue

                    consecutive_failed_pages = 0
                    page_urls, valid_link_count = self._extract_listing_urls(page_source, seen_urls)
                    page_count = len(page_urls)
                    for href in page_urls:
                        url_category_pairs.append((href, cat_name))

                    logger.info(
                        f"  [{cat_name}] Page {page_num}: links={valid_link_count}, "
                        f"new={page_count} URLs (total: {len(url_category_pairs)})"
                    )

                    if valid_link_count == 0:
                        consecutive_empty_pages += 1
                        logger.warning(f"  [{cat_name}] Page {page_num} has no valid job links ({consecutive_empty_pages}/3)")
                        if consecutive_empty_pages >= 3:
                            logger.warning(f"  [{cat_name}] Stop collecting after 3 consecutive empty pages.")
                            break
                    else:
                        consecutive_empty_pages = 0

                    time.sleep(1)
        finally:
            driver.quit()

        return url_category_pairs

    # ============= PRODUCER-CONSUMER PIPELINE =============
    def run_pipeline(
        self, pages=5, threads=3, output_file=None, categories=None,
        source="general", recrawl_after_days=7, force_recrawl_existing=False,
    ):
        """
        Full pipeline:
          Collect URLs → Crawl workers → queue → Clean → geocode → CSV
        """
        from job_matching.crawling.crawl_topcv import setup_driver, extract_job_simple

        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = str(SRC_DIR / f'topcv_pipeline_{timestamp}.csv')

        logger.info("=" * 70)
        logger.info(f"  PIPELINE START — source={source}, pages={pages}, threads={threads}")
        logger.info(f"  Categories: {categories or 'N/A'}")
        logger.info(f"  Output: {output_file}")
        logger.info(f"  LLM Clean: {'ON' if self.cleaner else 'OFF'}")
        logger.info("=" * 70)

        # Phase 1: Collect URLs
        if categories and source == "general":
            logger.info("[PHASE 1] Categories provided; switching source to category.")
            source = "category"

        if source == "category":
            logger.info("[PHASE 1] Collecting job URLs by category...")
            url_pairs = self._collect_category_urls(pages=pages, categories=categories)
        else:
            logger.info("[PHASE 1] Collecting job URLs from general listing...")
            url_pairs = self._collect_general_urls(pages=pages)

        if not url_pairs:
            logger.error("[ERROR] No URLs found")
            return None, {"status": "no_urls"}

        logger.info(f"[PHASE 1] Found {len(url_pairs)} URLs")
        url_pairs, precheck_stats = self._filter_existing_url_pairs(
            url_pairs,
            recrawl_after_days=recrawl_after_days,
            force_recrawl_existing=force_recrawl_existing,
        )

        if not url_pairs:
            logger.info("[PHASE 1] All URLs already exist and are fresh. Nothing to crawl.")
            return None, {"status": "all_existing_fresh", **precheck_stats}

        # Phase 2: Crawl → Queue → Clean → CSV
        logger.info(f"[PHASE 2] Crawl + Clean pipeline ({len(url_pairs)} jobs)")

        raw_queue = queue.Queue(maxsize=50)
        csv_lock = threading.Lock()
        stats = {
            "crawled": 0, "cleaned": 0, "failed_crawl": 0,
            "failed_clean": 0, "geocoded": 0,
            **precheck_stats,
        }
        done_crawling = threading.Event()

        # Init CSV
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
            writer.writeheader()

        # --- Clean consumer thread ---
        def _clean_consumer():
            while True:
                try:
                    raw_job = raw_queue.get(timeout=5)
                except queue.Empty:
                    if done_crawling.is_set():
                        break
                    continue

                if raw_job is None:  # Poison pill
                    break

                try:
                    title = str(raw_job.get('title', ''))[:50]
                    logger.info(f"  [CLEAN] {title}...")

                    cleaned = self._clean_and_geocode(raw_job)

                    if cleaned:
                        with csv_lock:
                            with open(output_file, 'a', newline='', encoding='utf-8-sig') as f:
                                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
                                writer.writerow(cleaned)
                            stats["cleaned"] += 1

                        if cleaned.get('latitude'):
                            stats["geocoded"] += 1

                        logger.info(f"  [CLEAN] ✓ {title}")
                    else:
                        stats["failed_clean"] += 1
                        logger.warning(f"  [CLEAN] ✗ {title}")
                except Exception as e:
                    stats["failed_clean"] += 1
                    logger.error(f"  [CLEAN] Error: {e}")
                finally:
                    raw_queue.task_done()

        clean_thread = threading.Thread(target=_clean_consumer, daemon=True)
        clean_thread.start()

        # --- Crawl producer threads ---
        import random

        def _crawl_producer(pair_chunk, worker_id):
            """pair_chunk: list of (url, category_name)"""
            driver = None
            try:
                # Stagger driver init to avoid UC race condition
                with _driver_init_lock:
                    driver = setup_driver()
                    time.sleep(1)  # Small delay between driver inits

                # Warm up
                try:
                    driver.get("https://www.topcv.vn/")
                    time.sleep(random.uniform(2, 4))
                except:
                    pass

                for i, (url, cat_name) in enumerate(pair_chunk):
                    if not self.running:
                        break

                    # Restart browser mỗi 20 jobs
                    if i > 0 and i % 20 == 0:
                        logger.info(f"  [W{worker_id}] Restart browser (batch {i // 20 + 1})")
                        try:
                            driver.quit()
                        except:
                            pass
                        time.sleep(random.uniform(3, 6))
                        with _driver_init_lock:
                            driver = setup_driver()
                        try:
                            driver.get("https://www.topcv.vn/")
                            time.sleep(random.uniform(2, 3))
                        except:
                            pass

                    logger.info(f"  [W{worker_id}] [{i+1}/{len(pair_chunk)}] [{cat_name}] Crawling...")
                    job_data = extract_job_simple(driver, url)

                    if job_data:
                        job_data['category'] = cat_name  # Gán category từ listing URL
                        raw_queue.put(job_data)
                        stats["crawled"] += 1
                        time.sleep(random.uniform(3, 7))
                    else:
                        stats["failed_crawl"] += 1
                        time.sleep(random.uniform(1, 2))

            except Exception as e:
                logger.error(f"  [W{worker_id}] Error: {e}")
            finally:
                if driver:
                    try:
                        driver.quit()
                    except:
                        pass

        # Split URL pairs across workers
        chunks = [[] for _ in range(threads)]
        for i, pair in enumerate(url_pairs):
            chunks[i % threads].append(pair)

        crawl_threads = []
        for wid, chunk in enumerate(chunks):
            if chunk:
                t = threading.Thread(target=_crawl_producer, args=(chunk, wid + 1))
                t.start()
                crawl_threads.append(t)

        # Wait for crawl to finish
        for t in crawl_threads:
            t.join()

        done_crawling.set()
        raw_queue.put(None)  # Poison pill
        clean_thread.join(timeout=300)  # Wait max 5 min for clean to finish

        logger.info("=" * 70)
        logger.info("  PIPELINE COMPLETE")
        logger.info(f"  Crawled: {stats['crawled']} | Cleaned: {stats['cleaned']} | Geocoded: {stats['geocoded']}")
        logger.info(f"  Failed crawl: {stats['failed_crawl']} | Failed clean: {stats['failed_clean']}")
        logger.info(f"  Output: {output_file}")
        logger.info("=" * 70)

        return output_file, stats

    # ============= UPSERT TO ES =============
    def _url_to_doc_id(self, url):
        return hashlib.md5(url.encode('utf-8')).hexdigest()

    def _should_recrawl_existing(self, source, recrawl_after_days):
        """Return True if an existing ES doc should be crawled again."""
        if recrawl_after_days is None or recrawl_after_days < 0:
            return False

        last_crawled = source.get("last_crawled") or source.get("crawled_date")
        if not last_crawled:
            return True

        try:
            value = str(last_crawled).strip().replace("Z", "+00:00")
            crawled_at = datetime.fromisoformat(value)
            if crawled_at.tzinfo is not None:
                crawled_at = crawled_at.replace(tzinfo=None)
            return datetime.now() - crawled_at >= timedelta(days=recrawl_after_days)
        except Exception:
            return True

    def _filter_existing_url_pairs(self, url_pairs, recrawl_after_days=7, force_recrawl_existing=False):
        """
        Skip URLs already present in ES when they were crawled recently.

        Existing docs older than recrawl_after_days are kept so the pipeline can
        detect changed content_hash and update the document.
        """
        stats = {"input": len(url_pairs), "skipped_existing": 0, "recrawl_existing": 0}
        if not self.es or force_recrawl_existing or not url_pairs:
            return url_pairs, stats

        filtered = []
        try:
            ids = [self._url_to_doc_id(url) for url, _ in url_pairs]
            existing_by_id = {}
            for start in range(0, len(ids), 500):
                chunk_ids = ids[start:start + 500]
                response = self.es.mget(
                    index=self.index_name,
                    body={"ids": chunk_ids},
                    _source=["url", "title", "content_hash", "last_crawled", "crawled_date"],
                )
                for doc in response.get("docs", []):
                    if doc.get("found"):
                        existing_by_id[doc["_id"]] = doc.get("_source", {})

            for url, cat_name in url_pairs:
                doc_id = self._url_to_doc_id(url)
                existing = existing_by_id.get(doc_id)
                if not existing:
                    filtered.append((url, cat_name))
                    continue

                if self._should_recrawl_existing(existing, recrawl_after_days):
                    filtered.append((url, cat_name))
                    stats["recrawl_existing"] += 1
                else:
                    stats["skipped_existing"] += 1

            logger.info(
                "[PRECHECK] URLs: input=%s, skipped_existing=%s, recrawl_existing=%s, remaining=%s",
                stats["input"], stats["skipped_existing"], stats["recrawl_existing"], len(filtered),
            )
            return filtered, stats
        except Exception as e:
            logger.warning(f"[PRECHECK] Cannot check existing URLs, crawl all URLs: {e}")
            return url_pairs, stats

    def upsert_to_es(self, csv_file):
        """Upsert cleaned CSV vào ES"""
        if not self.es:
            logger.warning("[WARN] ES unavailable — skip upsert")
            return {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}

        logger.info(f"[UPSERT] {csv_file}")
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        logger.info(f"   Records: {len(df):,}")

        # Embeddings
        all_embeddings = None
        if self.use_embeddings and self.embedding_service:
            logger.info("[EMBED] Generating embeddings (bge-m3)...")
            embed_start = time.time()
            embed_texts = []
            for _, row in df.iterrows():
                text = self.embedding_service.build_job_text(row.to_dict())
                embed_texts.append(text)
            all_embeddings = self.embedding_service.encode(embed_texts, batch_size=32, show_progress=True)
            logger.info(f"[OK] {len(all_embeddings)} embeddings in {time.time()-embed_start:.1f}s")

        stats = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}
        actions = []

        for idx, (_, row) in enumerate(df.iterrows()):
            url = str(row.get("url", ""))
            if not url or url == "nan":
                continue

            doc_id = self._url_to_doc_id(url)
            new_hash = str(row.get("content_hash", ""))

            # Build document
            doc = {}
            for col in df.columns:
                val = row.get(col, "")
                if pd.isna(val) or val is None:
                    doc[col] = ""
                else:
                    doc[col] = str(val).strip()

            # Boolean fields
            for bool_field in ['is_expired', 'has_commission']:
                val = row.get(bool_field, False)
                if isinstance(val, str):
                    val = val.lower() in ("true", "1", "yes")
                doc[bool_field] = bool(val) if not pd.isna(val) else False

            # Numeric fields
            for num_field in ['salary_min', 'salary_max', 'latitude', 'longitude']:
                val = row.get(num_field)
                if val and not pd.isna(val):
                    try:
                        fval = float(val)
                        # salary: convert VND to triệu
                        if num_field.startswith('salary') and fval > 1000:
                            fval = fval / 1_000_000
                        doc[num_field] = fval
                    except (ValueError, TypeError):
                        pass

            # Geo coordinates
            lat = doc.get('latitude')
            lng = doc.get('longitude')
            if lat and lng and isinstance(lat, float) and isinstance(lng, float):
                doc['geo_coordinates'] = [{"lat": lat, "lng": lng, "address": doc.get('company_address', '')}]

            doc["last_crawled"] = datetime.now().isoformat()

            # Embedding
            if all_embeddings is not None:
                doc["embedding"] = all_embeddings[idx].tolist()

            # Check existing
            try:
                existing = self.es.get(index=self.index_name, id=doc_id, ignore=[404])
                if existing and existing.get('found'):
                    old_hash = existing['_source'].get('content_hash', '')
                    if old_hash == new_hash:
                        existing_source = existing.get('_source', {})
                        update_doc = {"last_crawled": doc["last_crawled"]}
                        if all_embeddings is not None and not existing_source.get("embedding"):
                            update_doc["embedding"] = doc["embedding"]
                            stats["updated"] += 1
                        else:
                            stats["unchanged"] += 1
                        actions.append({
                            "_op_type": "update", "_index": self.index_name,
                            "_id": doc_id, "doc": update_doc
                        })
                        continue
                    else:
                        stats["updated"] += 1
                        actions.append({
                            "_op_type": "update", "_index": self.index_name,
                            "_id": doc_id, "doc": doc
                        })
                else:
                    stats["new"] += 1
                    actions.append({
                        "_op_type": "index", "_index": self.index_name,
                        "_id": doc_id, "_source": doc
                    })
            except Exception:
                stats["new"] += 1
                actions.append({
                    "_op_type": "index", "_index": self.index_name,
                    "_id": doc_id, "_source": doc
                })

            if len(actions) >= 500:
                try:
                    success, errors = bulk(self.es, actions, raise_on_error=False)
                    if errors:
                        stats["errors"] += len(errors)
                except Exception as e:
                    logger.error(f"  Bulk error: {e}")
                    stats["errors"] += len(actions)
                actions = []

        if actions:
            try:
                success, errors = bulk(self.es, actions, raise_on_error=False)
                if errors:
                    stats["errors"] += len(errors)
            except Exception as e:
                logger.error(f"  Bulk error: {e}")
                stats["errors"] += len(actions)

        logger.info(f"[UPSERT] Done: {stats}")
        return stats

    # ============= CHECK EXPIRED =============
    def check_expired_jobs(self):
        if not self.es:
            return 0

        logger.info("[CHECK] Checking expired jobs...")
        try:
            result = self.es.search(
                index=self.index_name,
                body={
                    "query": {"bool": {
                        "must": [{"exists": {"field": "deadline"}}],
                        "must_not": [{"term": {"is_expired": True}}]
                    }},
                    "size": 1000,
                    "_source": ["deadline", "url", "title"]
                }
            )

            actions = []
            now = datetime.now()
            for hit in result['hits']['hits']:
                deadline_str = hit['_source'].get('deadline', '')
                if not deadline_str:
                    continue
                try:
                    deadline_date = datetime.strptime(deadline_str, '%d/%m/%Y')
                    if deadline_date < now:
                        actions.append({
                            "_op_type": "update", "_index": self.index_name,
                            "_id": hit['_id'], "doc": {"is_expired": True}
                        })
                except ValueError:
                    pass

            if actions:
                bulk(self.es, actions, raise_on_error=False)

            logger.info(f"[OK] Marked {len(actions)} expired jobs")
            return len(actions)
        except Exception as e:
            logger.error(f"[ERROR] Check expired: {e}")
            return 0

    # ============= ENSURE INDEX =============
    def ensure_index(self):
        if not self.es:
            return
        try:
            if not self.es.indices.exists(index=self.index_name):
                logger.info(f"[INDEX] Creating: {self.index_name}")
                from job_matching.ingestion.import_to_elastic import ElasticImporter
                importer = ElasticImporter(es_host=self.es_host)
                importer.create_index(index_name=self.index_name, force_recreate=False)
            else:
                count = self.es.count(index=self.index_name)['count']
                logger.info(f"[INDEX] Exists: {self.index_name} ({count:,} docs)")
        except Exception as e:
            logger.error(f"[ERROR] Ensure index: {e}")

    # ============= FULL CYCLE =============
    def run_cycle(
        self, pages=5, threads=3, categories=None, source="general",
        recrawl_after_days=7, force_recrawl_existing=False, **kwargs,
    ):
        """1 chu kỳ: crawl → clean → geocode → upsert → check expired"""
        start_time = datetime.now()
        logger.info("=" * 70)
        logger.info(f"[CYCLE] START — {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)

        stats = {"start_time": start_time.isoformat(), "pages": pages, "threads": threads, "source": source}

        # 1. Ensure index
        self.ensure_index()

        # 2. Pipeline: crawl + clean + geocode
        output_file, pipeline_stats = self.run_pipeline(
            pages, threads, categories=categories, source=source,
            recrawl_after_days=recrawl_after_days,
            force_recrawl_existing=force_recrawl_existing,
        )
        stats["pipeline"] = pipeline_stats

        if not output_file:
            stats["status"] = pipeline_stats.get("status", "pipeline_failed")
            self._log_run(stats)
            return stats

        # 3. Upsert to ES
        upsert_stats = self.upsert_to_es(output_file)
        stats["upsert"] = upsert_stats

        # 4. Check expired
        expired = self.check_expired_jobs()
        stats["expired_marked"] = expired

        # 5. Done
        duration = (datetime.now() - start_time).total_seconds()
        stats["duration_seconds"] = round(duration, 1)
        stats["status"] = "success"
        self._log_run(stats)

        logger.info("=" * 70)
        logger.info(f"[OK] CYCLE DONE — {duration:.0f}s")
        logger.info(f"   New: {upsert_stats['new']} | Updated: {upsert_stats['updated']} | Expired: {expired}")
        logger.info("=" * 70)
        return stats

    # ============= PERIODIC =============
    def run_periodic(self, interval_minutes=60, **kwargs):
        logger.info(f"[TIMER] Running every {interval_minutes} min (Ctrl+C to stop)")

        def signal_handler(sig, frame):
            logger.info("\n[STOP] Stopping scheduler...")
            self.running = False
        signal.signal(signal.SIGINT, signal_handler)

        cycle_count = 0
        while self.running:
            cycle_count += 1
            logger.info(f"\n[CYCLE #{cycle_count}]")
            try:
                self.run_cycle(**kwargs)
            except Exception as e:
                logger.error(f"[ERROR] Cycle #{cycle_count}: {e}")

            if not self.running:
                break

            logger.info(f"[WAIT] {interval_minutes} min until next cycle...")
            for _ in range(interval_minutes * 60):
                if not self.running:
                    break
                time.sleep(1)

        logger.info("[STOP] Scheduler stopped.")


def main():
    parser = argparse.ArgumentParser(
        description='TopCV Pipeline — Crawl → Clean → Geocode → Upsert ES',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python scheduler.py --once --pages 3
  python scheduler.py --interval 60 --pages 5
  python scheduler.py --check-expired-only
  python scheduler.py --upsert-file topcv_jobs_cleaned.csv
        """
    )

    parser.add_argument('--once', action='store_true', help='Run once')
    parser.add_argument('--interval', type=int, default=60, help='Cycle interval (minutes)')
    parser.add_argument('--pages', type=int, default=5, help='Listing pages to crawl')
    parser.add_argument('--threads', type=int, default=3, help='Crawl threads')
    parser.add_argument(
        '--source',
        choices=['general', 'category'],
        default=os.getenv("CRAWL_SOURCE", "general"),
        help='URL source: general uses https://www.topcv.vn/viec-lam-tot-nhat?page=N; category uses predefined category pages',
    )
    parser.add_argument('--categories', nargs='+', help='Category keys to crawl (e.g. it marketing ke-toan)')
    parser.add_argument(
        '--recrawl-after-days',
        type=int,
        default=int(os.getenv("CRAWL_RECHECK_DAYS", "7")),
        help='Skip existing URLs crawled within this many days. Use -1 to skip all existing URLs.',
    )
    parser.add_argument(
        '--force-recrawl-existing',
        action='store_true',
        help='Crawl existing URLs again even if they were crawled recently.',
    )
    parser.add_argument('--check-expired-only', action='store_true')
    parser.add_argument('--upsert-file', type=str, help='Upsert existing CSV')
    parser.add_argument('--es-host', type=str, default=ES_HOST)
    parser.add_argument('--no-embedding', action='store_true', help='Skip embedding generation')
    parser.add_argument('--no-es', action='store_true', help='Skip ES entirely (crawl + clean + CSV only)')

    args = parser.parse_args()

    scheduler = CrawlScheduler(
        es_host=args.es_host,
        use_embeddings=not args.no_embedding,
        skip_es=args.no_es,
    )

    if args.check_expired_only:
        count = scheduler.check_expired_jobs()
        print(f"Marked {count} expired jobs")
        return

    if args.upsert_file:
        scheduler.ensure_index()
        stats = scheduler.upsert_to_es(args.upsert_file)
        print(f"Upsert: {stats}")
        return

    if args.once:
        scheduler.run_cycle(
            pages=args.pages,
            threads=args.threads,
            categories=args.categories,
            source=args.source,
            recrawl_after_days=args.recrawl_after_days,
            force_recrawl_existing=args.force_recrawl_existing,
        )
    else:
        scheduler.run_periodic(
            interval_minutes=args.interval,
            pages=args.pages,
            threads=args.threads,
            categories=args.categories,
            source=args.source,
            recrawl_after_days=args.recrawl_after_days,
            force_recrawl_existing=args.force_recrawl_existing,
        )


if __name__ == '__main__':
    main()
