"""
Scheduler: Crawl → LLM Clean → Upsert ES (all-in-one pipeline)

Pipeline:
    Crawl workers (3 threads) → Queue → Clean worker (1 thread) → Append CSV → Upsert ES

Sử dụng:
    # Chạy 1 lần (local)
    python scheduler.py --once --pages 3

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
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan
import csv

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
JOBS_DIR = PROJECT_ROOT / "data" / "jobs"
LOG_DIR = SRC_DIR / "logs"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
JOBS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

try:
    from dotenv import load_dotenv
    load_dotenv(SRC_DIR / ".env")
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

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
ES_INDEX = os.getenv("ES_INDEX", "topcv_jobs_production")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
CRAWL_HISTORY_FILE = LOG_DIR / "crawl_history.json"
AUTO_CLASSIFY_TAXONOMY = os.getenv("AUTO_CLASSIFY_TAXONOMY", "1").strip().lower() in {
    "1", "true", "yes", "on",
}
TAXONOMY_CLASSIFY_THRESHOLD = int(os.getenv("TAXONOMY_CLASSIFY_THRESHOLD", "50"))
TAXONOMY_BATCH_SIZE = int(os.getenv("TAXONOMY_BATCH_SIZE", "20"))
TAXONOMY_SLEEP_SEC = float(os.getenv("TAXONOMY_SLEEP_SEC", "10"))
TAXONOMY_MODEL = os.getenv("TAXONOMY_MODEL", "openai/gpt-oss-120b")

# ============= CSV FIELDS =============
CSV_FIELDNAMES = [
    'title', 'url', 'company', 'company_address',
    'company_size',
    'job_salary', 'salary_min', 'salary_max', 'salary_type',
    'salary_note', 'has_commission',
    'job_location', 'work_location_text', 'experience',
    'education_level', 'education_field',
    'requirements_tags', 'specializations',
    'technical_skills', 'languages', 'certificates',
    'gender_requirement',
    'job_description', 'job_requirements', 'job_benefits', 'working_time',
    'overview', 'content_hash', 'deadline', 'is_expired',
    'crawled_date',
]

# Trang listing tổng dùng cho crawl production.
# TopCV giữ phân trang dạng /viec-lam-tot-nhat?page=N.
PRODUCTION_LISTING_URL = "https://www.topcv.vn/viec-lam-tot-nhat"

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
        self.semantic_profile_builder = None

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
                from job_matching.enrichment.semantic_job_profile import SemanticJobProfileBuilder
                self.embedding_service = get_embedding_service()
                self.semantic_profile_builder = SemanticJobProfileBuilder()
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

    # ============= CLEAN 1 JOB =============
    def _clean_job(self, raw_job):
        """LLM clean one job. Runtime distance uses Goong on demand."""
        if self.cleaner:
            cleaned = self.cleaner.clean_job(raw_job)
            if cleaned:
                self._collect_pending_taxonomy_terms(cleaned)
                return cleaned

        # Fallback: no clean, pass through while preserving location text.
        self._collect_pending_taxonomy_terms(raw_job)
        return raw_job

    def _collect_pending_taxonomy_terms(self, job):
        """Queue unknown crawl terms for later offline taxonomy classification."""
        try:
            from job_matching.enrichment.semantic_job_profile import append_pending_terms
            count = append_pending_terms(job)
            if count:
                logger.info("[TAXONOMY] Queued %s unknown terms", count)
        except Exception as exc:
            logger.warning("[TAXONOMY] Cannot queue unknown terms: %s", exc)

    def _backfill_taxonomy_terms(self, terms, batch_size=32):
        """Refresh semantic profiles for existing jobs affected by new taxonomy terms."""
        if not self.es or not self.embedding_service or not self.semantic_profile_builder:
            return {"targeted": 0, "updated": 0, "skipped": "embedding unavailable"}

        from job_matching.enrichment.build_term_taxonomy import split_terms

        target_terms = {
            str(term).strip().lower()
            for term in terms
            if str(term).strip()
        }
        if not target_terms:
            return {"targeted": 0, "updated": 0}

        source_fields = [
            "title", "technical_skills", "certificates", "languages",
            "job_description", "job_requirements",
        ]
        pending = []
        targeted = 0
        updated = 0

        def flush():
            nonlocal pending, updated
            if not pending:
                return
            profiles = [
                self.semantic_profile_builder.build(hit["_source"], include_searchable_fields=True)
                for hit in pending
            ]
            vectors = self.embedding_service.encode(
                [profile["semantic_text"] for profile in profiles],
                batch_size=batch_size,
                show_progress=False,
            )
            actions = []
            for hit, profile, vector in zip(pending, profiles, vectors):
                profile["embedding"] = vector.tolist()
                actions.append({
                    "_op_type": "update",
                    "_index": self.index_name,
                    "_id": hit["_id"],
                    "doc": profile,
                })
            success, errors = bulk(self.es, actions, raise_on_error=False)
            updated += success
            if errors:
                logger.warning("[TAXONOMY] Backfill errors: %s", len(errors))
            pending = []

        for hit in scan(
            self.es,
            index=self.index_name,
            query={
                "query": {"exists": {"field": "technical_skills"}},
                "_source": source_fields,
            },
            size=200,
            request_timeout=180,
        ):
            job_terms = {
                term.lower()
                for term in split_terms(hit["_source"].get("technical_skills", ""))
            }
            if not job_terms.intersection(target_terms):
                continue
            targeted += 1
            pending.append(hit)
            if len(pending) >= batch_size:
                flush()
        flush()
        self.es.indices.refresh(index=self.index_name)
        logger.info("[TAXONOMY] Backfilled %s/%s affected jobs", updated, targeted)
        return {"targeted": targeted, "updated": updated}

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

    def _collect_job_urls(self, pages=3, base_url=PRODUCTION_LISTING_URL):
        """
        Thu thap URLs tu trang viec lam tong cho production.
        Returns: list of job URLs (deduplicated).
        """
        from job_matching.crawling.crawl_topcv import setup_driver

        urls = []
        seen_urls = set()
        consecutive_failed_pages = 0
        consecutive_empty_pages = 0
        selector = 'h3[class*="title"] a, a[href*="/viec-lam/"]'

        with _driver_init_lock:
            driver = setup_driver()

        try:
            logger.info(f"  Collecting URLs from {base_url}")
            for page_num in range(1, pages + 1):
                page_url = self._build_listing_url(base_url, page_num)
                page_source = self._load_listing_page(driver, page_url, selector)

                if not page_source:
                    consecutive_failed_pages += 1
                    logger.warning(f"  Page {page_num} failed; skip ({consecutive_failed_pages}/3)")
                    if consecutive_failed_pages >= 3:
                        logger.warning("  Stop collecting after 3 consecutive failed pages.")
                        break
                    continue

                consecutive_failed_pages = 0
                page_urls, valid_link_count = self._extract_listing_urls(page_source, seen_urls)
                urls.extend(page_urls)

                logger.info(
                    f"  Page {page_num}: links={valid_link_count}, "
                    f"new={len(page_urls)} URLs (total: {len(urls)})"
                )

                if valid_link_count == 0:
                    consecutive_empty_pages += 1
                    logger.warning(f"  Page {page_num} has no valid job links ({consecutive_empty_pages}/3)")
                    if consecutive_empty_pages >= 3:
                        logger.warning("  Stop collecting after 3 consecutive empty pages.")
                        break
                else:
                    consecutive_empty_pages = 0

                time.sleep(1)
        finally:
            driver.quit()

        return urls

    # ============= PRODUCER-CONSUMER PIPELINE =============
    def run_pipeline(
        self, pages=5, threads=3, output_file=None,
        recrawl_after_days=7, force_recrawl_existing=False,
    ):
        """
        Full pipeline:
          Collect URLs → Crawl workers → queue → Clean → CSV
        """
        from job_matching.crawling.crawl_topcv import setup_driver, extract_job_simple

        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = str(JOBS_DIR / f'topcv_pipeline_{timestamp}.csv')

        logger.info("=" * 70)
        logger.info(f"  PIPELINE START — pages={pages}, threads={threads}")
        logger.info(f"  Output: {output_file}")
        logger.info(f"  LLM Clean: {'ON' if self.cleaner else 'OFF'}")
        logger.info("=" * 70)

        # Phase 1: Collect URLs
        logger.info("[PHASE 1] Collecting job URLs from general listing...")
        urls = self._collect_job_urls(pages=pages)

        if not urls:
            logger.error("[ERROR] No URLs found")
            return None, {"status": "no_urls"}

        logger.info(f"[PHASE 1] Found {len(urls)} URLs")
        urls, precheck_stats = self._filter_existing_urls(
            urls,
            recrawl_after_days=recrawl_after_days,
            force_recrawl_existing=force_recrawl_existing,
        )

        if not urls:
            logger.info("[PHASE 1] All URLs already exist and are fresh. Nothing to crawl.")
            return None, {"status": "all_existing_fresh", **precheck_stats}

        # Phase 2: Crawl → Queue → Clean → CSV
        logger.info(f"[PHASE 2] Crawl + Clean pipeline ({len(urls)} jobs)")

        raw_queue = queue.Queue(maxsize=50)
        csv_lock = threading.Lock()
        stats = {
            "crawled": 0, "cleaned": 0, "failed_crawl": 0,
            "failed_clean": 0,
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

                    cleaned = self._clean_job(raw_job)

                    if cleaned:
                        with csv_lock:
                            with open(output_file, 'a', newline='', encoding='utf-8-sig') as f:
                                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
                                writer.writerow(cleaned)
                            stats["cleaned"] += 1

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
        def _crawl_producer(url_chunk, worker_id):
            """url_chunk: list of job URLs"""
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
                except Exception:
                    pass

                for i, url in enumerate(url_chunk):
                    if not self.running:
                        break

                    # Restart browser mỗi 20 jobs
                    if i > 0 and i % 20 == 0:
                        logger.info(f"  [W{worker_id}] Restart browser (batch {i // 20 + 1})")
                        try:
                            driver.quit()
                        except Exception:
                            pass
                        time.sleep(random.uniform(3, 6))
                        with _driver_init_lock:
                            driver = setup_driver()
                        try:
                            driver.get("https://www.topcv.vn/")
                            time.sleep(random.uniform(2, 3))
                        except Exception:
                            pass

                    logger.info(f"  [W{worker_id}] [{i+1}/{len(url_chunk)}] Crawling...")
                    job_data = extract_job_simple(driver, url)

                    if job_data:
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
                    except Exception:
                        pass

        # Split URLs across workers
        chunks = [[] for _ in range(threads)]
        for i, url in enumerate(urls):
            chunks[i % threads].append(url)

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
        logger.info(f"  Crawled: {stats['crawled']} | Cleaned: {stats['cleaned']}")
        logger.info(f"  Failed crawl: {stats['failed_crawl']} | Failed clean: {stats['failed_clean']}")
        logger.info(f"  Output: {output_file}")
        logger.info("=" * 70)

        return output_file, stats

    # ============= UPSERT TO ES =============
    def _url_to_doc_id(self, url):
        return hashlib.md5(url.encode('utf-8')).hexdigest()

    def _should_recrawl_existing(self, existing_doc, recrawl_after_days):
        """Return True if an existing ES doc should be crawled again."""
        if recrawl_after_days is None or recrawl_after_days < 0:
            return False

        last_crawled = existing_doc.get("last_crawled") or existing_doc.get("crawled_date")
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

    def _filter_existing_urls(self, urls, recrawl_after_days=7, force_recrawl_existing=False):
        """
        Skip URLs already present in ES when they were crawled recently.

        Existing docs older than recrawl_after_days are kept so the pipeline can
        detect changed content_hash and update the document.
        """
        stats = {"input": len(urls), "skipped_existing": 0, "recrawl_existing": 0}
        if not self.es or force_recrawl_existing or not urls:
            return urls, stats

        filtered = []
        try:
            ids = [self._url_to_doc_id(url) for url in urls]
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

            for url in urls:
                doc_id = self._url_to_doc_id(url)
                existing = existing_by_id.get(doc_id)
                if not existing:
                    filtered.append(url)
                    continue

                if self._should_recrawl_existing(existing, recrawl_after_days):
                    filtered.append(url)
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
            return urls, stats

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
        semantic_profiles = None
        if self.use_embeddings and self.embedding_service:
            logger.info("[EMBED] Generating embeddings (bge-m3)...")
            embed_start = time.time()
            semantic_profiles = []
            semantic_texts = []
            for _, row in df.iterrows():
                job = row.to_dict()
                profile = self.semantic_profile_builder.build(job, include_searchable_fields=True)
                semantic_profiles.append(profile)
                semantic_texts.append(profile["semantic_text"])
            all_embeddings = self.embedding_service.encode(
                semantic_texts, batch_size=32, show_progress=True,
            )
            logger.info(
                f"[OK] {len(all_embeddings)} semantic embeddings in {time.time()-embed_start:.1f}s"
            )

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
            for num_field in ['salary_min', 'salary_max']:
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

            doc["last_crawled"] = datetime.now().isoformat()

            # Embedding
            if all_embeddings is not None:
                doc.update(semantic_profiles[idx])
                doc["embedding"] = all_embeddings[idx].tolist()

            # Check existing
            try:
                existing = self.es.get(index=self.index_name, id=doc_id, ignore=[404])
                if existing and existing.get('found'):
                    old_hash = existing['_source'].get('content_hash', '')
                    if old_hash == new_hash:
                        existing_source = existing.get('_source', {})
                        update_doc = {"last_crawled": doc["last_crawled"]}
                        embedding_updated = False
                        structured_updated = False
                        for field in [
                            "requirements_tags", "specializations", "technical_skills",
                            "languages", "certificates",
                        ]:
                            if (
                                doc.get(field, "")
                                and doc.get(field, "") != existing_source.get(field, "")
                            ):
                                update_doc[field] = doc[field]
                                structured_updated = True
                        if all_embeddings is not None and not existing_source.get("semantic_text"):
                            update_doc.update(semantic_profiles[idx])
                            update_doc["embedding"] = doc["embedding"]
                            embedding_updated = True
                        elif all_embeddings is not None and structured_updated:
                            update_doc.update(semantic_profiles[idx])
                            update_doc["embedding"] = doc["embedding"]
                            embedding_updated = True
                        if embedding_updated or structured_updated:
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
                from job_matching.ingestion.backfill_semantic import ensure_mapping
                ensure_mapping(self.es, self.index_name)
        except Exception as e:
            logger.error(f"[ERROR] Ensure index: {e}")

    # ============= FULL CYCLE =============
    def run_cycle(
        self, pages=5, threads=3,
        recrawl_after_days=7, force_recrawl_existing=False, **kwargs,
    ):
        """1 chu kỳ: crawl → clean → upsert → check expired"""
        start_time = datetime.now()
        logger.info("=" * 70)
        logger.info(f"[CYCLE] START — {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)

        stats = {"start_time": start_time.isoformat(), "pages": pages, "threads": threads}

        # 1. Ensure index
        self.ensure_index()

        # 2. Pipeline: crawl + clean
        output_file, pipeline_stats = self.run_pipeline(
            pages, threads,
            recrawl_after_days=recrawl_after_days,
            force_recrawl_existing=force_recrawl_existing,
        )
        stats["pipeline"] = pipeline_stats

        if not output_file:
            stats["status"] = pipeline_stats.get("status", "pipeline_failed")
            self._log_run(stats)
            return stats

        # Merge terms every cycle. Classify only after enough unrecognized
        # terms accumulate, so crawl does not create tiny API batches.
        try:
            from job_matching.enrichment.build_term_taxonomy import (
                DEFAULT_PENDING_JSONL,
                DEFAULT_TERMS_CSV,
                classify_new_terms,
                merge_pending_terms,
            )
            taxonomy_terms = merge_pending_terms(DEFAULT_TERMS_CSV, DEFAULT_PENDING_JSONL)
            stats["taxonomy_unique_terms"] = len(taxonomy_terms)
            logger.info("[TAXONOMY] Candidate list now has %s unique terms", len(taxonomy_terms))

            if AUTO_CLASSIFY_TAXONOMY:
                taxonomy_result = classify_new_terms(
                    minimum_terms=TAXONOMY_CLASSIFY_THRESHOLD,
                    batch_size=TAXONOMY_BATCH_SIZE,
                    sleep_sec=TAXONOMY_SLEEP_SEC,
                    model=TAXONOMY_MODEL,
                )
                stats["taxonomy"] = taxonomy_result
                if taxonomy_result["triggered"]:
                    logger.info(
                        "[TAXONOMY] Classified %s new terms; taxonomy now has %s rows",
                        taxonomy_result["classified"],
                        taxonomy_result["taxonomy_rows"],
                    )
                    if self.semantic_profile_builder is not None:
                        from job_matching.enrichment.semantic_job_profile import (
                            SemanticJobProfileBuilder,
                        )
                        self.semantic_profile_builder = SemanticJobProfileBuilder()
                    stats["taxonomy_backfill"] = self._backfill_taxonomy_terms(
                        taxonomy_result.get("classified_terms", [])
                    )
                else:
                    logger.info(
                        "[TAXONOMY] Waiting: %s/%s unclassified terms",
                        taxonomy_result["remaining"],
                        TAXONOMY_CLASSIFY_THRESHOLD,
                    )
        except Exception as exc:
            logger.warning("[TAXONOMY] Maintenance skipped after error: %s", exc)

        # The taxonomy file may have changed outside this scheduler process.
        # Always reload before building semantic_text for the current crawl batch.
        if self.semantic_profile_builder is not None:
            from job_matching.enrichment.semantic_job_profile import (
                SemanticJobProfileBuilder,
            )
            self.semantic_profile_builder = SemanticJobProfileBuilder()

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
            recrawl_after_days=args.recrawl_after_days,
            force_recrawl_existing=args.force_recrawl_existing,
        )
    else:
        scheduler.run_periodic(
            interval_minutes=args.interval,
            pages=args.pages,
            threads=args.threads,
            recrawl_after_days=args.recrawl_after_days,
            force_recrawl_existing=args.force_recrawl_existing,
        )


if __name__ == '__main__':
    main()
