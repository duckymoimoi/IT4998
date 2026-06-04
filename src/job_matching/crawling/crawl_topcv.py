
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import time
import csv
import re
import sys
import os
import subprocess
import random
import hashlib
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ============= CẤU HÌNH =============
BATCH_SIZE = 20
BASE_URL = "https://www.topcv.vn/viec-lam-tot-nhat"
MAX_WORKERS = 3  # Số browser song song

# Timeout & Retry settings
PAGE_LOAD_TIMEOUT = 35
MAX_RETRIES = 2
DELAY_BETWEEN_REQUESTS = (3, 7)

# User agents — khớp Chrome mới trong Docker/local
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) Gecko/20100101 Firefox/138.0",
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawl_topcv.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Lock để tránh race condition khi nhiều thread cùng khởi tạo undetected_chromedriver
DRIVER_LOCK = threading.Lock()

def _detect_chrome_major_version():
    """Return installed Chrome major version, or None if detection fails."""
    env_version = os.environ.get("CRAWL_CHROME_VERSION") or os.environ.get("CHROME_VERSION_MAIN")
    if env_version:
        try:
            return int(str(env_version).split(".")[0])
        except ValueError:
            logger.warning(f"Invalid Chrome version override: {env_version}")

    # On Windows, calling "chrome.exe --version" can open Chrome and block.
    # Read the Chrome version from Registry instead.
    if sys.platform.startswith("win"):
        try:
            import winreg

            registry_paths = [
                (winreg.HKEY_CURRENT_USER, r"Software\Google\Chrome\BLBeacon"),
                (winreg.HKEY_LOCAL_MACHINE, r"Software\Google\Chrome\BLBeacon"),
                (winreg.HKEY_LOCAL_MACHINE, r"Software\WOW6432Node\Google\Chrome\BLBeacon"),
            ]
            for hive, key_path in registry_paths:
                try:
                    with winreg.OpenKey(hive, key_path) as key:
                        version, _ = winreg.QueryValueEx(key, "version")
                    match = re.search(r"(\d+)\.", str(version))
                    if match:
                        return int(match.group(1))
                except OSError:
                    continue
        except Exception as exc:
            logger.warning(f"Could not detect Chrome version from Registry: {exc}")
        return None

    candidates = [
        os.environ.get("CHROME_BIN"),
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ]
    for cmd in candidates:
        if not cmd:
            continue
        try:
            output = subprocess.check_output(
                [cmd, "--version"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            match = re.search(r"(\d+)\.", output)
            if match:
                return int(match.group(1))
        except Exception:
            continue
    return None

def setup_driver(headless=False):
    """Cấu hình undetected-chromedriver (bypass anti-bot).
    
    headless=False (mặc định): bypass anti-bot tốt nhất.
    Cửa sổ sẽ minimize tự động.
    """
    env_headless = os.environ.get("CRAWL_HEADLESS")
    if env_headless is not None:
        headless = env_headless.lower() in ("1", "true", "yes")
    elif sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        headless = True

    chrome_major = _detect_chrome_major_version()

    options = uc.ChromeOptions()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-setuid-sandbox")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-background-networking")
    options.add_argument("--remote-debugging-address=127.0.0.1")
    options.add_argument("--remote-debugging-port=0")
    if not headless:
        options.add_argument("--start-minimized")

    # Tắt hình ảnh để tăng tốc
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
    }
    options.add_experimental_option("prefs", prefs)
    options.page_load_strategy = 'eager'  # Không chờ full load

    with DRIVER_LOCK:
        logger.info(f"Starting Chrome driver: headless={headless}, chrome_major={chrome_major}")
        driver = uc.Chrome(options=options, headless=headless, version_main=chrome_major)
        
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    driver.implicitly_wait(5)

    return driver


def clean_text(text):
    """Làm sạch text"""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.replace('\u00a0', '').replace('\u200b', '').strip())


def random_delay(min_sec=None, max_sec=None):
    """Random delay để giả lập người dùng thật"""
    if min_sec is None or max_sec is None:
        min_sec, max_sec = DELAY_BETWEEN_REQUESTS
    delay = random.uniform(min_sec, max_sec)
    time.sleep(delay)
    return delay


def is_valid_job_url(url):
    """Kiểm tra URL hợp lệ (standard + brand)"""
    if not url:
        return False

    # Standard: /viec-lam/{slug}/{id}.html
    if re.match(r'https://www\.topcv\.vn/viec-lam/[^/]+/\d+\.html', url):
        return True

    # Brand: /brand/{company}/tuyen-dung/{slug}-j{id}.html
    if re.match(r'https://www\.topcv\.vn/brand/[^/]+/tuyen-dung/.+-j\d+\.html', url):
        return True

    return False


def is_brand_url(url):
    """Kiểm tra xem URL có phải brand page không"""
    return '/brand/' in url if url else False


def _is_blocked_page(soup, page_source):
    """Phát hiện trang bị chặn/redirect bởi anti-bot TopCV"""
    # Kiểm tra page source quá ngắn (trang bình thường > 10KB)
    if len(page_source) < 5000:
        return True

    # === Standard page elements ===
    has_job_body = soup.find("div", class_="job-detail__body") is not None
    has_job_desc = soup.find("div", class_="job-description") is not None
    has_job_info = soup.find("div", class_="job-detail__info--title") is not None
    has_h1_title = soup.find("h1", class_="job-detail__info--title") is not None

    # === Brand page elements ===
    has_brand_title = soup.find("h1", class_="job-title") is not None
    has_brand_content = soup.find("div", class_="content-tab") is not None
    has_brand_body = soup.find("div", class_="brand-body") is not None
    has_brand_info = soup.find("div", class_="box-info") is not None
    has_brand_desc = soup.find("h2", class_="title") is not None  # Brand dùng h2.title cho sections

    all_checks = [
        has_job_body, has_job_desc, has_job_info, has_h1_title,  # standard
        has_brand_title, has_brand_content, has_brand_body, has_brand_info, has_brand_desc,  # brand
    ]

    if not any(all_checks):
        # Kiểm tra thêm: nếu page title chứa "topcv" nhưng không có nội dung job
        page_title = soup.find("title")
        if page_title:
            title_text = page_title.text.strip().lower()
            if title_text in ['www.topcv.vn', 'topcv', 'topcv.vn', '']:
                return True
        # Nếu không có bất kỳ element job nào
        return True

    return False


def _is_valid_title(title_text):
    """Kiểm tra title có hợp lệ không (không phải title rác)"""
    if not title_text:
        return False

    invalid_titles = [
        'www.topcv.vn', 'topcv.vn', 'topcv', 'trang chủ',
        'tin tuyển dụng', 'tuyển dụng', 'việc làm',
        'tìm việc làm', 'top cv', '404', 'not found',
        'error', 'access denied', 'forbidden',
    ]
    title_lower = title_text.strip().lower()

    # Kiểm tra exact match với danh sách title không hợp lệ
    if title_lower in invalid_titles:
        return False

    # Kiểm tra title chỉ là URL/domain
    if title_lower.startswith('http') or title_lower.startswith('www.'):
        return False

    # Title quá ngắn (ít hơn 3 ký tự) thường không hợp lệ
    if len(title_text.strip()) < 3:
        return False

    return True


def extract_job_simple(driver, job_url, retry_count=0):
    """Trích xuất job với retry mechanism (standard + brand pages)"""
    try:
        logger.debug(f"  Attempt {retry_count + 1}/{MAX_RETRIES + 1}: {job_url}")

        timeout = 10 + (retry_count * 5)
        brand_page = is_brand_url(job_url)

        driver.get(job_url)

        # Brand pages dùng selector khác
        wait_selector = "job-description" if brand_page else "job-detail__body"

        try:
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CLASS_NAME, wait_selector))
            )
        except TimeoutException:
            # Fallback: thử selector khác
            try:
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "h1, .job-detail__info--title, .job-title"))
                )
            except TimeoutException:
                logger.warning(f"  Timeout ({timeout}s): {job_url[:80]}...")
                if retry_count < MAX_RETRIES:
                    logger.info(f"  Retry {retry_count + 1}/{MAX_RETRIES}...")
                    random_delay(3, 6)  # Delay lâu hơn khi timeout
                    return extract_job_simple(driver, job_url, retry_count + 1)
                else:
                    logger.warning(f"  Max retries reached, skipping...")
                    return None

        time.sleep(1.5)

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")

        # === PHÁT HIỆN TRANG BỊ CHẶN ===
        if _is_blocked_page(soup, page_source):
            logger.warning(f"  [BLOCKED] Trang bi chan/redirect: {job_url[:80]}...")
            if retry_count < MAX_RETRIES:
                # Cooldown delay dài hơn khi bị chặn
                cooldown = 5 + (retry_count * 5)  # 5s, 10s, 15s...
                logger.info(f"  Cooldown {cooldown}s truoc khi retry {retry_count + 1}/{MAX_RETRIES}...")
                time.sleep(cooldown)
                return extract_job_simple(driver, job_url, retry_count + 1)
            else:
                logger.warning(f"  [SKIP] Max retries sau khi bi chan: {job_url[:60]}")
                return None

        job_data = {
            'url': job_url,
            'crawled_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # === TITLE ===
        try:
            title_tag = soup.find("h1", class_="job-detail__info--title")
            if not title_tag:
                # Brand page fallback: tìm h1.job-title
                title_tag = soup.find("h1", class_="job-title")
            if not title_tag and brand_page:
                # Brand pages: title nằm trong thẻ <title> hoặc og:title
                og_title = soup.find("meta", attrs={"property": "og:title"})
                if og_title:
                    raw_title = og_title.get("content", "")
                    # Loại bỏ suffix "tại COMPANY_NAME" nếu có
                    raw_title = re.sub(r'\s+(tại|làm việc tại)\s+.+$', '', raw_title, flags=re.IGNORECASE)
                    # Loại bỏ prefix "Tuyển" nếu có
                    raw_title = re.sub(r'^Tuyển\s+', '', raw_title).strip()
                    job_data['title'] = clean_text(raw_title)
                else:
                    # Fallback: lấy từ page <title>
                    page_title = soup.find("title")
                    if page_title:
                        raw_title = page_title.text
                        raw_title = re.sub(r'\s+(tại|làm việc tại)\s+.+$', '', raw_title, flags=re.IGNORECASE)
                        raw_title = re.sub(r'^Tuyển\s+', '', raw_title).strip()
                        job_data['title'] = clean_text(raw_title)
                    else:
                        job_data['title'] = ""
            elif title_tag:
                job_data['title'] = clean_text(title_tag.text)
            else:
                title_tag = soup.find("h1")
                title_text = clean_text(title_tag.text) if title_tag else ""
                # Validate title bằng hàm riêng
                if _is_valid_title(title_text):
                    job_data['title'] = title_text
                else:
                    job_data['title'] = ""
        except Exception as e:
            logger.debug(f"  Warning: Error parsing title: {e}")
            job_data['title'] = ""

        # Validate title ngay sau khi extract
        if not _is_valid_title(job_data.get('title', '')):
            job_data['title'] = ""

        # === COMPANY ===
        try:
            company_tag = soup.find("div", class_="job-detail__company-title")
            if not company_tag:
                company_tag = soup.find("a", class_="job-detail__company-title-link")
            if not company_tag:
                # Brand page fallback
                company_tag = soup.find("a", class_="company-name")
            if not company_tag:
                company_tag = soup.find("div", class_="company-name")
            if not company_tag:
                company_tag = soup.find("h2", class_="company-name")
            if not company_tag and brand_page:
                # Brand page: thử lấy từ sidebar "Pro Company" card
                pro_cards = soup.find_all("span", string=re.compile(r'Pro', re.IGNORECASE))
                for card in pro_cards:
                    parent = card.find_parent("div")
                    if parent:
                        company_span = parent.find("span")
                        if company_span and 'Pro' not in company_span.text:
                            job_data['company'] = clean_text(company_span.text)
                            break
                # Fallback: og:site_name hoặc meta
                if not job_data.get('company'):
                    meta_company = soup.find("meta", attrs={"property": "og:site_name"})
                    if meta_company:
                        job_data['company'] = meta_company.get("content", "")
                    else:
                        # Thử lấy từ <title>: "... tại COMPANY_NAME"
                        page_title = soup.find("title")
                        if page_title:
                            title_match = re.search(r'(?:tại|làm việc tại)\s+(.+?)(?:\s*-\s*TopCV|$)', page_title.text)
                            if title_match:
                                job_data['company'] = clean_text(title_match.group(1))
                            else:
                                job_data['company'] = ""
                        else:
                            job_data['company'] = ""
            else:
                job_data['company'] = clean_text(company_tag.text) if company_tag else ""
        except Exception as e:
            logger.debug(f"  Warning: Error parsing company: {e}")
            job_data['company'] = ""



        # === 1. OVERVIEW - Thông tin tổng quan (CHỈ LẤY ĐẾN HẠN NỘP HỒ SƠ) ===
        overview_parts = []

        try:
            # === Phần thông tin chính (blue box - left side) ===
            blue_box = soup.find("div", class_="job-detail__body-left")
            if blue_box:
                info_items = []

                # 1. Lấy các section: Mức lương, Địa điểm, Kinh nghiệm
                info_sections = blue_box.find_all("div", class_="job-detail__info--section")
                for section in info_sections:
                    title_el = section.find("div", class_="job-detail__info--section-content-title")
                    value_el = section.find("div", class_="job-detail__info--section-content-value")
                    if title_el and value_el:
                        title = clean_text(title_el.text)
                        value = clean_text(value_el.get_text(separator=" ", strip=True))
                        if title and value:
                            info_items.append(f"{title}: {value}")

                # 2. Lấy Hạn nộp hồ sơ
                deadline_el = blue_box.find("div", class_="job-detail__info--deadline")
                if deadline_el:
                    deadline_text = clean_text(deadline_el.get_text(separator=" ", strip=True))
                    if deadline_text:
                        info_items.append(deadline_text)

                if info_items:
                    overview_parts.append("=== THÔNG TIN CHÍNH ===\n" + "\n".join(info_items))

                # 3. Lấy phần Chi tiết tin tuyển dụng (Yêu cầu, Quyền lợi, Chuyên môn)
                job_tags = blue_box.find("div", class_="job-tags")
                if job_tags:
                    tag_groups = job_tags.find_all("div", class_="job-tags__group")
                    for group in tag_groups:
                        group_name_el = group.find("div", class_="job-tags__group-name")
                        if group_name_el:
                            group_name = clean_text(group_name_el.text)
                            # Lấy các tag item
                            tags = group.find_all("a", class_="item")
                            if tags:
                                tag_texts = [clean_text(tag.text) for tag in tags if tag.text.strip() and "Xem thêm" not in tag.text]
                                if tag_texts:
                                    overview_parts.append(f"\n{group_name}\n" + "\n".join(tag_texts))

            # === Phần thông tin bổ sung (right side - sidebar) ===
            sidebar = soup.find("div", class_="job-detail__body-right")
            if sidebar:
                # Box thông tin chung (Cấp bậc, Học vấn, Số lượng tuyển, Hình thức...)
                general_box = sidebar.find("div", class_="box-general-group")
                if general_box:
                    general_items = []
                    info_items = general_box.find_all("div", class_="box-general-group-info")
                    for item in info_items:
                        title_el = item.find("div", class_="box-general-group-info-title")
                        value_el = item.find("div", class_="box-general-group-info-value")
                        if title_el and value_el:
                            title = clean_text(title_el.text)
                            value = clean_text(value_el.text)
                            if title and value:
                                general_items.append(f"{title}: {value}")

                    if general_items:
                        overview_parts.append("\n=== THÔNG TIN CHUNG ===\n" + "\n".join(general_items))

                # === THÔNG TIN CÔNG TY (sidebar company box) ===
                company_box = sidebar.find("div", class_="job-detail__company")
                if company_box:
                    # Lấy tất cả text trong company box để parse
                    company_info_text = company_box.get_text(separator="\n", strip=True)

                    # Quy mô
                    size_match = re.search(r'Quy mô[:\s]*(.+?)(?:\n|$)', company_info_text)
                    if size_match:
                        job_data['company_size'] = clean_text(size_match.group(1))

                    # Lĩnh vực
                    field_match = re.search(r'Lĩnh vực[:\s]*(.+?)(?:\n|$)', company_info_text)
                    if field_match:
                        job_data['company_field'] = clean_text(field_match.group(1))

                    # Địa điểm (trụ sở công ty)
                    addr_match = re.search(r'Địa điểm[:\s]*(.+?)(?:\n|$)', company_info_text)
                    if addr_match:
                        addr = clean_text(addr_match.group(1))
                        # Loại bỏ "Xem bản đồ" nếu có
                        addr = re.sub(r'Xem bản đồ.*$', '', addr).strip()
                        if addr and addr != 'Xem trang công ty':
                            job_data['company_address'] = addr

        except Exception as e:
            logger.debug(f"  Warning: Error parsing overview: {str(e)}")

        # === BRAND PAGE: Fallback cho overview nếu rỗng ===
        if brand_page and not overview_parts:
            try:
                # Thử lấy thông tin từ .box-info hoặc .job-info
                box_info = soup.find("div", class_="box-info") or soup.find("div", class_="box-info-job")
                if box_info:
                    brand_items = []
                    items = box_info.find_all("div", class_="item-info") or box_info.find_all("div", recursive=False)
                    for item in items:
                        item_text = clean_text(item.get_text(separator=" ", strip=True))
                        if item_text:
                            brand_items.append(item_text)
                    if brand_items:
                        overview_parts.append("=== THÔNG TIN CHÍNH ===\n" + "\n".join(brand_items))
            except Exception as e:
                logger.debug(f"  Warning: Brand overview fallback error: {e}")

        job_data['overview'] = "\n".join(overview_parts)

        # Đảm bảo 3 field company info luôn tồn tại
        job_data.setdefault('company_address', '')
        job_data.setdefault('company_size', '')
        job_data.setdefault('company_field', '')

        # === 2. JOB DETAILS - Chi tiết công việc ===
        job_details_parts = []

        try:
            description_container = soup.find("div", class_="job-description")
            if description_container:
                items = description_container.find_all("div", class_="job-description__item")

                for item in items:
                    title_tag = item.find("h3")
                    content_div = item.find("div", class_="job-description__item--content")

                    if title_tag and content_div:
                        section_title = clean_text(title_tag.text)
                        section_content = content_div.get_text(separator="\n", strip=True)

                        # Format: === TIÊU ĐỀ === \n Nội dung
                        job_details_parts.append(f"=== {section_title.upper()} ===\n{section_content}")
        except Exception as e:
            logger.debug(f"  Warning: Error parsing job details: {str(e)}")

        job_data['job_details'] = "\n\n".join(job_details_parts)

        # === Fallback cho brand pages: nếu overview/job_details rỗng ===
        if brand_page and not job_data.get('job_details'):
            try:
                # Brand pages dùng cấu trúc: <h2 class="title">Mô tả công việc</h2>
                #                             <div class="content-tab">...nội dung...</div>
                section_headers_map = {
                    'mô tả công việc': 'MÔ TẢ CÔNG VIỆC',
                    'yêu cầu ứng viên': 'YÊU CẦU ỨNG VIÊN',
                    'quyền lợi': 'QUYỀN LỢI',
                    'quyền lợi được hưởng': 'QUYỀN LỢI',
                    'thời gian làm việc': 'THỜI GIAN LÀM VIỆC',
                    'địa điểm làm việc': 'ĐỊA ĐIỂM LÀM VIỆC',
                    'thu nhập': 'THU NHẬP',
                    'mức lương': 'THU NHẬP',
                }

                # Tìm tất cả h2.title — đây là label của mỗi section
                h2_titles = soup.find_all('h2', class_='title')
                for h2 in h2_titles:
                    section_label = clean_text(h2.text).lower().strip()

                    mapped_title = None
                    for key, value in section_headers_map.items():
                        if key in section_label:
                            mapped_title = value
                            break

                    if not mapped_title:
                        continue

                    # Lấy div.content-tab ngay sau h2
                    content_tab = h2.find_next_sibling('div', class_='content-tab')
                    if content_tab:
                        content_text = content_tab.get_text(separator="\n", strip=True)
                        if content_text:
                            job_details_parts.append(f"=== {mapped_title} ===\n{content_text}")

                # Fallback 2: nếu không có h2.title, thử h3/h4 trong body
                if not job_details_parts:
                    body = soup.find('div', class_='job-detail__body') or soup.find('main') or soup.find('body')
                    if body:
                        for header in body.find_all(['h3', 'h4']):
                            header_text = clean_text(header.text).lower().strip()
                            mapped_title = None
                            for key, value in section_headers_map.items():
                                if key in header_text:
                                    mapped_title = value
                                    break
                            if not mapped_title:
                                continue
                            content_parts = []
                            for sibling in header.next_siblings:
                                if sibling.name in ['h3', 'h4', 'h2']:
                                    break
                                text = sibling.get_text(separator="\n", strip=True) if hasattr(sibling, 'get_text') else str(sibling).strip()
                                if text:
                                    content_parts.append(text)
                            if content_parts:
                                job_details_parts.append(f"=== {mapped_title} ===\n" + "\n".join(content_parts))

                job_data['job_details'] = "\n\n".join(job_details_parts)
            except Exception as e:
                logger.debug(f"  Warning: Brand page fallback error: {e}")



        # === CONTENT HASH & DEADLINE (cho scheduler) ===
        # Content hash: dùng để phát hiện thay đổi khi crawl lại
        hash_content = f"{job_data.get('title', '')}|{job_data.get('job_details', '')}|{job_data.get('overview', '')}"
        job_data['content_hash'] = hashlib.md5(hash_content.encode('utf-8')).hexdigest()

        # Trích xuất deadline từ overview
        job_data['deadline'] = ''
        job_data['is_expired'] = False
        try:
            overview_text = job_data.get('overview', '')
            # Pattern: "Hạn nộp hồ sơ: 15/05/2026" hoặc "(Còn X ngày)"
            deadline_match = re.search(
                r'Hạn nộp[^:]*:\s*(\d{1,2}/\d{1,2}/\d{4})',
                overview_text
            )
            if deadline_match:
                job_data['deadline'] = deadline_match.group(1)
                # Parse và kiểm tra hết hạn
                try:
                    deadline_date = datetime.strptime(deadline_match.group(1), '%d/%m/%Y')
                    if deadline_date < datetime.now():
                        job_data['is_expired'] = True
                except ValueError:
                    pass
            # Kiểm tra nếu có text "Đã hết hạn"
            if 'hết hạn' in overview_text.lower() or 'đã hết hạn' in overview_text.lower():
                job_data['is_expired'] = True
        except Exception as e:
            logger.debug(f"  Warning: Error parsing deadline: {e}")
        # === DATA QUALITY CHECK ===
        title = job_data.get('title', '')
        has_overview = bool(job_data.get('overview', '').strip())
        has_details = bool(job_data.get('job_details', '').strip())
        has_company = bool(job_data.get('company', '').strip())

        # Coi như thất bại nếu không có title VÀ không có nội dung
        if not title and not has_overview and not has_details:
            logger.warning(f"  [BAD DATA] Khong co title, overview, job_details: {job_url[:60]}")
            if retry_count < MAX_RETRIES:
                cooldown = 4 + (retry_count * 3)
                logger.info(f"  Retry {retry_count + 1}/{MAX_RETRIES} sau {cooldown}s (bad data)...")
                time.sleep(cooldown)
                return extract_job_simple(driver, job_url, retry_count + 1)
            return None

        # Coi như thất bại nếu chỉ có title nhưng không có gì khác
        if title and not has_overview and not has_details and not has_company:
            logger.warning(f"  [INCOMPLETE] Chi co title, thieu data: {job_url[:60]}")
            if retry_count < MAX_RETRIES:
                cooldown = 4 + (retry_count * 3)
                logger.info(f"  Retry {retry_count + 1}/{MAX_RETRIES} sau {cooldown}s (incomplete)...")
                time.sleep(cooldown)
                return extract_job_simple(driver, job_url, retry_count + 1)
            # Vẫn trả về nếu có title (tốt hơn là bỏ hoàn toàn)

        if not title:
            logger.warning(f"  Missing title: {job_url[:60]}")
        if not has_company:
            logger.debug(f"  Missing company: {job_url[:60]}")

        logger.info(f"  [OK] {title[:70] if title else '(no title)'}")
        return job_data

    except Exception as e:
        logger.error(f"  Error: {str(e)}")

        if retry_count < MAX_RETRIES:
            logger.info(f"  Retry {retry_count + 1}/{MAX_RETRIES} after error...")
            random_delay(2, 4)
            return extract_job_simple(driver, job_url, retry_count + 1)

        return None


def get_all_job_urls(driver, start_page=1, end_page=None):
    """Thu thập URLs từ các trang listing"""
    logger.info("=" * 70)
    logger.info("GIAI ĐOẠN 1: Thu thập URLs")
    logger.info("=" * 70)

    all_urls = set()
    page_num = start_page
    consecutive_failed_pages = 0
    consecutive_empty_pages = 0

    while True:
        if end_page and page_num > end_page:
            break

        page_url = BASE_URL if page_num == 1 else f"{BASE_URL}?page={page_num}"
        logger.info(f"\nTrang {page_num}: {page_url}")

        try:
            driver.get(page_url)
            time.sleep(3)  # Chờ page render

            try:
                WebDriverWait(driver, 25).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'h3[class*="title"] a'))
                )
            except TimeoutException:
                # Thử lại 1 lần với delay dài hơn
                logger.warning(f"Trang {page_num} timeout, retry...")
                time.sleep(5)
                driver.refresh()
                try:
                    WebDriverWait(driver, 25).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'h3[class*="title"] a'))
                    )
                except TimeoutException:
                    logger.warning(f"Trang {page_num} timeout lần 2, skip")
                    page_num += 1
                    continue  # Tiếp tục trang sau thay vì dừng

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            job_links = soup.select('h3[class*="title"] a')

            if not job_links:
                logger.warning(f"Trang {page_num} khong co jobs")
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= 3:
                    logger.warning("Dung thu URL sau 3 trang lien tiep khong co jobs")
                    break
                page_num += 1
                continue

            page_urls = set()
            for link_tag in job_links:
                href = link_tag.get("href", "")
                if href.startswith("/"):
                    href = "https://www.topcv.vn" + href

                if is_valid_job_url(href):
                    page_urls.add(href)

            logger.info(f"  Tìm thấy {len(job_links)} links")
            logger.info(f"  Lọc được {len(page_urls)} URLs hợp lệ")

            all_urls.update(page_urls)
            logger.info(f"  Tổng: {len(all_urls)} URLs")

            if len(page_urls) == 0:
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= 3:
                    logger.warning("Dung thu URL sau 3 trang lien tiep khong co URL hop le")
                    break
                page_num += 1
                continue

            consecutive_failed_pages = 0
            consecutive_empty_pages = 0

            page_num += 1
            time.sleep(1)

        except Exception as e:
            logger.error(f"Loi trang {page_num}: {str(e)}")
            consecutive_failed_pages += 1
            if consecutive_failed_pages >= 3:
                logger.warning("Dung thu URL sau 3 trang lien tiep bi loi")
                break
            page_num += 1
            continue

    logger.info("\n" + "=" * 70)
    logger.info(f"Hoàn thành: {len(all_urls)} URLs")
    logger.info("=" * 70)

    return list(all_urls)


def crawl_jobs(all_job_urls, output_file=None, max_workers=MAX_WORKERS):
    """Crawl chi tiết jobs (multi-thread)"""
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'topcv_jobs_simple_{timestamp}.csv'

    logger.info("\n" + "=" * 70)
    logger.info(f"GIAI ĐOẠN 2: Crawl {len(all_job_urls)} jobs ({max_workers} threads)")
    logger.info("=" * 70)

    fieldnames = ['title', 'url', 'company', 'company_address',
                  'company_size', 'company_field',
                  'overview', 'job_details', 'crawled_date',
                  'content_hash', 'deadline', 'is_expired']

    file_exists = False
    try:
        with open(output_file, 'r', encoding='utf-8-sig'):
            file_exists = True
    except:
        pass

    csv_lock = threading.Lock()
    csv_file = open(output_file, 'a', newline='', encoding='utf-8-sig')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    crawled_count = 0
    skipped_count = 0

    def _worker_crawl(url_chunk, worker_id):
        """Mỗi worker có browser riêng"""
        nonlocal crawled_count, skipped_count
        driver = None
        local_count = 0

        try:
            driver = setup_driver()
            # Warm up
            try:
                driver.get("https://www.topcv.vn/")
                random_delay(2, 4)
            except:
                pass

            for i, job_url in enumerate(url_chunk):
                # Restart browser mỗi BATCH_SIZE jobs
                if i > 0 and i % BATCH_SIZE == 0:
                    logger.info(f"  [W{worker_id}] Restart browser (batch {i // BATCH_SIZE + 1})...")
                    try:
                        driver.quit()
                    except:
                        pass
                    random_delay(3, 6)
                    driver = setup_driver()
                    try:
                        driver.get("https://www.topcv.vn/")
                        random_delay(2, 3)
                    except:
                        pass

                logger.info(f"  [W{worker_id}] [{i + 1}/{len(url_chunk)}] Crawling...")
                job_data = extract_job_simple(driver, job_url)

                if job_data:
                    row = {field: job_data.get(field, '') for field in fieldnames}
                    with csv_lock:
                        writer.writerow(row)
                        csv_file.flush()
                        crawled_count += 1
                    local_count += 1
                    random_delay()
                else:
                    logger.warning(f"  [W{worker_id}] Skip: {job_url[:60]}...")
                    with csv_lock:
                        skipped_count += 1
                    random_delay(1, 2)

        except Exception as e:
            logger.error(f"  [W{worker_id}] Worker error: {e}")
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

        logger.info(f"  [W{worker_id}] Done: {local_count}/{len(url_chunk)} jobs")
        return local_count

    # Chia URLs cho các workers
    chunks = [[] for _ in range(max_workers)]
    for i, url in enumerate(all_job_urls):
        chunks[i % max_workers].append(url)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for wid, chunk in enumerate(chunks):
                if chunk:
                    futures.append(executor.submit(_worker_crawl, chunk, wid + 1))
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logger.error(f"Worker exception: {e}")

    except KeyboardInterrupt:
        logger.warning("\nDừng bởi user (Ctrl+C)")

    finally:
        csv_file.close()

    logger.info("\n" + "=" * 70)
    logger.info("HOÀN THÀNH!")
    logger.info("=" * 70)
    logger.info(f"Crawl thành công: {crawled_count}/{len(all_job_urls)} jobs")
    logger.info(f"Bị skip: {skipped_count}/{len(all_job_urls)} jobs")
    logger.info(f"File: {output_file}")
    logger.info("=" * 70)

    return crawled_count


def main():
    """Main function"""
    args = sys.argv[1:]
    output_file = 'topcv_jobs_simple.csv'

    start_page = 1
    end_page = None

    if len(args) == 2:
        try:
            start_page = int(args[0])
            end_page = int(args[1])

            if start_page < 1 or end_page < start_page:
                print("[ERROR] Loi: start_page >= 1 va end_page >= start_page")
                return
        except ValueError:
            print("[ERROR] Loi: Tham so phai la so nguyen")
            return

    elif len(args) == 0:
        confirm = input("[WARN] Crawl TAT CA trang? (y/n): ")
        if confirm.lower() != 'y':
            logger.info("Da huy")
            return

    else:
        print("=" * 70)
        print("HUONG DAN:")
        print("=" * 70)
        print("1. Crawl từ trang X → Y:")
        print("   python crawl_topcv.py <start> <end>")
        print("   Ví dụ: python crawl_topcv.py 1 5")
        print("")
        print("2. Crawl tất cả:")
        print("   python crawl_topcv.py")
        print("=" * 70)
        return

    try:
        # Giai đoạn 1: Lấy URLs
        driver = setup_driver()
        all_job_urls = get_all_job_urls(driver, start_page, end_page)
        driver.quit()

        if not all_job_urls:
            logger.error("[ERROR] Khong co URLs")
            return

        # Giai đoạn 2: Crawl chi tiết
        crawl_jobs(all_job_urls, output_file)

    except KeyboardInterrupt:
        logger.warning("\n[WARN] Dung boi user (Ctrl+C)")
    except Exception as e:
        logger.error(f"[ERROR] Loi: {str(e)}")


if __name__ == "__main__":
    main()

