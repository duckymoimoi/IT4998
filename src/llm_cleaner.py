"""
LLM Data Cleaner -- Trích xuất dữ liệu tuyển dụng bằng Ollama (Qwen 2.5)

Pipeline: CSV thô -> LLM clean (batch) -> CSV sạch -> import ES

Sử dụng:
    # Clean toàn bộ file
    python llm_cleaner.py --input topcv_jobs_multicategory_xxx.csv --output cleaned.csv

    # Chỉ clean N jobs đầu (thử nghiệm)
    python llm_cleaner.py --input raw.csv --output test.csv --limit 5

    # Dùng model khác
    python llm_cleaner.py --input raw.csv --output clean.csv --model llama3:8b
"""

import pandas as pd
import json
import time
import re
import logging
import argparse
import requests
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [LLM-CLEAN] %(levelname)-8s %(message)s'
)
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b"

# ---------------------------------------------------------------------------
# SYSTEM PROMPT -- tiếng Việt có dấu, không dùng emoji
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Bạn là công cụ trích xuất dữ liệu tuyển dụng.
Nhận vào text tin tuyển dụng, trả về JSON chuẩn hóa.
Luôn trả về JSON. Nếu không tìm thấy thông tin, để chuỗi rỗng "" hoặc null.
Không giải thích, không viết gì ngoài JSON."""

# ---------------------------------------------------------------------------
# EXTRACT PROMPT -- cải tiến: tiếng Việt có dấu, ví dụ rõ ràng, quy tắc chặt
# ---------------------------------------------------------------------------
EXTRACT_PROMPT = """Trích xuất thông tin từ tin tuyển dụng thành JSON với định dạng CHÍNH XÁC sau:

{{
  "salary": {{
    "text": "<text gốc về lương, sao chép nguyên văn>",
    "min": <số tiền tối thiểu (VND), null nếu thỏa thuận>,
    "max": <số tiền tối đa (VND), null nếu thỏa thuận>,
    "type": "<range|fixed|negotiable|upto>",
    "note": "<ghi chú NET/GROSS, hoa hồng>",
    "has_commission": <true|false>
  }},
  "location": {{
    "city": "<ten tinh/thanh pho>"
  }},
  "education": {{
    "level": "<Đại học|Cao đẳng|Trung cấp|Trung học|Không yêu cầu>",
    "field": "<chuyên ngành>"
  }},
  "skills": {{
    "technical": ["<công cụ/phần mềm/kỹ thuật>"],
    "soft": ["<kỹ năng mềm>"],
    "languages": ["<ngôn ngữ>"],
    "certificates": ["<chứng chỉ>"]
  }},
  "gender_requirement": "<Nam|Nữ|Không yêu cầu>"
}}

=== QUY TẮC LƯƠNG (ĐỌC KỸ) ===

Đơn vị là VND. Quy đổi: 1 triệu = 1.000.000 (6 chữ số 0).
  Ví dụ đúng: 12 triệu = 12000000, 40 triệu = 40000000, 200 triệu = 200000000
  Ví dụ sai: 40 triệu = 4000000 (thiếu một chữ số 0 -- lỗi phổ biến)
  Kiểm tra: nếu kết quả nhỏ hơn 1.000.000 thì chắc chắn bị thiếu chữ số 0.

Phân loại "type":
  - "range": có mức tối thiểu VÀ tối đa (ví dụ: "20 - 50 triệu")
  - "fixed": một mức cụ thể (ví dụ: "15 triệu")
  - "upto": chỉ có mức tối đa (ví dụ: "Upto 25 triệu" -> min=null, max=25000000)
  - "negotiable": không có con số cụ thể

Ví dụ:
  "20 - 50 triệu" -> min=20000000, max=50000000, type="range"
  "Thu nhập từ 40-200tr" -> min=40000000, max=200000000, type="range"
  "Upto 25 triệu" -> min=null, max=25000000, type="upto"
  "Thỏa thuận" -> min=null, max=null, type="negotiable"

has_commission=true nếu có: hoa hồng, thưởng doanh số, KPI, thu nhập không giới hạn.

=== QUY TẮC ĐỊA ĐIỂM ===

Ưu tiên lấy từ ĐỊA CHỈ CÔNG TY (đáng tin cậy nhất).
  Ví dụ: "171 Hai Bà Trưng, TP Hồ Chí Minh" -> city = "Hồ Chí Minh"
  Ví dụ: "21 Lê Đức Thọ, Hà Nội" -> city = "Hà Nội"
Dùng tên đầy đủ: "Hồ Chí Minh" (không phải "HCM").

=== QUY TẮC HỌC VẤN ===

level là mức TỐI THIỂU yêu cầu: Đại học, Cao đẳng, Trung cấp, Trung học, Không yêu cầu
Ví dụ: "Cao đẳng trở lên" -> "Cao đẳng"

=== QUY TẮC KỸ NĂNG ===

CHỈ trích xuất kỹ năng XUẤT HIỆN TRỰC TIẾP trong phần yêu cầu hoặc mô tả.
KHÔNG suy đoán, KHÔNG thêm kỹ năng không có trong text.
technical: công cụ/phần mềm cụ thể (Excel, AutoCAD, Python, SAP...)
soft: kỹ năng mềm (giao tiếp, teamwork...)
languages: ngôn ngữ (tiếng Anh, tiếng Trung...)
certificates: chứng chỉ (TOEIC, CPA, PMP...)
Nếu không tìm thấy, để mảng rỗng [].

=== QUY TẮC GIỚI TÍNH ===

Đọc phần yêu cầu và tiêu đề: "Nữ" -> "Nữ", "Nam" -> "Nam", mặc định: "Không yêu cầu"

=== TIN TUYỂN DỤNG ===
Tiêu đề: {title}
Công ty: {company}
Địa chỉ công ty: {company_address}

THÔNG TIN CHÍNH:
{overview}

CHI TIẾT:
{job_details}
"""


class LLMCleaner:
    """Clean dữ liệu tuyển dụng bằng Ollama LLM"""

    def __init__(self, model=DEFAULT_MODEL, ollama_url=OLLAMA_URL):
        self.model = model
        self.ollama_url = ollama_url
        self.stats = {"thanh_cong": 0, "that_bai": 0, "tong_thoi_gian": 0}

    def _kiem_tra_ollama(self):
        """Kiểm tra Ollama đang chạy và model có sẵn"""
        try:
            resp = requests.get(self.ollama_url.replace("/api/generate", "/api/tags"), timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if any(self.model in m for m in models):
                    logger.info(f"Ollama hoạt động -- model: {self.model}")
                    return True
                else:
                    logger.error(f"Model '{self.model}' chưa được pull. Các model hiện có: {models}")
                    return False
        except Exception as e:
            logger.error(f"Không kết nối được Ollama ({self.ollama_url}): {e}")
            return False

    def _goi_llm(self, prompt, timeout=90):
        """Gọi Ollama API, trả về (text_response, thoi_gian_giay)"""
        try:
            resp = requests.post(self.ollama_url, json={
                "model": self.model,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1500,
                    "top_p": 0.9,
                }
            }, timeout=timeout)

            if resp.status_code != 200:
                return None, 0

            result = resp.json()
            raw = result.get("response", "")
            thoi_gian = result.get("total_duration", 0) / 1e9
            return raw, thoi_gian

        except requests.Timeout:
            logger.warning("Timeout khi gọi LLM")
            return None, 0
        except Exception as e:
            logger.error(f"Lỗi LLM: {e}")
            return None, 0

    def _phan_tich_json(self, phan_hoi):
        """Parse JSON từ response LLM, xử lý các trường hợp có text thừa"""
        if not phan_hoi:
            return None

        # Tìm JSON block đầu tiên
        json_match = re.search(r'\{.*\}', phan_hoi, re.DOTALL)
        if not json_match:
            return None

        json_str = json_match.group()

        # Sửa lỗi JSON phổ biến của LLM
        json_str = re.sub(r',\s*}', '}', json_str)   # trailing comma trong object
        json_str = re.sub(r',\s*]', ']', json_str)   # trailing comma trong array
        json_str = re.sub(r'//.*?\n', '\n', json_str) # bỏ comment dạng //

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Không parse được JSON từ LLM")
            return None

    # -----------------------------------------------------------------------
    # Các hàm hậu xử lý (post-processing)
    # -----------------------------------------------------------------------

    def _sua_luong(self, result, original_row):
        """
        Sửa lỗi magnitude lương (VD: LLM trả 4000000 thay vì 40000000).
        Nguồn tham chiếu: salary_text và tiêu đề.
        """
        def trich_so_trieu(text):
            """Trích xuất các giá trị triệu từ text, trả về danh sách VND"""
            nums = []
            # Pattern: "40-200tr", "40 - 200 triệu"
            range_match = re.search(
                r'(\d+(?:[.,]\d+)?)\s*[-~]\s*(\d+(?:[.,]\d+)?)\s*(?:triệu|tr\b)',
                text, re.IGNORECASE
            )
            if range_match:
                nums.append(float(range_match.group(1).replace(',', '.')) * 1_000_000)
                nums.append(float(range_match.group(2).replace(',', '.')) * 1_000_000)
                return nums
            # Pattern: "25 triệu", "8tr", "Upto 25 Triệu"
            for m in re.finditer(
                r'(\d+(?:[.,]\d+)?)\s*(?:triệu|tr\b)', text, re.IGNORECASE
            ):
                nums.append(float(m.group(1).replace(',', '.')) * 1_000_000)
            return nums

        salary_text = str(result.get('job_salary', ''))
        title = str(result.get('title', ''))
        tham_chieu = trich_so_trieu(salary_text) or trich_so_trieu(title)

        sal_min = result.get('salary_min')
        sal_max = result.get('salary_max')

        def la_nan(val):
            return val is None or (isinstance(val, float) and pd.isna(val))

        def to_float(val):
            """Chuyen gia tri ve float, tra ve None neu khong the"""
            if la_nan(val):
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        sal_min = to_float(sal_min)
        sal_max = to_float(sal_max)
        result['salary_min'] = sal_min
        result['salary_max'] = sal_max

        # Sua neu LLM tra sai bac lon (chenh lech >= 5 lan)
        if tham_chieu and sal_min and sal_min > 0:
            ref_min = min(tham_chieu)
            if ref_min / sal_min >= 5:
                result['salary_min'] = ref_min
                if len(tham_chieu) > 1:
                    result['salary_max'] = max(tham_chieu)

        if tham_chieu and sal_max and sal_max > 0:
            ref_max = max(tham_chieu)
            if ref_max / sal_max >= 5:
                result['salary_max'] = ref_max

        # Fallback: neu min/max van None nhung co so trong text
        if tham_chieu:
            if result.get('salary_min') is None:
                result['salary_min'] = min(tham_chieu)
            if result.get('salary_max') is None and len(tham_chieu) > 1:
                result['salary_max'] = max(tham_chieu)

        # Kiểm tra cuối: nếu giá trị < 1 triệu thì nhân 10 (sai bậc đơn vị)
        if result.get('salary_min') and result['salary_min'] < 1_000_000:
            result['salary_min'] = result['salary_min'] * 10
        if result.get('salary_max') and result['salary_max'] < 1_000_000:
            result['salary_max'] = result['salary_max'] * 10

        return result

    def _sua_dia_diem(self, result, original_row):
        """
        Ưu tiên lấy tỉnh/thành từ địa chỉ công ty.
        LLM đôi khi lấy từ overview thay vì địa chỉ công ty, dẫn đến sai.
        """
        bang_thanh_pho = {
            'Hà Nội': 'Hà Nội', 'Ha Noi': 'Hà Nội',
            'Hồ Chí Minh': 'Hồ Chí Minh', 'Ho Chi Minh': 'Hồ Chí Minh',
            'TP HCM': 'Hồ Chí Minh', 'TP.HCM': 'Hồ Chí Minh', 'HCM': 'Hồ Chí Minh',
            'Đà Nẵng': 'Đà Nẵng', 'Da Nang': 'Đà Nẵng',
            'Hải Phòng': 'Hải Phòng', 'Hai Phong': 'Hải Phòng',
            'Bình Dương': 'Bình Dương', 'Binh Duong': 'Bình Dương',
            'Đồng Nai': 'Đồng Nai', 'Dong Nai': 'Đồng Nai',
            'Cần Thơ': 'Cần Thơ', 'Can Tho': 'Cần Thơ',
        }

        dia_chi_cty = str(original_row.get('company_address', ''))
        if dia_chi_cty and dia_chi_cty != 'nan':
            for tu_khoa, ten_chuan in bang_thanh_pho.items():
                if tu_khoa in dia_chi_cty:
                    result['job_location'] = ten_chuan
                    break

        return result

    def _sua_boolean(self, result):
        """Đảm bảo has_commission là True/False, không phải NaN"""
        co_hh = result.get('has_commission')
        if co_hh is None or (isinstance(co_hh, float) and pd.isna(co_hh)):
            van_ban = (
                str(result.get('job_benefits', '')).lower() + ' ' +
                str(result.get('salary_note', '')).lower() + ' ' +
                str(result.get('job_description', '')).lower()
            )
            tu_khoa_hh = ['hoa hồng', 'thưởng doanh số', 'commission',
                          '% dự án', 'kpi', 'thu nhập không giới hạn',
                          'thuong doanh so', 'hoa hong']
            result['has_commission'] = any(kw in van_ban for kw in tu_khoa_hh)
        else:
            result['has_commission'] = bool(co_hh)
        return result

    def _sua_gioi_tinh(self, result, original_row):
        """
        Sửa giới tính -- LLM nhỏ hay nhầm.
        Ưu tiên: tiêu đề > 30 ký tự đầu yêu cầu > mặc định Không yêu cầu.
        """
        title = str(original_row.get('title', ''))
        title_lower = title.lower()
        gioi_tinh = str(result.get('gender_requirement', '')).strip()
        gia_tri_hop_le = {'Nam', 'Nữ', 'Không yêu cầu'}

        # Kiểm tra tiêu đề trước (đáng tin cậy nhất)
        # Pattern: "- Nữ", "(Nữ)", "– Nữ"
        if re.search(r'[-–(]\s*nữ\b', title_lower):
            result['gender_requirement'] = 'Nữ'
            return result
        if re.search(r'[-–(]\s*nam\b', title_lower):
            result['gender_requirement'] = 'Nam'
            return result

        # Nếu LLM trả giá trị hợp lệ thì giữ nguyên
        if gioi_tinh in gia_tri_hop_le:
            return result

        # Fallback: đọc 30 ký tự đầu phần yêu cầu
        yeu_cau = str(result.get('job_requirements', '')).lower()[:30]
        if 'nữ' in yeu_cau and 'nam' not in yeu_cau:
            result['gender_requirement'] = 'Nữ'
        elif 'nam' in yeu_cau and 'nữ' not in yeu_cau:
            result['gender_requirement'] = 'Nam'
        else:
            result['gender_requirement'] = 'Không yêu cầu'

        return result

    def _sua_hoc_van(self, result):
        """Chuẩn hóa education_level về 5 giá trị hợp lệ"""
        gia_tri_hop_le = ['Đại học', 'Cao đẳng', 'Trung cấp', 'Trung học', 'Không yêu cầu']
        trinh_do = str(result.get('education_level', '')).strip()

        if not trinh_do or trinh_do in ('nan', 'None', ''):
            result['education_level'] = ''
            return result

        for level in gia_tri_hop_le:
            if level.lower() in trinh_do.lower():
                result['education_level'] = level
                return result

        # Ánh xạ các biến thể phổ biến
        bang_anh_xa = {
            'trung học phổ thông': 'Trung học',
            'cấp 3': 'Trung học',
            'thpt': 'Trung học',
            'trung cấp nghề': 'Trung cấp',
            'cao đẳng nghề': 'Cao đẳng',
        }
        for bien_the, chuan in bang_anh_xa.items():
            if bien_the in trinh_do.lower():
                result['education_level'] = chuan
                return result

        result['education_level'] = ''
        return result

    def _trich_xuat_regex(self, original_row):
        """Trích xuất các trường có cấu trúc cố định bằng regex từ overview và job_details.
        
        Format UC (crawl_topcv.py): === MÔ TẢ CÔNG VIỆC ===
        """
        overview = str(original_row.get('overview', ''))
        job_details = str(original_row.get('job_details', ''))
        result = {}

        # --- Experience: từ overview ---
        exp_match = re.search(r'Kinh nghiệm[:\s]*\n?\s*(.+?)(?:\n|$)', overview)
        result['experience'] = exp_match.group(1).strip() if exp_match else ''

        # --- Requirements tags (Chuyên môn) ---
        chuyen_mon_match = re.search(
            r'Chuyên môn[:\s]*\n(.*?)(?:\n===|\n\n===|$)',
            overview, re.DOTALL
        )
        if chuyen_mon_match:
            lines = [l.strip() for l in chuyen_mon_match.group(1).strip().split('\n') if l.strip()]
            result['requirements_tags'] = ', '.join(lines)
        else:
            result['requirements_tags'] = ''

        # --- Specializations (Yêu cầu) ---
        yeu_cau_match = re.search(
            r'Yêu cầu[:\s]*\n(.*?)(?:\nQuyền lợi:|\nChuyên môn:|\n===|$)',
            overview, re.DOTALL
        )
        if yeu_cau_match:
            lines = [l.strip() for l in yeu_cau_match.group(1).strip().split('\n') if l.strip()]
            result['specializations'] = ', '.join(lines)
        else:
            result['specializations'] = ''

        # --- JD Sections: format UC (=== HEADER ===) ---
        def trich_section(text, tieu_de):
            """Tìm section theo format === HEADER ==="""
            pattern = r'===\s*' + re.escape(tieu_de) + r'\s*===\s*\n(.*?)(?:\n===|$)'
            match = re.search(pattern, text, re.DOTALL)
            return match.group(1).strip() if match else ''

        result['job_description']  = trich_section(job_details, 'MÔ TẢ CÔNG VIỆC')
        result['job_requirements'] = trich_section(job_details, 'YÊU CẦU ỨNG VIÊN')
        result['job_benefits']     = trich_section(job_details, 'QUYỀN LỢI')
        result['working_time']     = trich_section(job_details, 'THỜI GIAN LÀM VIỆC')

        return result

    # -----------------------------------------------------------------------
    # Hàm chính
    # -----------------------------------------------------------------------

    def _ap_phan_hoi(self, parsed, original_row):
        """
        Ghép kết quả LLM (salary, location, education, skills, gender)
        với kết quả regex (JD sections, overview tags) thành flat dict.
        """
        result = {}

        # Giữ nguyên các trường gốc từ crawl
        for key in ['title', 'url', 'company', 'company_address',
                    'crawled_date', 'category', 'content_hash',
                    'deadline', 'is_expired']:
            result[key] = original_row.get(key, '')

        # --- LLM: Lương ---
        salary = parsed.get('salary', {}) or {}
        result['job_salary']     = salary.get('text', '')
        result['salary_min']     = salary.get('min')
        result['salary_max']     = salary.get('max')
        result['salary_type']    = salary.get('type', '')
        result['salary_note']    = salary.get('note', '')
        result['has_commission'] = salary.get('has_commission', False)

        location = parsed.get('location', {}) or {}
        result['job_location'] = location.get('city', '')

        # --- LLM: Học vấn ---
        education = parsed.get('education', {}) or {}
        result['education_level'] = education.get('level', '')
        result['education_field'] = education.get('field', '')

        # --- LLM: Kỹ năng ---
        skills = parsed.get('skills', {}) or {}
        if isinstance(skills, list):
            tech = skills
            soft, langs, certs = [], [], []
        else:
            tech  = skills.get('technical', []) or []
            soft  = skills.get('soft', [])       or []
            langs = skills.get('languages', [])  or []
            certs = skills.get('certificates', []) or []

        def join_list(lst):
            return ', '.join(lst) if isinstance(lst, list) else str(lst)

        result['technical_skills'] = join_list(tech)
        result['soft_skills']      = join_list(soft)
        result['languages']        = join_list(langs)
        result['certificates']     = join_list(certs)
        # --- LLM: Giới tính ---
        result['gender_requirement'] = parsed.get('gender_requirement', '')

        # --- Regex: Overview tags + JD sections ---
        regex_data = self._trich_xuat_regex(original_row)
        result['experience']        = regex_data['experience']
        result['requirements_tags'] = regex_data['requirements_tags']
        result['specializations']   = regex_data['specializations']
        result['job_description']   = regex_data['job_description']
        result['job_requirements']  = regex_data['job_requirements']
        result['job_benefits']      = regex_data['job_benefits']
        result['working_time']      = regex_data['working_time']

        result = self._sua_luong(result, original_row)
        result = self._sua_dia_diem(result, original_row)
        result = self._sua_boolean(result)
        result = self._sua_gioi_tinh(result, original_row)
        result = self._sua_hoc_van(result)

        return result

    def clean_job(self, row):
        """Clean 1 job bằng LLM, trả về dict các trường đã chuẩn hóa"""
        title           = str(row.get('title', ''))
        company         = str(row.get('company', ''))
        company_address = str(row.get('company_address', ''))
        overview        = str(row.get('overview', ''))
        job_details     = str(row.get('job_details', ''))

        # Cắt bớt nếu quá dài (Qwen 3B context khoảng 4K tokens)
        if len(overview) > 1500:
            overview = overview[:1500] + "..."
        if len(job_details) > 2500:
            job_details = job_details[:2500] + "..."

        prompt = EXTRACT_PROMPT.format(
            title=title,
            company=company,
            company_address=company_address,
            overview=overview,
            job_details=job_details
        )

        raw, elapsed = self._goi_llm(prompt)
        self.stats["tong_thoi_gian"] += elapsed

        if not raw:
            self.stats["that_bai"] += 1
            return None

        parsed = self._phan_tich_json(raw)
        if not parsed:
            self.stats["that_bai"] += 1
            return None

        self.stats["thanh_cong"] += 1
        return self._ap_phan_hoi(parsed, row)

    def clean_batch(self, input_file, output_file, limit=0):
        """Clean toàn bộ file CSV"""
        logger.info("=" * 70)
        logger.info(f"  LLM CLEAN -- {self.model}")
        logger.info("=" * 70)

        if not self._kiem_tra_ollama():
            return False

        df = pd.read_csv(input_file, encoding='utf-8-sig')
        total = len(df) if limit == 0 else min(limit, len(df))
        logger.info(f"  Dữ liệu vào : {input_file} ({len(df)} dòng)")
        logger.info(f"  Xử lý       : {total} dòng")
        logger.info(f"  Dữ liệu ra  : {output_file}")
        logger.info("=" * 70)

        ket_qua = []
        thoi_gian_bat_dau = time.time()

        for i, (_, row) in enumerate(df.iterrows()):
            if limit > 0 and i >= limit:
                break

            ten_job = str(row.get('title', ''))[:60]
            logger.info(f"[{i+1}/{total}] {ten_job}...")

            cleaned = self.clean_job(row)

            if cleaned:
                ket_qua.append(cleaned)
                logger.info(f"  Thanh cong ({self.stats['tong_thoi_gian']:.1f}s tong cong)")
            else:
                logger.warning(f"  LLM that bai -- giu nguyen du lieu thu")
                fallback = {k: row.get(k, '') for k in row.index}
                ket_qua.append(fallback)

        if ket_qua:
            out_df = pd.DataFrame(ket_qua)
            out_df.to_csv(output_file, index=False, encoding='utf-8-sig')

            tong_thoi_gian = time.time() - thoi_gian_bat_dau
            so_thanh_cong = self.stats['thanh_cong']

            logger.info("=" * 70)
            logger.info("  HOAN THANH")
            logger.info("=" * 70)
            logger.info(f"  File ra         : {output_file}")
            logger.info(f"  So dong         : {len(ket_qua)}")
            logger.info(f"  Thanh cong      : {so_thanh_cong}/{total}")
            logger.info(f"  That bai        : {self.stats['that_bai']}/{total}")
            logger.info(f"  Tong thoi gian  : {tong_thoi_gian:.1f}s")
            logger.info(f"  Trung binh/job  : {self.stats['tong_thoi_gian']/max(so_thanh_cong,1):.1f}s")
            logger.info(f"  Cac truong      : {list(out_df.columns)}")
            logger.info("=" * 70)
            return True
        else:
            logger.error("Khong co ket qua!")
            return False


def main():
    parser = argparse.ArgumentParser(
        description='LLM Data Cleaner -- Clean dữ liệu tuyển dụng bằng Ollama',
    )
    parser.add_argument('--input',      required=True,               help='File CSV đầu vào (dữ liệu crawl thô)')
    parser.add_argument('--output',     required=True,               help='File CSV đầu ra (dữ liệu sạch)')
    parser.add_argument('--model',      default=DEFAULT_MODEL,       help=f'Ollama model (mặc định: {DEFAULT_MODEL})')
    parser.add_argument('--limit',      type=int, default=0,         help='Giới hạn số job xử lý (0 = tất cả)')
    parser.add_argument('--ollama-url', default=OLLAMA_URL,          help='Ollama API URL')

    args = parser.parse_args()

    cleaner = LLMCleaner(model=args.model, ollama_url=args.ollama_url)
    thanh_cong = cleaner.clean_batch(args.input, args.output, limit=args.limit)

    exit(0 if thanh_cong else 1)


if __name__ == '__main__':
    main()