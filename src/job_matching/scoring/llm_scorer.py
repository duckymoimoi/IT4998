"""
LLM Scorer - Đánh giá đa chiều CV-Job bằng LLM (Groq API).

Pipeline production:
  Hybrid Search (BM25 + kNN + ESCO) → top-N → LLM 6-dim scoring → Final ranking

Batch mode: gửi 1 CV + N jobs trong 1 request, nhưng yêu cầu LLM chấm từng
job độc lập, không so sánh thứ hạng giữa các job trong cùng batch.
"""

import os
import json
import logging
import re
import queue
import threading
import time
import unicodedata
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


DEFAULT_SCORING_MODEL = os.environ.get("GROQ_SCORING_MODEL", "openai/gpt-oss-120b")
DEFAULT_GROQ_DETAIL_MODEL = os.environ.get("GROQ_DETAIL_MODEL", "llama-3.3-70b-versatile")
DEFAULT_SCORING_MAX_COMPLETION_TOKENS = _env_int(
    "GROQ_SCORING_MAX_COMPLETION_TOKENS",
    3000,
)
DEFAULT_SCORING_PROVIDER = os.environ.get("SCORING_PROVIDER", "groq").strip().lower()
DEFAULT_DETAIL_SCORING_PROVIDER = os.environ.get(
    "DETAIL_SCORING_PROVIDER",
    DEFAULT_SCORING_PROVIDER,
).strip().lower()
DEFAULT_COHERE_SCORING_MODEL = os.environ.get("COHERE_SCORING_MODEL", "command-r-08-2024")
DEFAULT_COHERE_SCORING_MAX_TOKENS = _env_int("COHERE_SCORING_MAX_TOKENS", 2500)
DEFAULT_AEROLINK_BASE_URL = os.environ.get(
    "ANTHROPIC_BASE_URL",
    os.environ.get("AEROLINK_BASE_URL", "https://capi.aerolink.lat"),
)
DEFAULT_AEROLINK_SCORING_MODEL = os.environ.get(
    "AEROLINK_SCORING_MODEL",
    os.environ.get("CLAUDE_ANNOTATION_MODEL", "claude-sonnet-4-6"),
)
DEFAULT_AEROLINK_DETAIL_MODEL = os.environ.get(
    "AEROLINK_DETAIL_MODEL",
    os.environ.get("CLAUDE_ANNOTATION_MODEL", DEFAULT_AEROLINK_SCORING_MODEL),
)
DEFAULT_AEROLINK_SCORING_MAX_TOKENS = _env_int("AEROLINK_SCORING_MAX_TOKENS", 3000)
DEFAULT_AEROLINK_DETAIL_MAX_TOKENS = _env_int("AEROLINK_DETAIL_MAX_TOKENS", 5000)
DEFAULT_DETAIL_MAX_COMPLETION_TOKENS = _env_int("GROQ_DETAIL_MAX_COMPLETION_TOKENS", 5000)
DEFAULT_COHERE_DETAIL_MAX_TOKENS = _env_int("COHERE_DETAIL_MAX_TOKENS", 4000)
DEFAULT_SCORING_BATCH_WORKERS = _env_int("SCORING_BATCH_WORKERS", 0)
DEFAULT_SCORING_BATCH_SIZE = _env_int("SCORING_BATCH_SIZE", 5)
DEFAULT_SCORING_TIME_LIMIT_SECONDS = _env_int("SCORING_TIME_LIMIT_SECONDS", 30)
DEFAULT_SCORING_REQUEST_TIMEOUT_SECONDS = _env_int(
    "SCORING_REQUEST_TIMEOUT_SECONDS",
    30,
)

# Groq API keys (rotation để tránh rate limit)
_API_KEYS = []
_current_key_idx = 0
_COHERE_API_KEYS = []
_current_cohere_key_idx = 0
_AEROLINK_API_KEYS = []
_current_aerolink_key_idx = 0
_GROQ_KEY_LOCK = threading.RLock()
_COHERE_KEY_LOCK = threading.RLock()
_AEROLINK_KEY_LOCK = threading.RLock()

SCORING_DIMENSIONS = [
    "relevance",   # Phù hợp tổng thể
    "skills",      # Khớp kỹ năng
    "experience",  # Kinh nghiệm
    "education",   # Học vấn
    "location",    # Địa điểm
    "salary",      # Mức lương
]

DEFAULT_WEIGHTS = {
    "relevance": 0.25,
    "skills": 0.30,
    "experience": 0.15,
    "education": 0.10,
    "location": 0.10,
    "salary": 0.10,
}


def _load_api_keys():
    """Load tất cả Groq API keys từ environment."""
    global _API_KEYS
    if _API_KEYS:
        return

    keys = []
    # Thử GROQ_API_KEY_1, _2, _3, ...
    for i in range(1, 10):
        key = os.environ.get(f"GROQ_API_KEY_{i}")
        if key:
            keys.append(key)

    # Fallback: GROQ_API_KEY đơn
    if not keys:
        single = os.environ.get("GROQ_API_KEY")
        if single:
            keys.append(single)

    _API_KEYS = keys
    if keys:
        logger.info(f"Loaded {len(keys)} Groq API key(s)")
    else:
        logger.warning("No Groq API keys found — LLM scoring disabled")


def _get_next_key():
    """Round-robin key rotation."""
    global _current_key_idx
    with _GROQ_KEY_LOCK:
        _load_api_keys()
        if not _API_KEYS:
            return None
        key = _API_KEYS[_current_key_idx % len(_API_KEYS)]
        _current_key_idx += 1
        return key


def _get_groq_key_pool() -> List[str]:
    """Return configured Groq keys in stable order."""
    with _GROQ_KEY_LOCK:
        _load_api_keys()
        return list(_API_KEYS)


def _load_cohere_api_keys():
    """Load Cohere API keys from environment."""
    global _COHERE_API_KEYS
    if _COHERE_API_KEYS:
        return

    keys = []
    for i in range(1, 10):
        key = os.environ.get(f"COHERE_API_KEY_{i}")
        if key:
            keys.append(key)

    if not keys:
        single = os.environ.get("COHERE_API_KEY")
        if single:
            keys.append(single)

    _COHERE_API_KEYS = keys
    if keys:
        logger.info(f"Loaded {len(keys)} Cohere API key(s)")
    else:
        logger.warning("No Cohere API keys found")


def _get_next_cohere_key():
    """Round-robin Cohere key rotation."""
    global _current_cohere_key_idx
    with _COHERE_KEY_LOCK:
        _load_cohere_api_keys()
        if not _COHERE_API_KEYS:
            return None
        key = _COHERE_API_KEYS[_current_cohere_key_idx % len(_COHERE_API_KEYS)]
        _current_cohere_key_idx += 1
        return key


def _get_cohere_key_pool() -> List[str]:
    """Return configured Cohere keys in stable order."""
    with _COHERE_KEY_LOCK:
        _load_cohere_api_keys()
        return list(_COHERE_API_KEYS)


def _load_aerolink_api_keys():
    """Load Aerolink API keys from environment."""
    global _AEROLINK_API_KEYS
    if _AEROLINK_API_KEYS:
        return

    keys = []
    for prefix in ("ANTHROPIC_API_KEY", "AEROLINK_API_KEY", "aero_api_key"):
        for i in range(1, 10):
            key = os.environ.get(f"{prefix}_{i}")
            if key and key not in keys:
                keys.append(key)
        single = os.environ.get(prefix)
        if single and single not in keys:
            keys.append(single)

    _AEROLINK_API_KEYS = keys
    if keys:
        logger.info(f"Loaded {len(keys)} Aerolink API key(s)")
    else:
        logger.warning("No Aerolink API keys found")


def _get_next_aerolink_key():
    """Round-robin Aerolink key rotation."""
    global _current_aerolink_key_idx
    with _AEROLINK_KEY_LOCK:
        _load_aerolink_api_keys()
        if not _AEROLINK_API_KEYS:
            return None
        key = _AEROLINK_API_KEYS[_current_aerolink_key_idx % len(_AEROLINK_API_KEYS)]
        _current_aerolink_key_idx += 1
        return key


def _get_aerolink_key_pool() -> List[str]:
    """Return configured Aerolink keys in stable order."""
    with _AEROLINK_KEY_LOCK:
        _load_aerolink_api_keys()
        return list(_AEROLINK_API_KEYS)


def _resolve_provider_config(
    model: Optional[str] = None,
    max_retries: Optional[int] = None,
    detail: bool = False,
) -> tuple[str, str, int]:
    """Resolve the explicitly configured scoring provider and its limits."""
    provider = DEFAULT_DETAIL_SCORING_PROVIDER if detail else DEFAULT_SCORING_PROVIDER
    if provider == "cohere":
        _load_cohere_api_keys()
        return (
            provider,
            model or DEFAULT_COHERE_SCORING_MODEL,
            max_retries if max_retries is not None else max(3, len(_COHERE_API_KEYS) or 1),
        )
    if provider == "aerolink":
        _load_aerolink_api_keys()
        return (
            provider,
            model or (DEFAULT_AEROLINK_DETAIL_MODEL if detail else DEFAULT_AEROLINK_SCORING_MODEL),
            max_retries if max_retries is not None else max(3, len(_AEROLINK_API_KEYS) or 1),
        )

    _load_api_keys()
    return (
        "groq",
        model or (DEFAULT_GROQ_DETAIL_MODEL if detail else DEFAULT_SCORING_MODEL),
        max_retries if max_retries is not None else max(3, len(_API_KEYS) or 1),
    )


def _truncate_prompt_text(value, limit: int) -> str:
    """Trim prompt evidence near a natural boundary without exceeding limit."""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    candidate = text[:limit]
    boundary = max(candidate.rfind("\n"), candidate.rfind(". "), candidate.rfind("; "))
    if boundary >= int(limit * 0.65):
        candidate = candidate[:boundary + 1]
    return candidate.rstrip() + "..."


def _clean_prompt_value(value) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"none", "nan", "null"} else text


def _split_prompt_terms(value) -> List[str]:
    text = _clean_prompt_value(value)
    if not text:
        return []
    parts = re.split(r"[,;|/]\s*|\n+", text)
    return [_clean_prompt_value(part) for part in parts if _clean_prompt_value(part)]


def _dedupe_prompt_terms(values: List[str], limit: int = 16) -> List[str]:
    seen = set()
    result = []
    for value in values:
        key = _fold_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def _has_phrase_evidence(evidence: str, term: str) -> bool:
    term = _clean_prompt_value(term)
    if not term:
        return False
    pattern = r"(?<!\w)" + re.escape(term).replace(r"\ ", r"\s+") + r"(?!\w)"
    return bool(re.search(pattern, evidence, flags=re.I | re.UNICODE))


def _semantic_requirement_terms(job: dict) -> List[str]:
    """Use taxonomy-backed semantic profile terms before noisy raw skills."""
    semantic_text = _clean_prompt_value(job.get("semantic_text"))
    if not semantic_text:
        return []

    useful_labels = {
        "linh vuc",
        "nghiep vu",
        "ky nang ky thuat",
        "cong cu",
        "ngoai ngu",
        "chung chi",
    }
    terms = []
    for match in re.finditer(r"(?:^|\.\s+)([^.:]{1,48}):\s*([^.]*)", semantic_text):
        label = _fold_text(match.group(1))
        if label in useful_labels:
            terms.extend(_split_prompt_terms(match.group(2)))
    return _dedupe_prompt_terms(terms)


def _job_requirement_terms_for_prompt(job: dict) -> str:
    """Prefer curated/evidence-backed job requirements over stale raw skills."""
    req_tags = _clean_prompt_value(job.get("requirements_tags"))
    if req_tags:
        return req_tags

    semantic_terms = _semantic_requirement_terms(job)
    if semantic_terms:
        return ", ".join(semantic_terms)

    raw_terms = _split_prompt_terms(job.get("technical_skills"))
    if not raw_terms:
        return ""

    evidence = " ".join(
        _clean_prompt_value(job.get(field))
        for field in ["title", "job_description", "job_requirements"]
    )
    kept = [term for term in raw_terms if _has_phrase_evidence(evidence, term)]
    return ", ".join(_dedupe_prompt_terms(kept))


def _format_job_for_prompt(job: dict, index: int) -> str:
    """Format 1 job thành text ngắn gọn cho LLM prompt."""
    title = _clean_prompt_value(job.get("title")) or "N/A"
    company = _clean_prompt_value(job.get("company") or job.get("company_name")) or "N/A"
    salary = job.get("job_salary", "Thỏa thuận")
    location = _clean_prompt_value(job.get("job_location")) or "N/A"
    experience = _clean_prompt_value(job.get("experience") or job.get("job_experience")) or "N/A"
    education_req = _clean_prompt_value(job.get("education_level") or job.get("job_education"))
    education_field = _clean_prompt_value(job.get("education_field"))
    requirements = _truncate_prompt_text(job.get("job_requirements", ""), 1200)
    description = _truncate_prompt_text(job.get("job_description", ""), 200)
    requirement_terms = _job_requirement_terms_for_prompt(job)
    specializations = _clean_prompt_value(job.get("specializations"))
    languages = _clean_prompt_value(job.get("languages"))
    certificates = _clean_prompt_value(job.get("certificates"))

    return f"""[JOB {index + 1}] {title}
Công ty: {company}
Nhóm chuyên môn: {specializations}
Lương: {salary} | Địa điểm: {location} | Kinh nghiệm yêu cầu: {experience}
Yêu cầu học vấn: {education_req} {education_field}
Kỹ năng/yêu cầu chính: {requirement_terms or "Không nêu rõ"}
Ngoại ngữ yêu cầu: {languages or "Không nêu rõ"}
Chứng chỉ yêu cầu: {certificates or "Không nêu rõ"}
Mô tả công việc: {description}
Yêu cầu chi tiết: {requirements}"""


def _format_cv_for_prompt(cv_data: dict) -> str:
    """Format CV data thành text cho LLM prompt."""
    target_roles = cv_data.get("target_roles", "")
    skills = cv_data.get("skills", "")
    experience = cv_data.get("experience", "Không rõ")
    education = cv_data.get("education", "Không rõ")
    location = cv_data.get("location", "Không rõ")
    salary = cv_data.get("salary", "Thỏa thuận")

    edu_map = {
        "dai_hoc": "Đại học",
        "cao_dang": "Cao đẳng",
        "trung_cap": "Trung cấp",
        "trung_hoc": "Trung học",
    }
    edu_text = edu_map.get(education, education)

    exp_map = {
        "no_requirement": "Chưa có",
        "under_1": "Dưới 1 năm",
        "1": "1 năm",
        "2": "2 năm",
        "3": "3 năm",
        "4": "4 năm",
        "5": "5 năm",
        "over_5": "Trên 5 năm",
        "all": "Không giới hạn",
    }
    exp_text = exp_map.get(str(experience), str(experience))

    languages = cv_data.get("languages", "")
    certificates = cv_data.get("certificates", "")

    result = ""
    if target_roles:
        result += f"Vai trò mục tiêu: {target_roles}\n"
    result += f"Kỹ năng chuyên môn: {skills}"
    if languages:
        result += f"\nNgoại ngữ: {languages}"
    if certificates:
        result += f"\nChứng chỉ: {certificates}"
    result += f"\nKinh nghiệm tổng số: {exp_text}"
    result += "\nLưu ý kinh nghiệm: đây là tổng số năm do ứng viên khai báo; chỉ coi là kinh nghiệm đúng vai trò/domain nếu kỹ năng hoặc mô tả CV chứng minh."
    result += f"\nHọc vấn: {edu_text}"
    result += f"\nĐịa điểm mong muốn: {location}"
    result += f"\nMức lương mong muốn: {salary} triệu VND"

    cv_markdown = str(cv_data.get("cv_markdown") or cv_data.get("raw_text") or "").strip()
    if cv_markdown:
        result += "\n\nFULL CV MARKDOWN (dùng để kiểm tra kinh nghiệm theo vai trò, dự án và domain):"
        result += f"\n{cv_markdown[:6000]}"

    return result


def _build_batch_prompt(cv_data: dict, jobs: List[dict]) -> str:
    """Tạo prompt batch dựa trên rubric đánh giá CV-JD của hệ thống."""
    cv_text = _format_cv_for_prompt(cv_data)
    jobs_text = "\n\n".join(
        _format_job_for_prompt(job, i) for i, job in enumerate(jobs)
    )
    n = len(jobs)

    return f"""# ĐÁNH GIÁ MỨC ĐỘ PHÙ HỢP CV-JD (Batch {n} jobs)

## Bối cảnh
Bạn là chuyên gia tuyển dụng Việt Nam (TopCV). Đánh giá 1 ứng viên với {n} vị trí.
Mỗi JOB phải được chấm độc lập như một cặp CV-JD riêng; không so sánh tương đối,
không cố cân bằng điểm giữa các JOB trong batch, và không để JOB trước/sau ảnh hưởng đến điểm.

## THÔNG TIN ỨNG VIÊN
{cv_text}

## DANH SÁCH CÔNG VIỆC
{jobs_text}

## RUBRIC CHẤM ĐIỂM (thang 0-10, số nguyên)

### 1. relevance — Phù hợp tổng thể
Ứng viên có NÊN được mời phỏng vấn không? Đánh giá theo 3 lớp:
1) đúng vai trò/chức năng công việc, 2) đúng tech stack/kỹ năng lõi, 3) đúng domain/ngữ cảnh ngành nếu JD nhấn mạnh.
| 0-2: Hoàn toàn không liên quan | 3-4: Liên quan xa | 5-6: Tiềm năng, thiếu nhiều | 7-8: Phù hợp tốt | 9-10: Lý tưởng |
Nếu CV chỉ có tổng số năm nhưng không chứng minh đúng vai trò/domain, relevance KHÔNG được vượt 6-7.
VD: CV có Java nhưng thiên AI/frontend, JD là Backend Java Bank → phải nêu thiếu bằng chứng backend/banking, không được chấm như backend Java đủ kinh nghiệm.

### 2. skills — Khớp kỹ năng (QUAN TRỌNG NHẤT)
So sánh kỹ năng CV vs yêu cầu JD theo kỹ năng lõi của VAI TRÒ.
| 0-2: <20% khớp | 3-4: 20-40% | 5-6: 40-60%, thiếu skill core | 7-8: 60-80% | 9-10: >80% |
Không chỉ đếm keyword. Phân biệt:
- Java Backend: Java + Spring/Spring Boot + API/backend/service + SQL/database là lõi.
- AI/ML: Python + ML/DL framework + model/data pipeline là lõi.
- Frontend: JS/TS + React/Vue + UI/API integration là lõi.
Nếu chỉ khớp 1 keyword lớn (VD: Java) nhưng thiếu stack lõi (VD: Spring, backend API, SQL), skills tối đa 5-6.

### Đối chiếu ngoại ngữ và chứng chỉ
Khi JD yêu cầu ngoại ngữ hoặc chứng chỉ ngoại ngữ, phải đối chiếu theo cùng ngôn ngữ trước khi kết luận thiếu:
- Tiếng Anh: IELTS, TOEIC, TOEFL, VSTEP, Cambridge đều là bằng chứng năng lực tiếng Anh, nhưng khác hệ quy đổi.
- Tiếng Nhật: JLPT là bằng chứng năng lực tiếng Nhật.
- Tiếng Trung: HSK là bằng chứng năng lực tiếng Trung.
- Tiếng Hàn: TOPIK là bằng chứng năng lực tiếng Hàn.
Nếu CV có chứng chỉ cùng ngôn ngữ ở mức cao hơn hoặc tương đương, KHÔNG viết là "thiếu chứng chỉ" một cách máy móc. Ví dụ: CV có IELTS 8.5 và JD cần tiếng Anh tốt/TOEIC thì phải ghi là tiếng Anh mạnh; chỉ nêu thiếu TOEIC nếu JD bắt buộc đúng TOEIC và không chấp nhận chứng chỉ tương đương.

### 3. experience — Kinh nghiệm ĐÚNG VAI TRÒ/DOMAIN
Không được chấm experience chỉ bằng số năm. Tách 3 yếu tố:
- Số năm so với JD (40%)
- Kinh nghiệm đúng vai trò/stack (40%): backend Java, frontend, AI, BA, sales...
- Kinh nghiệm đúng domain/ngữ cảnh (20%): banking/fintech, ecommerce, xây dựng, giáo dục... nếu JD nêu rõ.
| 0-2: Chênh rất lớn | 3-4: Thiếu/thừa đáng kể | 5-6: Thiếu/thừa nhẹ | 7-8: ±1 năm | 9-10: Đúng |
Job "Không yêu cầu KN": Fresher=10, 1-2 năm=8-9, 3-5 năm=5-6, 5+ năm=3-4
Over-qualification: Job cần 1-2 năm, CV 5+ năm → MAX 5 điểm
Nếu CV có 3 năm tổng nhưng không thể hiện là 3 năm backend Java/banking thì KHÔNG viết "KN 3 năm đạt yêu cầu"; phải viết "tổng KN 3 năm, nhưng chưa thấy kinh nghiệm backend Java/banking".
Nếu JD yêu cầu Bank/Fintech/Onsite Bank, kinh nghiệm ngân hàng là lợi thế riêng; thiếu domain này phải nêu là rủi ro, không tự suy diễn từ số năm.

### 4. education — Học vấn
| 0-3: Không đạt | 4-6: Thiếu 1 bậc | 7-8: Đạt yêu cầu | 9-10: Đúng/vượt | Job không nêu rõ → 7 |

### 5. salary — Lương
| 0-3: Job thấp hơn kỳ vọng rất nhiều | 4-5: Thấp hơn | 5-6: Thỏa thuận/không rõ | 7-8: Trong kỳ vọng | 9-10: Cao hơn kỳ vọng |
KHÔNG chấm location (tính riêng bằng Goong Maps GPS).

## NHẬN XÉT (comment)
Viết đúng 1 câu tiếng Việt ngắn, tối đa 30 từ, có tính hướng dẫn hành động. PHẢI có đủ:
- Khớp gì: kỹ năng/role/domain nào đang khớp.
- Thiếu/rủi ro gì: kỹ năng lõi, vai trò, domain, số năm đúng vai trò.
- Nên cải thiện/ứng tuyển thế nào: bổ sung skill/domain nào hoặc phù hợp hơn với loại job nào.
KHÔNG viết chung chung "phù hợp", "khớp", "KN đạt yêu cầu" nếu không nêu rõ kinh nghiệm thuộc vai trò/domain nào.
Giữ comment ngắn để JSON không bị cắt cụt khi chấm batch.

## OUTPUT — CHỈ JSON array, không giải thích
[
  {{"job": 1, "relevance": 8, "skills": 8, "experience": 7, "education": 9, "salary": 7, "comment": "Khớp Python/React, còn nên bổ sung CI/CD và backend API rõ hơn."}},
  {{"job": 2, "relevance": 5, "skills": 5, "experience": 5, "education": 8, "salary": 6, "comment": "Có Java nhưng thiếu Spring/backend API và kinh nghiệm banking, nên bổ sung dự án đúng domain."}}
]"""


def _build_cohere_batch_prompt(cv_data: dict, jobs: List[dict]) -> str:
    """Cohere is more likely to copy examples, so use a schema-only ending."""
    prompt = _build_batch_prompt(cv_data, jobs)
    marker = "## OUTPUT"
    if marker in prompt:
        prompt = prompt.split(marker, 1)[0].rstrip()

    n = len(jobs)
    return f"""{prompt}

## OUTPUT
Trả về DUY NHẤT một JSON array có đúng {n} object, theo đúng thứ tự JOB 1 đến JOB {n}.
Không sao chép ví dụ, không thêm markdown, không thêm giải thích.

Mỗi object bắt buộc có schema:
{{"job": <số thứ tự job>, "relevance": <int 0-10>, "skills": <int 0-10>, "experience": <int 0-10>, "education": <int 0-10>, "salary": <int 0-10>, "comment": "<2-3 câu tiếng Việt, nêu rõ khớp gì, thiếu gì, nên cải thiện gì>"}}
"""


def _build_detail_evidence_prompt(cv_data: dict, job: dict) -> str:
    """Build a focused one-job prompt that returns evidence plus scores."""
    cv_text = _format_cv_for_prompt(cv_data)
    job_text = _format_job_for_prompt(job, 0)

    return f"""# CHẤM CHI TIẾT CV-JD BẰNG EVIDENCE SCHEMA

Bạn là chuyên gia tuyển dụng Việt Nam. Hãy phân tích 1 ứng viên với 1 công việc.
Mục tiêu là tạo bằng chứng trung gian có cấu trúc trước khi cho điểm.

## THÔNG TIN ỨNG VIÊN
{cv_text}

## CÔNG VIỆC
{job_text}

## QUY TẮC ĐỌC HIỂU
- Không suy diễn kinh nghiệm đúng vai trò/domain chỉ từ tổng số năm.
- `matched_exact` chỉ gồm kỹ năng khớp trực tiếp hoặc được CV chứng minh rõ.
- `matched_synonym_or_esco` gồm kỹ năng gần tương đương, ví dụ REST API ~ backend API.
- `transferable` gồm kỹ năng hỗ trợ gián tiếp, không thay thế kỹ năng cốt lõi.
- Nếu JD yêu cầu domain cụ thể như banking/fintech/viễn thông mà CV không có bằng chứng, phải ghi rõ trong `experience_evidence.gap_note`.
- Không chấm location; location được hệ thống tính riêng bằng rule.
- Với salary: nếu lương job cao hơn hoặc nằm trong kỳ vọng của ứng viên thì điểm cao; chỉ chấm thấp khi lương job thấp hơn kỳ vọng đáng kể. Nếu job ghi thỏa thuận/không rõ thì cho điểm trung tính 5-6.
- Với ngoại ngữ/chứng chỉ ngoại ngữ, đối chiếu theo cùng ngôn ngữ. IELTS/TOEIC/TOEFL/VSTEP/Cambridge đều là bằng chứng tiếng Anh; JLPT là tiếng Nhật; HSK là tiếng Trung; TOPIK là tiếng Hàn. Không kết luận thiếu TOEIC nếu CV có IELTS cao và JD chỉ yêu cầu tiếng Anh hoặc chấp nhận chứng chỉ tương đương.
- Không tự chấm điểm. Hệ thống sẽ tính điểm bằng công thức từ evidence bạn trả về.

## OUTPUT
Chỉ trả về JSON array có đúng 1 object, không markdown, không giải thích ngoài JSON.
Object bắt buộc có schema:
[
  {{
    "job": 1,
    "required_core_skills": ["kỹ năng cốt lõi JD yêu cầu"],
    "matched_exact": ["kỹ năng CV khớp trực tiếp"],
    "matched_synonym_or_esco": ["kỹ năng CV gần tương đương"],
    "transferable": ["kỹ năng chuyển giao/hỗ trợ"],
    "missing_core": ["kỹ năng cốt lõi còn thiếu"],
    "experience_evidence": {{
      "cv_total_years": "số năm hoặc mô tả từ CV",
      "jd_required_years": "số năm hoặc mô tả từ JD",
      "role_match": "strong|partial|weak|none",
      "domain_match": "strong|partial|weak|none|not_required",
      "gap_note": "giải thích ngắn về chênh lệch kinh nghiệm/vai trò/domain"
    }},
    "education_gap": "đạt/thiếu/vượt yêu cầu học vấn, kèm lý do ngắn",
    "salary_note": "so sánh lương kỳ vọng với lương job",
    "comment": "2-3 câu tiếng Việt: khớp gì, thiếu gì, nên cải thiện gì"
  }}
]
"""


def _split_evenly(items: List[dict], parts: int) -> List[tuple[int, List[dict]]]:
    """Split items into near-even chunks while preserving original offsets."""
    if not items:
        return []
    parts = max(1, min(parts, len(items)))
    base, extra = divmod(len(items), parts)
    chunks = []
    start = 0
    for idx in range(parts):
        size = base + (1 if idx < extra else 0)
        chunk = items[start:start + size]
        if chunk:
            chunks.append((start, chunk))
        start += size
    return chunks


def score_batch(
    cv_data: dict,
    jobs: List[dict],
    weights: Optional[Dict[str, float]] = None,
    model: Optional[str] = None,
    max_retries: Optional[int] = None,
) -> List[dict]:
    """
    Batch scoring: 1 CV + N jobs → LLM → 6-dim scores + WSM total.

    Args:
        cv_data: dict CV (skills, experience, education, location, salary)
        jobs: list[dict] — top-N jobs từ retrieval
        weights: dict trọng số 6 chiều (default: DEFAULT_WEIGHTS)
        model: Groq model name
        max_retries: số lần retry nếu API fail

    Returns:
        list[dict] — mỗi item có:
            scores: {relevance, skills, experience, education, location, salary}
            total: WSM weighted score (0-10)
            job_index: vị trí trong input list
    """
    if not jobs:
        return []

    w = weights or DEFAULT_WEIGHTS
    provider, model, max_retries = _resolve_provider_config(model, max_retries)

    # Gọi LLM (có timing)
    t0 = time.time()
    if provider == "cohere":
        raw_scores = _call_parallel_batches(
            provider, cv_data, jobs, model, _get_cohere_key_pool(), max_retries
        )
    elif provider == "aerolink":
        raw_scores = _call_parallel_batches(
            provider, cv_data, jobs, model, _get_aerolink_key_pool(), max_retries
        )
    else:
        raw_scores = _call_parallel_batches(
            provider, cv_data, jobs, model, _get_groq_key_pool(), max_retries
        )
    llm_elapsed = time.time() - t0
    logger.info(f"LLM scoring ({provider}): {len(jobs)} jobs in {llm_elapsed:.1f}s")

    if not raw_scores:
        logger.warning("LLM scoring failed — returning empty scores")
        return [_empty_score(i) for i in range(len(jobs))]

    # Parse + tính WSM
    results = []
    for i, job in enumerate(jobs):
        s = raw_scores[i] if i < len(raw_scores) else {}
        if not s or s.get("_fallback"):
            results.append(_empty_score(i))
            continue

        scores = {}
        for dim in SCORING_DIMENSIONS:
            val = s.get(dim, 5)
            scores[dim] = _coerce_score(val)

        total = sum(scores[dim] * w.get(dim, 0) for dim in SCORING_DIMENSIONS)
        comment = s.get("comment", "")

        results.append({
            "job_index": i,
            "scores": scores,
            "total": round(total, 2),
            "comment": comment,
            "llm_time": round(llm_elapsed, 1),
        })

    return results


def score_detail_with_evidence(
    cv_data: dict,
    job: dict,
    weights: Optional[Dict[str, float]] = None,
    model: Optional[str] = None,
    max_retries: Optional[int] = None,
) -> dict:
    """
    Focused one-job scoring for the detail view.

    Unlike score_batch(), this asks the LLM for intermediate evidence schema
    (required skills, exact/synonym/transferable matches, missing core skills,
    and role/domain experience evidence). It is intentionally used on demand
    because it costs more tokens than fast batch scoring.
    """
    w = weights or DEFAULT_WEIGHTS
    provider, model, max_retries = _resolve_provider_config(model, max_retries, detail=True)

    prompt = _build_detail_evidence_prompt(cv_data, job)
    t0 = time.time()
    if provider == "cohere":
        raw = _call_cohere_prompt(
            prompt, model, max_retries,
            max_tokens=DEFAULT_COHERE_DETAIL_MAX_TOKENS,
        )
    elif provider == "aerolink":
        raw = _call_aerolink_prompt(
            prompt, model, max_retries,
            max_tokens=DEFAULT_AEROLINK_DETAIL_MAX_TOKENS,
        )
    else:
        raw = _call_groq_prompt(
            prompt, model, max_retries,
            max_tokens=DEFAULT_DETAIL_MAX_COMPLETION_TOKENS,
        )
    llm_elapsed = time.time() - t0

    parsed = _parse_scores_json(raw or "", 1) if raw else None
    if not parsed:
        logger.warning("Detail evidence scoring failed - fallback to fast score")
        fallback = score_batch(cv_data, [job], weights=w, model=model, max_retries=max_retries)
        if fallback:
            result = fallback[0]
            result["evidence"] = {}
            result["fallback"] = True
            return result
        return _empty_score(0)

    item = parsed[0] or {}
    if item.get("_fallback"):
        logger.warning("Detail evidence response omitted the requested job - fallback to fast score")
        fallback = score_batch(cv_data, [job], weights=w, model=model, max_retries=max_retries)
        if fallback:
            result = fallback[0]
            result["evidence"] = {}
            result["fallback"] = True
            return result
        return _empty_score(0)
    evidence = {
        "required_core_skills": _as_list(item.get("required_core_skills")),
        "matched_exact": _as_list(item.get("matched_exact")),
        "matched_synonym_or_esco": _as_list(item.get("matched_synonym_or_esco")),
        "transferable": _as_list(item.get("transferable")),
        "missing_core": _as_list(item.get("missing_core")),
        "experience_evidence": item.get("experience_evidence") if isinstance(item.get("experience_evidence"), dict) else {},
        "education_gap": str(item.get("education_gap", "") or ""),
        "salary_note": str(item.get("salary_note", "") or ""),
    }
    scores, score_basis = _compute_detail_scores(cv_data, job, evidence)
    evidence["score_basis"] = score_basis

    total = sum(scores[dim] * w.get(dim, 0) for dim in SCORING_DIMENSIONS)

    return {
        "job_index": 0,
        "scores": scores,
        "total": round(total, 2),
        "comment": str(item.get("comment", "") or ""),
        "evidence": evidence,
        "scoring_method": "formula_from_llm_evidence",
        "llm_time": round(llm_elapsed, 1),
    }


def _call_parallel_batches(
    provider: str,
    cv_data: dict,
    jobs: List[dict],
    model: str,
    key_pool: List[str],
    max_retries: int,
) -> Optional[List[dict]]:
    """Score fixed-size batches through a shared queue of API-key workers.

    A key that fails leaves its current batch in the queue for another healthy
    key. The function returns partial results when the global time limit ends.
    """
    if not key_pool:
        return None
    batch_size = max(1, DEFAULT_SCORING_BATCH_SIZE)
    if len(jobs) <= batch_size:
        if provider == "cohere":
            return _call_cohere_batch(cv_data, jobs, model, max_retries)
        if provider == "aerolink":
            return _call_aerolink_batch(cv_data, jobs, model, max_retries)
        return _call_groq_batch(cv_data, jobs, model, max_retries)

    chunks = [
        (offset, jobs[offset:offset + batch_size])
        for offset in range(0, len(jobs), batch_size)
    ]
    configured_workers = DEFAULT_SCORING_BATCH_WORKERS or len(key_pool)
    worker_count = min(configured_workers, len(key_pool), len(chunks))
    logger.info(
        "Scoring queue start: provider=%s, jobs=%s, batches=%s, keys=%s, workers=%s, batch_size=%s, limit=%ss",
        provider,
        len(jobs),
        len(chunks),
        len(key_pool),
        worker_count,
        batch_size,
        DEFAULT_SCORING_TIME_LIMIT_SECONDS,
    )
    raw_scores: List[dict] = [{} for _ in jobs]
    task_queue: queue.Queue = queue.Queue()
    for chunk in chunks:
        task_queue.put(chunk)

    result_lock = threading.Lock()
    stop_event = threading.Event()
    completed_offsets = set()
    deadline = time.monotonic() + max(1, DEFAULT_SCORING_TIME_LIMIT_SECONDS)

    def worker(api_key: str):
        while not stop_event.is_set() and time.monotonic() < deadline:
            try:
                offset, chunk_jobs = task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            remaining = max(1, int(deadline - time.monotonic()))
            request_timeout = min(DEFAULT_SCORING_REQUEST_TIMEOUT_SECONDS, remaining)
            try:
                if provider == "cohere":
                    scores = _call_cohere_batch(
                        cv_data, chunk_jobs, model, max_retries=1,
                        api_key=api_key, request_timeout=request_timeout,
                    )
                elif provider == "aerolink":
                    scores = _call_aerolink_batch(
                        cv_data, chunk_jobs, model, max_retries=1,
                        api_key=api_key, request_timeout=request_timeout,
                    )
                else:
                    scores = _call_groq_batch(
                        cv_data, chunk_jobs, model, max_retries=1,
                        api_key=api_key, request_timeout=request_timeout,
                    )
            except Exception as exc:
                logger.error("Scoring queue worker failed: %s", exc)
                scores = None
            finally:
                task_queue.task_done()

            if not scores:
                # Treat this key as unavailable for the current search. A
                # healthy worker can pick up the returned batch.
                task_queue.put((offset, chunk_jobs))
                logger.warning(
                    "Scoring key unavailable; returned batch offset=%s to queue",
                    offset,
                )
                return

            if stop_event.is_set() or time.monotonic() >= deadline:
                return

            with result_lock:
                if offset not in completed_offsets:
                    for local_idx, score in enumerate(scores[:len(chunk_jobs)]):
                        raw_scores[offset + local_idx] = score
                    completed_offsets.add(offset)
                    if len(completed_offsets) == len(chunks):
                        stop_event.set()

    threads = [
        threading.Thread(target=worker, args=(key_pool[idx],), daemon=True)
        for idx in range(worker_count)
    ]
    for thread in threads:
        thread.start()

    while time.monotonic() < deadline and not stop_event.is_set():
        if not any(thread.is_alive() for thread in threads):
            break
        time.sleep(0.1)
    stop_event.set()
    for thread in threads:
        thread.join(timeout=0.5)

    scored_count = sum(1 for score in raw_scores if score)
    remaining_batches = len(chunks) - len(completed_offsets)
    if remaining_batches:
        logger.warning(
            "Scoring stopped with %s unfinished batch(es); unscored jobs keep retrieval order",
            remaining_batches,
        )
    logger.info(
        "Scoring queue finished: %s/%s jobs, %s/%s batches, limit=%ss",
        scored_count, len(jobs), len(completed_offsets), len(chunks),
        DEFAULT_SCORING_TIME_LIMIT_SECONDS,
    )

    if all(not score for score in raw_scores):
        return None
    return raw_scores


def _call_groq_batch(
    cv_data: dict,
    jobs: List[dict],
    model: str,
    max_retries: int,
    api_key: Optional[str] = None,
    request_timeout: Optional[int] = None,
) -> Optional[List[dict]]:
    """Gọi Groq API với retry + key rotation."""
    try:
        from groq import Groq
    except ImportError:
        logger.error("groq package not installed — pip install groq")
        return None

    prompt = _build_batch_prompt(cv_data, jobs)

    for attempt in range(max_retries):
        selected_key = api_key or _get_next_key()
        if not selected_key:
            logger.error("No Groq API keys available")
            return None

        try:
            client = Groq(
                api_key=selected_key,
                timeout=request_timeout or DEFAULT_SCORING_REQUEST_TIMEOUT_SECONDS,
                max_retries=0,
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là hệ thống chấm điểm. Chỉ trả về JSON array, không giải thích.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_completion_tokens=DEFAULT_SCORING_MAX_COMPLETION_TOKENS,
            )

            text = response.choices[0].message.content.strip()
            return _parse_scores_json(text, len(jobs))

        except Exception as e:
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str:
                logger.warning(
                    f"Groq rate limit hit; switching key immediately "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                continue
            else:
                logger.error(f"Groq API error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

    return None


def _call_groq_prompt(
    prompt: str,
    model: str,
    max_retries: int,
    max_tokens: int,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Call Groq with an already-built prompt and return raw text."""
    try:
        from groq import Groq
    except ImportError:
        logger.error("groq package not installed - pip install groq")
        return None

    for attempt in range(max_retries):
        selected_key = api_key or _get_next_key()
        if not selected_key:
            logger.error("No Groq API keys available")
            return None

        try:
            client = Groq(api_key=selected_key, max_retries=0)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là hệ thống chấm điểm. Chỉ trả về JSON array hợp lệ, không markdown.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_completion_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str:
                logger.warning(
                    f"Groq rate limit hit; switching key immediately "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                continue
            logger.error(f"Groq API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return None


def _call_cohere_batch(
    cv_data: dict,
    jobs: List[dict],
    model: str,
    max_retries: int,
    api_key: Optional[str] = None,
    request_timeout: Optional[int] = None,
) -> Optional[List[dict]]:
    """Call Cohere Chat v2 via REST with retry + key rotation."""
    try:
        import requests
    except ImportError:
        logger.error("requests package not installed")
        return None

    prompt = _build_cohere_batch_prompt(cv_data, jobs)
    url = "https://api.cohere.com/v2/chat"

    for attempt in range(max_retries):
        selected_key = api_key or _get_next_cohere_key()
        if not selected_key:
            logger.error("No Cohere API keys available")
            return None

        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {selected_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Bạn là hệ thống chấm điểm. Chỉ trả về JSON array, không giải thích.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": DEFAULT_COHERE_SCORING_MAX_TOKENS,
                },
                timeout=request_timeout or DEFAULT_SCORING_REQUEST_TIMEOUT_SECONDS,
            )

            if response.status_code == 429:
                logger.warning(
                    f"Cohere rate limit hit; switching key immediately "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                continue

            if response.status_code >= 400:
                logger.error(f"Cohere API error {response.status_code}: {response.text[:300]}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None

            data = response.json()
            content = data.get("message", {}).get("content", [])
            text = ""
            if content and isinstance(content, list):
                text = content[0].get("text", "")
            if not text:
                text = data.get("text", "")

            return _parse_scores_json(text.strip(), len(jobs))

        except Exception as e:
            logger.error(f"Cohere API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return None


def _call_cohere_prompt(
    prompt: str,
    model: str,
    max_retries: int,
    max_tokens: int,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Call Cohere with an already-built prompt and return raw text."""
    try:
        import requests
    except ImportError:
        logger.error("requests package not installed")
        return None

    url = "https://api.cohere.com/v2/chat"
    for attempt in range(max_retries):
        selected_key = api_key or _get_next_cohere_key()
        if not selected_key:
            logger.error("No Cohere API keys available")
            return None

        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {selected_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Bạn là hệ thống chấm điểm. Chỉ trả về JSON array hợp lệ, không markdown.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": max_tokens,
                },
                timeout=90,
            )

            if response.status_code == 429:
                logger.warning(
                    f"Cohere rate limit hit; switching key immediately "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                continue

            if response.status_code >= 400:
                logger.error(f"Cohere API error {response.status_code}: {response.text[:300]}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None

            data = response.json()
            content = data.get("message", {}).get("content", [])
            if content and isinstance(content, list):
                return content[0].get("text", "").strip()
            return data.get("text", "").strip()
        except Exception as e:
            logger.error(f"Cohere API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return None


def _call_aerolink_batch(
    cv_data: dict,
    jobs: List[dict],
    model: str,
    max_retries: int,
    api_key: Optional[str] = None,
    request_timeout: Optional[int] = None,
) -> Optional[List[dict]]:
    """Call Claude through Aerolink's Anthropic-compatible endpoint."""
    prompt = _build_batch_prompt(cv_data, jobs)
    raw = _call_aerolink_chat(
        prompt,
        model,
        max_retries,
        max_tokens=DEFAULT_AEROLINK_SCORING_MAX_TOKENS,
        api_key=api_key,
        request_timeout=request_timeout,
    )
    return _parse_scores_json(raw.strip(), len(jobs)) if raw else None


def _call_aerolink_prompt(
    prompt: str,
    model: str,
    max_retries: int,
    max_tokens: int,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Call Aerolink with an already-built prompt and return raw text."""
    return _call_aerolink_chat(
        prompt,
        model,
        max_retries,
        max_tokens=max_tokens,
        api_key=api_key,
        request_timeout=90,
    )


def _call_aerolink_chat(
    prompt: str,
    model: str,
    max_retries: int,
    max_tokens: int,
    api_key: Optional[str] = None,
    request_timeout: Optional[int] = None,
) -> Optional[str]:
    """Shared Anthropic-compatible Aerolink call."""
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed - pip install anthropic")
        return None

    for attempt in range(max_retries):
        selected_key = api_key or _get_next_aerolink_key()
        if not selected_key:
            logger.error("No Aerolink API keys available")
            return None

        try:
            client = anthropic.Anthropic(
                api_key=selected_key,
                base_url=DEFAULT_AEROLINK_BASE_URL,
                timeout=request_timeout or DEFAULT_SCORING_REQUEST_TIMEOUT_SECONDS,
            )
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                thinking={"type": "disabled"},
                system="Bạn là hệ thống phân tích CV-JD. Chỉ trả về JSON hợp lệ, không markdown.",
                messages=[{"role": "user", "content": prompt}],
            )
            raw = ""
            for block in (message.content or []):
                if hasattr(block, "text") and getattr(block, "type", "") != "thinking":
                    raw += block.text
            return raw.strip()
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str:
                logger.warning(
                    "Aerolink rate limit hit; switching key immediately "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                continue
            logger.error(f"Aerolink API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return None


def _parse_scores_json(text: str, expected_count: int) -> Optional[List[dict]]:
    """Parse LLM response thành list of score dicts."""
    # Xóa markdown backticks
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    # Tìm JSON array trong text
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        logger.error(f"No JSON array found in LLM response: {text[:200]}")
        return None

    try:
        scores = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}\nText: {text[:300]}")
        return None

    if not isinstance(scores, list):
        logger.error(f"Expected list, got {type(scores)}")
        return None

    # Pad nếu thiếu
    while len(scores) < expected_count:
        scores.append({"_fallback": True})

    return scores


def _coerce_score(value) -> int:
    """Convert an LLM score value to int 0-10, tolerating common bad outputs."""
    if isinstance(value, bool):
        return 10 if value else 0
    if isinstance(value, (int, float)):
        return max(0, min(10, int(round(value))))

    text = str(value or "").strip().lower()
    if not text:
        return 5

    # Prefer the first explicit number if present.
    import re
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        return max(0, min(10, int(round(float(match.group())))))

    qualitative = {
        "none": 0,
        "very low": 2,
        "low": 3,
        "weak": 3,
        "partial": 5,
        "medium": 5,
        "moderate": 5,
        "good": 7,
        "strong": 8,
        "high": 8,
        "excellent": 9,
        "perfect": 10,
    }
    for key, score in qualitative.items():
        if key in text:
            return score
    return 5


def _as_list(value) -> List[str]:
    """Normalize model output to a clean list of strings."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value).strip()]


def _fold_text(value) -> str:
    """Lowercase ASCII-ish text for robust Vietnamese keyword checks."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", text).strip().lower()


def _parse_years(value) -> Optional[float]:
    """Parse common CV/JD experience values to years."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = _fold_text(value)
    if not text:
        return None
    code_map = {
        "no_requirement": 0.0,
        "under_1": 0.5,
        "1": 1.0,
        "2": 2.0,
        "3": 3.0,
        "4": 4.0,
        "5": 5.0,
        "over_5": 6.0,
        "all": 0.0,
    }
    if text in code_map:
        return code_map[text]
    if any(kw in text for kw in ["khong yeu cau", "chua co", "fresher"]):
        return 0.0
    if "duoi 1" in text or "under 1" in text:
        return 0.5
    if "tren 5" in text or "hon 5" in text or "5+" in text:
        return 6.0

    numbers = [float(x) for x in re.findall(r"\d+(?:[.,]\d+)?", text.replace(",", "."))]
    if not numbers:
        return None
    return max(numbers)


def _education_level(value) -> Optional[int]:
    """Map education text/code to ordered level."""
    text = _fold_text(value)
    if not text:
        return None
    if any(kw in text for kw in ["sau dai hoc", "thac si", "tien si", "master", "phd"]):
        return 5
    if any(kw in text for kw in ["dai hoc", "cu nhan", "ky su", "bachelor", "dai_hoc"]):
        return 4
    if any(kw in text for kw in ["cao dang", "college", "cao_dang"]):
        return 3
    if any(kw in text for kw in ["trung cap", "trung_cap"]):
        return 2
    if any(kw in text for kw in ["thpt", "trung hoc", "trung_hoc", "pho thong"]):
        return 1
    if any(kw in text for kw in ["khong yeu cau", "khong ro", "all"]):
        return 0
    return None


def _match_factor(value, *, not_required: float = 1.0) -> float:
    """Convert LLM qualitative match labels to a continuous 0-1 factor."""
    text = _fold_text(value)
    if "not_required" in text or "not required" in text or "khong yeu cau" in text:
        return not_required
    if "strong" in text or "tot" in text or "dat" in text:
        return 1.0
    if "partial" in text or "mot phan" in text or "gan" in text:
        return 0.65
    if "weak" in text or "yeu" in text:
        return 0.35
    if "none" in text or "khong" in text:
        return 0.0
    return 0.5


def _score_skills_from_evidence(evidence: dict) -> tuple[float, dict]:
    required = _as_list(evidence.get("required_core_skills"))
    exact = _as_list(evidence.get("matched_exact"))
    related = _as_list(evidence.get("matched_synonym_or_esco"))
    transferable = _as_list(evidence.get("transferable"))
    missing = _as_list(evidence.get("missing_core"))

    denominator = len(required) or (len(exact) + len(related) + len(transferable) + len(missing))
    if denominator <= 0:
        return 5.0, {"reason": "no_core_skill_evidence"}

    direct_weighted = len(exact) * 1.0 + len(related) * 0.75
    transferable_credit = min(len(transferable) * 0.35, denominator * 0.20)
    weighted = direct_weighted + transferable_credit
    score = max(0.0, min(10.0, 10.0 * weighted / denominator))
    if not exact and not related:
        score = min(score, 4.0)
    elif (len(exact) + len(related)) < max(1, denominator / 2):
        score = min(score, 6.0)
    return round(score, 1), {
        "formula": "10 * (exact*1.0 + related*0.75 + capped_transferable) / required_core",
        "required_core": denominator,
        "exact": len(exact),
        "related": len(related),
        "transferable": len(transferable),
        "transferable_credit": round(transferable_credit, 2),
        "missing": len(missing),
    }


def _score_experience_from_evidence(cv_data: dict, job: dict, evidence: dict) -> tuple[float, dict]:
    exp_ev = evidence.get("experience_evidence") if isinstance(evidence.get("experience_evidence"), dict) else {}
    cv_years = (
        _parse_years(exp_ev.get("cv_total_years"))
        or _parse_years(cv_data.get("experience"))
        or 0.0
    )
    req_years = (
        _parse_years(exp_ev.get("jd_required_years"))
        or _parse_years(job.get("experience"))
        or _parse_years(job.get("job_experience"))
        or 0.0
    )

    role_factor = _match_factor(exp_ev.get("role_match"))
    domain_factor = _match_factor(exp_ev.get("domain_match"), not_required=1.0)

    if req_years <= 0:
        year_factor = 1.0 if cv_years <= 1.0 else max(0.35, 1.0 - max(0.0, cv_years - 1.0) * 0.12)
    else:
        year_factor = min(1.0, cv_years / req_years)
        if cv_years > req_years + 3:
            year_factor *= 0.85

    score = 10.0 * (0.45 * year_factor + 0.40 * role_factor + 0.15 * domain_factor)
    return round(max(0.0, min(10.0, score)), 1), {
        "formula": "10 * (0.45*year + 0.40*role + 0.15*domain)",
        "cv_years": cv_years,
        "required_years": req_years,
        "year_factor": round(year_factor, 2),
        "role_factor": role_factor,
        "domain_factor": domain_factor,
    }


def _score_education_from_evidence(cv_data: dict, job: dict, evidence: dict) -> tuple[float, dict]:
    cv_level = _education_level(cv_data.get("education"))
    req_level = (
        _education_level(job.get("education_level"))
        or _education_level(job.get("job_education"))
        or _education_level(job.get("education"))
    )
    gap_text = _fold_text(evidence.get("education_gap"))

    if not req_level:
        score = 7.0
    elif cv_level is None:
        score = 5.0
    elif cv_level >= req_level:
        score = 9.0 if cv_level == req_level else 10.0
    else:
        score = max(2.0, 10.0 - 3.0 * (req_level - cv_level))

    if any(kw in gap_text for kw in ["thieu", "khong dat", "chua dat"]):
        score = min(score, 6.0)
    if any(kw in gap_text for kw in ["dat", "vuot"]):
        score = max(score, 8.0)

    return round(score, 1), {
        "formula": "10 if cv>=required else max(2, 10 - 3*gap)",
        "cv_level": cv_level,
        "required_level": req_level,
    }


def _score_salary_from_data(cv_data: dict, job: dict) -> tuple[float, dict]:
    expected_salary = cv_data.get("salary") or cv_data.get("expected_salary")
    job_salary = job.get("job_salary") or job.get("salary") or ""
    try:
        from job_matching.scoring.salary_normalizer import SalaryNormalizer
        score = SalaryNormalizer().compare_salary(expected_salary, job_salary) / 10.0
    except Exception as exc:
        logger.warning("Salary formula failed: %s", exc)
        score = 5.0
    return round(max(0.0, min(10.0, score)), 1), {
        "formula": "SalaryNormalizer.compare_salary / 10",
        "expected_salary": expected_salary,
        "job_salary": job_salary,
    }


def _compute_detail_scores(cv_data: dict, job: dict, evidence: dict) -> tuple[dict, dict]:
    """Compute detail scores from LLM evidence instead of trusting LLM numbers."""
    skills_score, skills_basis = _score_skills_from_evidence(evidence)
    experience_score, experience_basis = _score_experience_from_evidence(cv_data, job, evidence)
    education_score, education_basis = _score_education_from_evidence(cv_data, job, evidence)
    salary_score, salary_basis = _score_salary_from_data(cv_data, job)

    exp_ev = evidence.get("experience_evidence") if isinstance(evidence.get("experience_evidence"), dict) else {}
    role_factor = _match_factor(exp_ev.get("role_match"))
    domain_factor = _match_factor(exp_ev.get("domain_match"), not_required=1.0)
    relevance_score = round(
        max(0.0, min(10.0, 10.0 * (
            0.35 * role_factor +
            0.25 * domain_factor +
            0.40 * (skills_score / 10.0)
        ))),
        1,
    )

    scores = {
        "relevance": relevance_score,
        "skills": skills_score,
        "experience": experience_score,
        "education": education_score,
        "location": 5,
        "salary": salary_score,
    }
    basis = {
        "method": "formula_from_llm_evidence",
        "relevance": {
            "formula": "10 * (0.35*role + 0.25*domain + 0.40*skills)",
            "role_factor": role_factor,
            "domain_factor": domain_factor,
            "skills_factor": round(skills_score / 10.0, 2),
        },
        "skills": skills_basis,
        "experience": experience_basis,
        "education": education_basis,
        "salary": salary_basis,
    }
    return scores, basis


def _empty_score(index: int) -> dict:
    """Trả về score mặc định khi LLM fail."""
    return {
        "job_index": index,
        "scores": {dim: 5 for dim in SCORING_DIMENSIONS},
        "total": 5.0,
        "comment": "",
        "llm_time": 0,
        "fallback": True,
    }
