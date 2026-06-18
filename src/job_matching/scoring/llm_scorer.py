"""
LLM Scorer - Đánh giá đa chiều CV-Job bằng LLM (Groq API).

Pipeline production:
  Hybrid Search (BM25 + kNN + ESCO) → top-N → LLM 6-dim scoring → Final ranking

Batch mode: gửi 1 CV + N jobs trong 1 request → LLM so sánh listwise
→ ranking tốt hơn scoring từng cặp riêng lẻ.
"""

import os
import json
import logging
import queue
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


DEFAULT_SCORING_MODEL = os.environ.get("GROQ_SCORING_MODEL", "openai/gpt-oss-120b")
DEFAULT_SCORING_MAX_COMPLETION_TOKENS = _env_int(
    "GROQ_SCORING_MAX_COMPLETION_TOKENS",
    3000,
)
DEFAULT_SCORING_PROVIDER = os.environ.get("SCORING_PROVIDER", "groq").strip().lower()
DEFAULT_COHERE_SCORING_MODEL = os.environ.get("COHERE_SCORING_MODEL", "command-r-08-2024")
DEFAULT_COHERE_SCORING_MAX_TOKENS = _env_int("COHERE_SCORING_MAX_TOKENS", 2500)
DEFAULT_DETAIL_MAX_COMPLETION_TOKENS = _env_int("GROQ_DETAIL_MAX_COMPLETION_TOKENS", 5000)
DEFAULT_COHERE_DETAIL_MAX_TOKENS = _env_int("COHERE_DETAIL_MAX_TOKENS", 4000)
DEFAULT_SCORING_BATCH_WORKERS = _env_int("SCORING_BATCH_WORKERS", 6)
DEFAULT_SCORING_BATCH_SIZE = _env_int("SCORING_BATCH_SIZE", 5)
DEFAULT_SCORING_TIME_LIMIT_SECONDS = _env_int("SCORING_TIME_LIMIT_SECONDS", 60)
DEFAULT_SCORING_REQUEST_TIMEOUT_SECONDS = _env_int(
    "SCORING_REQUEST_TIMEOUT_SECONDS",
    45,
)

# Groq API keys (rotation để tránh rate limit)
_API_KEYS = []
_current_key_idx = 0
_COHERE_API_KEYS = []
_current_cohere_key_idx = 0
_GROQ_KEY_LOCK = threading.RLock()
_COHERE_KEY_LOCK = threading.RLock()

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


def _resolve_provider_config(
    model: Optional[str] = None,
    max_retries: Optional[int] = None,
) -> tuple[str, str, int]:
    """Resolve the explicitly configured scoring provider and its limits."""
    provider = DEFAULT_SCORING_PROVIDER
    if provider == "cohere":
        _load_cohere_api_keys()
        return (
            provider,
            model or DEFAULT_COHERE_SCORING_MODEL,
            max_retries if max_retries is not None else max(3, len(_COHERE_API_KEYS) or 1),
        )

    _load_api_keys()
    return (
        "groq",
        model or DEFAULT_SCORING_MODEL,
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


def _format_job_for_prompt(job: dict, index: int) -> str:
    """Format 1 job thành text ngắn gọn cho LLM prompt."""
    title = job.get("title", "N/A")
    company = job.get("company") or job.get("company_name", "N/A")
    salary = job.get("job_salary", "Thỏa thuận")
    location = job.get("job_location", "N/A")
    experience = job.get("experience") or job.get("job_experience", "N/A")
    education_req = job.get("education_level") or job.get("job_education", "")
    education_field = job.get("education_field", "")
    requirements = _truncate_prompt_text(job.get("job_requirements", ""), 1200)
    description = _truncate_prompt_text(job.get("job_description", ""), 200)
    tech_skills = job.get("technical_skills", "")
    specializations = job.get("specializations", "")
    req_tags = job.get("requirements_tags", "")

    return f"""[JOB {index + 1}] {title}
Công ty: {company}
Nhóm chuyên môn: {specializations}
Lương: {salary} | Địa điểm: {location} | Kinh nghiệm yêu cầu: {experience}
Yêu cầu học vấn: {education_req} {education_field}
Kỹ năng/yêu cầu chính: {req_tags or tech_skills}
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
Viết 2-3 câu tiếng Việt (35-70 từ), có tính hướng dẫn hành động. PHẢI có đủ:
- Khớp gì: kỹ năng/role/domain nào đang khớp.
- Thiếu/rủi ro gì: kỹ năng lõi, vai trò, domain, số năm đúng vai trò.
- Nên cải thiện/ứng tuyển thế nào: bổ sung skill/domain nào hoặc phù hợp hơn với loại job nào.
KHÔNG viết chung chung "phù hợp", "khớp", "KN đạt yêu cầu" nếu không nêu rõ kinh nghiệm thuộc vai trò/domain nào.

## OUTPUT — CHỈ JSON array, không giải thích
[
  {{"job": 1, "relevance": 8, "skills": 8, "experience": 7, "education": 9, "salary": 7, "comment": "Khớp Python, React và Docker với vị trí Fullstack; tổng KN 2 năm gần yêu cầu 3 năm và đúng hướng phát triển web. Nên bổ sung CI/CD và dự án backend API rõ hơn để tăng độ tin cậy."}},
  {{"job": 2, "relevance": 5, "skills": 5, "experience": 5, "education": 8, "salary": 6, "comment": "Có Java nhưng chưa thấy Spring Boot, backend API, SQL hay kinh nghiệm domain ngân hàng; tổng KN 3 năm không đủ chứng minh là 3 năm Backend Java Bank. Nên bổ sung dự án Java backend/Spring và nghiệp vụ banking/fintech trước khi ưu tiên job này."}}
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
- Các trường `relevance`, `skills`, `experience`, `education`, `salary` BẮT BUỘC là số nguyên 0-10. Không dùng chữ như "medium", "good", "high".

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
    "relevance": 0,
    "skills": 0,
    "experience": 0,
    "education": 0,
    "salary": 0,
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
            scores[dim] = max(0, min(10, int(val)))

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
    provider, model, max_retries = _resolve_provider_config(model, max_retries)

    prompt = _build_detail_evidence_prompt(cv_data, job)
    t0 = time.time()
    if provider == "cohere":
        raw = _call_cohere_prompt(
            prompt, model, max_retries,
            max_tokens=DEFAULT_COHERE_DETAIL_MAX_TOKENS,
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
    scores = {}
    for dim in ["relevance", "skills", "experience", "education", "salary"]:
        scores[dim] = _coerce_score(item.get(dim, 5))
    scores["location"] = 5

    total = sum(scores[dim] * w.get(dim, 0) for dim in SCORING_DIMENSIONS)
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

    return {
        "job_index": 0,
        "scores": scores,
        "total": round(total, 2),
        "comment": str(item.get("comment", "") or ""),
        "evidence": evidence,
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
    if len(key_pool) <= 1 or len(jobs) <= 1:
        if provider == "cohere":
            return _call_cohere_batch(cv_data, jobs, model, max_retries)
        return _call_groq_batch(cv_data, jobs, model, max_retries)

    batch_size = max(1, DEFAULT_SCORING_BATCH_SIZE)
    chunks = [
        (offset, jobs[offset:offset + batch_size])
        for offset in range(0, len(jobs), batch_size)
    ]
    worker_count = min(DEFAULT_SCORING_BATCH_WORKERS, len(key_pool), len(chunks))
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
            client = Groq(api_key=selected_key)
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
