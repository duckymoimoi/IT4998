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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
DEFAULT_SCORING_BATCH_WORKERS = _env_int("SCORING_BATCH_WORKERS", 4)

# Groq API keys (rotation để tránh rate limit)
_API_KEYS = []
_current_key_idx = 0
_COHERE_API_KEYS = []
_current_cohere_key_idx = 0

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
    _load_api_keys()
    if not _API_KEYS:
        return None
    key = _API_KEYS[_current_key_idx % len(_API_KEYS)]
    _current_key_idx += 1
    return key


def _get_groq_key_pool() -> List[str]:
    """Return configured Groq keys in stable order."""
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
    _load_cohere_api_keys()
    if not _COHERE_API_KEYS:
        return None
    key = _COHERE_API_KEYS[_current_cohere_key_idx % len(_COHERE_API_KEYS)]
    _current_cohere_key_idx += 1
    return key


def _get_cohere_key_pool() -> List[str]:
    """Return configured Cohere keys in stable order."""
    _load_cohere_api_keys()
    return list(_COHERE_API_KEYS)


def _format_job_for_prompt(job: dict, index: int) -> str:
    """Format 1 job thành text ngắn gọn cho LLM prompt."""
    title = job.get("title", "N/A")
    company = job.get("company") or job.get("company_name", "N/A")
    category = job.get("category", "")
    company_field = job.get("company_field", "")
    salary = job.get("job_salary", "Thỏa thuận")
    location = job.get("job_location", "N/A")
    experience = job.get("experience") or job.get("job_experience", "N/A")
    education_req = job.get("education_level") or job.get("job_education", "")
    education_field = job.get("education_field", "")
    requirements = str(job.get("job_requirements", ""))[:800]
    description = str(job.get("job_description", ""))[:350]
    tech_skills = job.get("technical_skills", "")
    soft_skills = job.get("soft_skills", "")
    specializations = job.get("specializations", "")
    req_tags = job.get("requirements_tags", "")

    return f"""[JOB {index + 1}] {title}
Công ty: {company}
Ngành/domain: {category} | Lĩnh vực công ty: {company_field}
Nhóm chuyên môn: {specializations}
Lương: {salary} | Địa điểm: {location} | Kinh nghiệm yêu cầu: {experience}
Yêu cầu học vấn: {education_req} {education_field}
Kỹ năng/yêu cầu chính: {req_tags or tech_skills}
Kỹ năng mềm: {soft_skills}
Mô tả công việc: {description}
Yêu cầu chi tiết: {requirements}"""


def _format_cv_for_prompt(cv_data: dict) -> str:
    """Format CV data thành text cho LLM prompt."""
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

    soft_skills = cv_data.get("soft_skills", "")
    languages = cv_data.get("languages", "")
    certificates = cv_data.get("certificates", "")

    result = f"Kỹ năng chuyên môn: {skills}"
    if soft_skills:
        result += f"\nKỹ năng mềm: {soft_skills}"
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
    """Tạo prompt batch dựa trên rubric annotation đã validate (Kappa 0.70-0.76)."""
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
⚠️ Nếu CV chỉ có tổng số năm nhưng không chứng minh đúng vai trò/domain, relevance KHÔNG được vượt 6-7.
VD: CV có Java nhưng thiên AI/frontend, JD là Backend Java Bank → phải nêu thiếu bằng chứng backend/banking, không được chấm như backend Java đủ kinh nghiệm.

### 2. skills — Khớp kỹ năng (QUAN TRỌNG NHẤT)
So sánh kỹ năng CV vs yêu cầu JD theo kỹ năng lõi của VAI TRÒ.
| 0-2: <20% khớp | 3-4: 20-40% | 5-6: 40-60%, thiếu skill core | 7-8: 60-80% | 9-10: >80% |
⚠️ Không chỉ đếm keyword. Phân biệt:
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
⚠️ Job "Không yêu cầu KN": Fresher=10, 1-2 năm=8-9, 3-5 năm=5-6, 5+ năm=3-4
⚠️ Over-qualification: Job cần 1-2 năm, CV 5+ năm → MAX 5 điểm
⚠️ Nếu CV có 3 năm tổng nhưng không thể hiện là 3 năm backend Java/banking thì KHÔNG viết "KN 3 năm đạt yêu cầu"; phải viết "tổng KN 3 năm, nhưng chưa thấy kinh nghiệm backend Java/banking".
⚠️ Nếu JD yêu cầu Bank/Fintech/Onsite Bank, kinh nghiệm ngân hàng là lợi thế riêng; thiếu domain này phải nêu là rủi ro, không tự suy diễn từ số năm.

### 4. education — Học vấn
| 0-3: Không đạt | 4-6: Thiếu 1 bậc | 7-8: Đạt yêu cầu | 9-10: Đúng/vượt | Job không nêu rõ → 7 |

### 5. salary — Lương
| 0-3: Job thấp hơn kỳ vọng rất nhiều | 4-5: Thấp hơn | 5-6: Thỏa thuận/không rõ | 7-8: Trong kỳ vọng | 9-10: Cao hơn kỳ vọng |

⚠️ KHÔNG chấm location (tính riêng bằng Goong Maps GPS).

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
    provider = DEFAULT_SCORING_PROVIDER
    if provider == "cohere":
        _load_cohere_api_keys()
        model = model or DEFAULT_COHERE_SCORING_MODEL
        if max_retries is None:
            max_retries = max(3, len(_COHERE_API_KEYS) or 1)
    else:
        provider = "groq"
        _load_api_keys()
        model = model or DEFAULT_SCORING_MODEL
        if max_retries is None:
            # Try at least once per configured key, so several free-tier keys can
            # absorb short rate-limit bursts during interactive demos.
            max_retries = max(3, len(_API_KEYS) or 1)

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
        if i < len(raw_scores):
            s = raw_scores[i]
        else:
            s = {}

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


def _call_parallel_batches(
    provider: str,
    cv_data: dict,
    jobs: List[dict],
    model: str,
    key_pool: List[str],
    max_retries: int,
) -> Optional[List[dict]]:
    """Score chunks in parallel, one configured API key per chunk."""
    if not key_pool or len(key_pool) <= 1 or len(jobs) <= 1:
        if provider == "cohere":
            return _call_cohere_batch(cv_data, jobs, model, max_retries)
        return _call_groq_batch(cv_data, jobs, model, max_retries)

    worker_count = min(DEFAULT_SCORING_BATCH_WORKERS, len(key_pool), len(jobs))
    chunks = _split_evenly(jobs, worker_count)
    raw_scores: List[dict] = [{} for _ in jobs]
    failed_chunks: List[tuple[int, List[dict]]] = []

    def call_chunk(chunk_offset: int, chunk_jobs: List[dict], api_key: str):
        if provider == "cohere":
            scores = _call_cohere_batch(
                cv_data, chunk_jobs, model, max_retries=1, api_key=api_key
            )
        else:
            scores = _call_groq_batch(
                cv_data, chunk_jobs, model, max_retries=1, api_key=api_key
            )
        return chunk_offset, chunk_jobs, scores

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for chunk_idx, (offset, chunk_jobs) in enumerate(chunks):
            futures.append(
                executor.submit(call_chunk, offset, chunk_jobs, key_pool[chunk_idx])
            )

        for future in as_completed(futures):
            try:
                offset, chunk_jobs, scores = future.result()
            except Exception as e:
                logger.error(f"Parallel scoring chunk failed: {e}")
                continue
            if not scores:
                failed_chunks.append((offset, chunk_jobs))
                continue
            for local_idx, score in enumerate(scores[:len(chunk_jobs)]):
                raw_scores[offset + local_idx] = score

    # Retry failed chunks with normal round-robin rotation. This avoids waiting
    # on one exhausted key when other independent keys are still usable.
    for offset, chunk_jobs in failed_chunks:
        if provider == "cohere":
            retry_scores = _call_cohere_batch(cv_data, chunk_jobs, model, max_retries)
        else:
            retry_scores = _call_groq_batch(cv_data, chunk_jobs, model, max_retries)
        if retry_scores:
            for local_idx, score in enumerate(retry_scores[:len(chunk_jobs)]):
                raw_scores[offset + local_idx] = score

    if all(not score for score in raw_scores):
        return None
    return raw_scores


def _call_groq_batch(
    cv_data: dict,
    jobs: List[dict],
    model: str,
    max_retries: int,
    api_key: Optional[str] = None,
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
            client = Groq(api_key=selected_key)
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


def _call_cohere_batch(
    cv_data: dict,
    jobs: List[dict],
    model: str,
    max_retries: int,
    api_key: Optional[str] = None,
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
        scores.append({})

    return scores


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
