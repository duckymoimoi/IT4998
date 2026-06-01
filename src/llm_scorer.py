"""
LLM Scorer - Đánh giá đa chiều CV-Job bằng LLM (Groq API).

Pipeline production:
  Hybrid Search (BM25 + kNN + ESCO) → top-20 → LLM 6-dim scoring → Final ranking

Batch mode: gửi 1 CV + N jobs trong 1 request → LLM so sánh listwise
→ ranking tốt hơn scoring từng cặp riêng lẻ.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Groq API keys (rotation để tránh rate limit)
_API_KEYS = []
_current_key_idx = 0

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


def _format_job_for_prompt(job: dict, index: int) -> str:
    """Format 1 job thành text ngắn gọn cho LLM prompt."""
    title = job.get("title", "N/A")
    company = job.get("company_name", "N/A")
    salary = job.get("job_salary", "Thỏa thuận")
    location = job.get("job_location", "N/A")
    experience = job.get("job_experience", "N/A")
    education_req = job.get("job_education", "")
    requirements = str(job.get("job_requirements", ""))[:400]
    tech_skills = job.get("technical_skills", "")
    req_tags = job.get("requirements_tags", "")

    return f"""[JOB {index + 1}] {title}
Công ty: {company}
Lương: {salary} | Địa điểm: {location} | Kinh nghiệm: {experience}
Yêu cầu học vấn: {education_req}
Kỹ năng: {req_tags or tech_skills}
Mô tả yêu cầu: {requirements}"""


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
    result += f"\nKinh nghiệm: {exp_text}"
    result += f"\nHọc vấn: {edu_text}"
    result += f"\nĐịa điểm mong muốn: {location}"
    result += f"\nMức lương mong muốn: {salary} triệu VND"

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
Ứng viên có NÊN được mời phỏng vấn không? Skills_match có trọng số cao nhất.
| 0-2: Hoàn toàn không liên quan | 3-4: Liên quan xa | 5-6: Tiềm năng, thiếu nhiều | 7-8: Phù hợp tốt | 9-10: Lý tưởng |

### 2. skills — Khớp kỹ năng (QUAN TRỌNG NHẤT)
So sánh kỹ năng CV vs yêu cầu JD. Exact match = 1 điểm, synonym = 1, partial = 0.5.
| 0-2: <20% khớp | 3-4: 20-40% | 5-6: 40-60%, thiếu skill core | 7-8: 60-80% | 9-10: >80% |

### 3. experience — Kinh nghiệm (CẢ THIẾU LẪN THỪA ĐỀU PHẠT)
| 0-2: Chênh rất lớn | 3-4: Thiếu/thừa đáng kể | 5-6: Thiếu/thừa nhẹ | 7-8: ±1 năm | 9-10: Đúng |
⚠️ Job "Không yêu cầu KN": Fresher=10, 1-2 năm=8-9, 3-5 năm=5-6, 5+ năm=3-4
⚠️ Over-qualification: Job cần 1-2 năm, CV 5+ năm → MAX 5 điểm

### 4. education — Học vấn
| 0-3: Không đạt | 4-6: Thiếu 1 bậc | 7-8: Đạt yêu cầu | 9-10: Đúng/vượt | Job không nêu rõ → 7 |

### 5. salary — Lương
| 0-3: Job thấp hơn kỳ vọng rất nhiều | 4-5: Thấp hơn | 5-6: Thỏa thuận/không rõ | 7-8: Trong kỳ vọng | 9-10: Cao hơn kỳ vọng |

⚠️ KHÔNG chấm location (tính riêng bằng Goong Maps GPS).

## NHẬN XÉT (comment)
Viết 1-2 câu tiếng Việt (25-50 từ). PHẢI nêu CỤ THỂ:
- Tên kỹ năng KHỚP và THIẾU (VD: "có Python, React nhưng thiếu Java/Spring")
- KN đủ/thiếu/thừa bao nhiêu so với yêu cầu
- Điểm mạnh và rủi ro chính
KHÔNG viết chung chung "phù hợp", "khớp". Phải phân tích cụ thể.

## OUTPUT — CHỈ JSON array, không giải thích
[
  {{"job": 1, "relevance": 8, "skills": 9, "experience": 7, "education": 9, "salary": 7, "comment": "Có Python, React, Docker đúng yêu cầu Fullstack. KN 2 năm gần đạt yêu cầu 3 năm. Cần bổ sung CI/CD."}},
  {{"job": 2, "relevance": 3, "skills": 2, "experience": 5, "education": 8, "salary": 6, "comment": "Yêu cầu Java/Spring Boot nhưng CV chỉ có Python/Django — thiếu core tech stack. KN đạt nhưng cần chuyển đổi ngôn ngữ."}}
]"""


def score_batch(
    cv_data: dict,
    jobs: List[dict],
    weights: Optional[Dict[str, float]] = None,
    model: str = "llama-3.3-70b-versatile",
    max_retries: int = 3,
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

    # Gọi LLM (có timing)
    t0 = time.time()
    raw_scores = _call_groq_batch(cv_data, jobs, model, max_retries)
    llm_elapsed = time.time() - t0
    logger.info(f"LLM scoring: {len(jobs)} jobs in {llm_elapsed:.1f}s")

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


def _call_groq_batch(
    cv_data: dict,
    jobs: List[dict],
    model: str,
    max_retries: int,
) -> Optional[List[dict]]:
    """Gọi Groq API với retry + key rotation."""
    try:
        from groq import Groq
    except ImportError:
        logger.error("groq package not installed — pip install groq")
        return None

    prompt = _build_batch_prompt(cv_data, jobs)

    for attempt in range(max_retries):
        api_key = _get_next_key()
        if not api_key:
            logger.error("No Groq API keys available")
            return None

        try:
            client = Groq(api_key=api_key)
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
                max_tokens=4000,
            )

            text = response.choices[0].message.content.strip()
            return _parse_scores_json(text, len(jobs))

        except Exception as e:
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str:
                wait = 5 * (attempt + 1)
                logger.warning(f"Rate limit hit (key #{_current_key_idx}), "
                             f"waiting {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                logger.error(f"Groq API error: {e}")
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
    }
