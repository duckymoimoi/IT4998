"""
Job Matching Web Application — Production Pipeline.

Pipeline:
  CV Input → ESCO Expansion → Hybrid Search (BM25 + kNN + RRF)
  → Top-20 → LLM 6-dim Scoring → WSM Final Ranking

Tách biệt rõ ràng: retrieval (ES) → scoring (LLM) → presentation (Flask).
"""

import os
import math
import re
import tempfile
import logging

from dotenv import load_dotenv
load_dotenv()  # Load .env (GROQ_API_KEY_*, GOONG_API_KEY)

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from elastic_helper import ElasticHelper
from vietnam_cities_data import get_city_info

logger = logging.getLogger(__name__)

# ============================================================
# Flask App
# ============================================================

app = Flask(__name__, template_folder=".", static_folder=".", static_url_path="")

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "bmp", "tiff"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

CATEGORIES = [
    "Nhân viên kinh doanh", "Kế toán", "Marketing",
    "Hành chính nhân sự", "Chăm sóc khách hàng", "Ngân hàng",
    "IT", "Kỹ sư xây dựng", "Thiết kế đồ họa",
    "Bất động sản", "Giáo dục", "Telesales", "Lao động phổ thông",
]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# Service Initialization (lazy)
# ============================================================

es_helper = ElasticHelper()

# Embedding + ESCO — lazy load
_embedding_service = None
_esco_expander = None
_skill_graph = None
_use_hybrid = False


def _env_bool(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _init_services():
    """Initialize embedding + ESCO + SkillGraph services (chỉ gọi 1 lần)."""
    global _embedding_service, _esco_expander, _skill_graph, _use_hybrid

    if _embedding_service is not None:
        return

    # Skill Knowledge Graph is an opt-in enrichment. Experiments selected
    # ESCO-only hybrid search as the stable production default.
    if _env_bool("ENABLE_SKILL_GRAPH", default=False):
        try:
            from skill_graph import get_skill_graph
            _skill_graph = get_skill_graph()
            logger.info("Skill Knowledge Graph enabled")
        except Exception as e:
            logger.warning(f"SkillGraph not available: {e}")
    else:
        logger.info("Skill Knowledge Graph disabled (set ENABLE_SKILL_GRAPH=1 to enable)")

    try:
        if es_helper.has_embedding_field():
            from embedding_service import get_embedding_service
            _embedding_service = get_embedding_service()
            _use_hybrid = True
            logger.info("Hybrid search (BM25 + kNN) enabled")

            # ESCO expander
            try:
                from esco_expander import get_esco_expander
                _esco_expander = get_esco_expander(embedding_service=_embedding_service)
                logger.info("ESCO skill expansion enabled")
            except Exception as e:
                logger.warning(f"ESCO not available: {e}")
        else:
            logger.info("No embeddings — BM25-only mode")
    except Exception as e:
        logger.warning(f"Cannot init embedding service: {e}")
        logger.info("Fallback to BM25-only search")


# ============================================================
# Location Scoring (city default, detailed distance optional)
# ============================================================

import requests as http_requests

_goong_api_key = os.environ.get("GOONG_API_KEY", "")
_geocode_cache = {}
LOCATION_SCORE_MODE = os.environ.get("LOCATION_SCORE_MODE", "city").strip().lower()
ENABLE_CITY_PRIORITY = _env_bool("ENABLE_CITY_PRIORITY", default=True)


def _goong_geocode(address):
    """Geocode address → (lat, lon) via Goong API. Cached."""
    if not _goong_api_key or not address:
        return None
    if address in _geocode_cache:
        return _geocode_cache[address]
    try:
        resp = http_requests.get(
            "https://rsapi.goong.io/Geocode",
            params={"address": address, "api_key": _goong_api_key},
            timeout=5,
        )
        data = resp.json()
        results = data.get("results", [])
        if results:
            loc = results[0]["geometry"]["location"]
            coords = (loc["lat"], loc["lng"])
            _geocode_cache[address] = coords
            return coords
    except Exception as e:
        logger.warning(f"Goong geocode error: {e}")
    _geocode_cache[address] = None
    return None


def _goong_distance_km(origin, destination):
    """Tính khoảng cách lái xe (km) giữa 2 tọa độ via Goong Distance Matrix."""
    if not _goong_api_key:
        return None
    try:
        resp = http_requests.get(
            "https://rsapi.goong.io/DistanceMatrix",
            params={
                "origins": f"{origin[0]},{origin[1]}",
                "destinations": f"{destination[0]},{destination[1]}",
                "vehicle": "car",
                "api_key": _goong_api_key,
            },
            timeout=5,
        )
        data = resp.json()
        rows = data.get("rows", [])
        if rows and rows[0].get("elements"):
            elem = rows[0]["elements"][0]
            if elem.get("status") == "OK":
                meters = elem["distance"]["value"]
                return meters / 1000.0
    except Exception as e:
        logger.warning(f"Goong distance error: {e}")
    return None


def _extract_city_names(location_text):
    """Return normalized Vietnamese city/province names found in free text."""
    if not location_text:
        return set()
    try:
        return {city["name"] for city in get_city_info(str(location_text))}
    except Exception:
        return set()


def _is_same_city(cv_location, job_location):
    cv_cities = _extract_city_names(cv_location)
    job_cities = _extract_city_names(job_location)
    return bool(cv_cities and job_cities and cv_cities.intersection(job_cities))


def calculate_location_score(cv_address, job, detailed=None):
    """Location score.

    Default LOCATION_SCORE_MODE=city uses city/province matching only and does
    not call the Distance Matrix API. Set LOCATION_SCORE_MODE=detailed or pass
    detailed=True to calculate kilometer distance.
    """
    if detailed is None:
        detailed = LOCATION_SCORE_MODE == "detailed"

    LAMBDA = 30
    FLOOR = 1.5
    job_location = job.get("job_location", "") if isinstance(job, dict) else str(job)

    if not cv_address:
        return 5, None

    if not detailed:
        return _city_fallback_score(cv_address, job_location), None

    cv_coords = _goong_geocode(cv_address)
    if not cv_coords:
        return _city_fallback_score(cv_address, job_location), None

    # Collect job coordinates — prefer pre-geocoded from ES
    job_coord_list = []
    if isinstance(job, dict):
        geo = job.get("geo_coordinates", [])
        if geo and isinstance(geo, list):
            for g in geo:
                if g.get("lat") and g.get("lng"):
                    job_coord_list.append((g["lat"], g["lng"]))
        if not job_coord_list:
            company_addr = job.get("company_address", "")
            if company_addr:
                coords = _goong_geocode(company_addr)
                if coords:
                    job_coord_list.append(coords)
    if not job_coord_list and job_location:
        for loc in str(job_location).split(","):
            coords = _goong_geocode(loc.strip())
            if coords:
                job_coord_list.append(coords)

    if not job_coord_list:
        return _city_fallback_score(cv_address, job_location), None

    best_score, best_dist = 0, None
    for job_coords in job_coord_list:
        dist = _goong_distance_km(cv_coords, job_coords)
        if dist is None:
            from geopy.distance import geodesic
            dist = geodesic(cv_coords, job_coords).km
        score = max(FLOOR, 10 * math.exp(-dist / LAMBDA))
        if score > best_score:
            best_score = score
            best_dist = dist

    return round(best_score, 1) if best_score > 0 else 5, best_dist


def _city_fallback_score(cv_location, job_location):
    """Fallback: so sánh tên thành phố/tỉnh.

    Có city match rõ ràng -> 8.5. Có city ở cả hai phía nhưng khác nhau -> 1.5.
    Không đủ thông tin city -> 5.0 để không phạt nhầm dữ liệu thiếu địa điểm.
    """
    cv_cities = _extract_city_names(cv_location)
    job_cities = _extract_city_names(job_location)

    if cv_cities and job_cities:
        return 8.5 if cv_cities.intersection(job_cities) else 1.5
    return 5.0


def _prioritize_same_city(jobs, cv_location):
    """Stable partition: đưa job cùng tỉnh/thành lên trước trong tập retrieved."""
    if not ENABLE_CITY_PRIORITY or not cv_location:
        return jobs

    same_city = []
    others = []
    for job in jobs:
        job_location = job.get("job_location", "") if isinstance(job, dict) else ""
        if _is_same_city(cv_location, job_location):
            job["_location_priority"] = "same_city"
            same_city.append(job)
        else:
            job["_location_priority"] = "other_or_unknown"
            others.append(job)

    return same_city + others


# ============================================================
# Search Pipeline
# ============================================================

def search_pipeline(cv_data, categories=None, top_n=20):
    """
    Full pipeline: ESCO → Hybrid Search → LLM Scoring.
    
    Args:
        cv_data: dict with skills, experience, education, location, salary
        categories: list[str] filter
        top_n: số jobs gửi cho LLM scoring
    
    Returns:
        (jobs_with_scores, search_mode, total_found)
    """
    _init_services()

    profile_text = cv_data.get("skills", "").strip()
    # Gộp thêm soft_skills, languages, certificates vào search text
    extra_parts = []
    for field in ["soft_skills", "languages", "certificates"]:
        val = cv_data.get(field, "").strip()
        if val:
            extra_parts.append(val)
    full_text = profile_text + (", " + ", ".join(extra_parts) if extra_parts else "")
    if not profile_text:
        return [], "error", 0

    cv_gender = cv_data.get("gender", "both")

    # --- Stage 1a: ESCO Expansion (technical skills only) ---
    expanded_tech = profile_text
    if _esco_expander:
        expanded_tech = _esco_expander.expand_skills(profile_text)
        logger.info(f"ESCO expanded: {len(profile_text.split(','))} → "
                    f"{len(expanded_tech.split(','))} terms")

    # --- Stage 1b: Knowledge Graph Expansion (kNN semantic enrichment) ---
    knn_text = expanded_tech
    if extra_parts:
        knn_text += ", " + ", ".join(extra_parts)
    if _skill_graph:
        knn_text = _skill_graph.expand_skills_text(knn_text, max_terms=20)
        logger.info(f"Graph enriched kNN: {len(expanded_tech.split(','))} → "
                    f"{len(knn_text.split(','))} terms")

    # --- Stage 2: Hybrid Retrieval ---
    if _use_hybrid and _embedding_service:
        # kNN: ESCO-enriched text; KG terms are included only when enabled.
        cv_text = _embedding_service.build_cv_text({"skills": knn_text})
        query_vector = _embedding_service.encode_single(cv_text)
        # BM25: flat multi_match (best for current data structure)
        jobs, total = es_helper.search_jobs_hybrid(
            full_text, query_vector, size=50,
            categories=categories or None,
            cv_gender=cv_gender, exclude_expired=True,
        )
        search_mode = "hybrid"
        if _esco_expander:
            search_mode += "+esco"
        if _skill_graph:
            search_mode += "+graph"
    else:
        jobs, total = es_helper.search_jobs_by_profile(
            full_text, size=50,
            categories=categories or None,
            cv_gender=cv_gender, exclude_expired=True,
        )
        search_mode = "bm25"

    if not jobs:
        return [], search_mode, 0

    # Location is a lightweight preference at retrieval time: keep semantic
    # retrieval results, but score same-city jobs first when the user provides
    # a desired city/province.
    jobs = _prioritize_same_city(jobs, cv_data.get("location", ""))

    # --- Stage 3: LLM 6-dim Scoring (top-N) ---
    top_jobs = jobs[:top_n]
    remaining_jobs = jobs[top_n:]

    scored_jobs = _score_with_llm(cv_data, top_jobs)

    # Remaining jobs lấy score mặc định thấp hơn
    for i, job in enumerate(remaining_jobs):
        job["match_score"] = 3.0
        job["score_breakdown"] = {dim: 3 for dim in
            ["relevance", "skills", "experience", "education", "location", "salary"]}
        job["llm_scored"] = False
        job["comment"] = ""
        job["distance_km"] = None

    all_jobs = scored_jobs + remaining_jobs
    all_jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)

    return all_jobs, search_mode, total


def _score_with_llm(cv_data, jobs):
    """Gọi LLM scorer, fallback sang heuristic nếu fail."""
    try:
        from llm_scorer import score_batch, DEFAULT_WEIGHTS

        weights = cv_data.get("weights", DEFAULT_WEIGHTS)
        results = score_batch(cv_data, jobs, weights=weights)

        for i, job in enumerate(jobs):
            if i < len(results):
                score_data = results[i]
                scores = score_data["scores"]

                # Override location outside the LLM. Default is city/province
                # match; detailed kilometer distance is opt-in via env.
                cv_addr = cv_data.get("address") or cv_data.get("location", "")
                goong_score, dist_km = calculate_location_score(
                    cv_addr, job,
                )
                scores["location"] = goong_score

                # Tính lại WSM total
                w = weights
                total = sum(scores[dim] * w.get(dim, 0)
                           for dim in scores)

                job["match_score"] = round(total, 2)
                job["score_breakdown"] = scores
                job["comment"] = score_data.get("comment", "")
                job["llm_scored"] = True
                job["llm_time"] = score_data.get("llm_time", 0)
                job["distance_km"] = round(dist_km, 1) if dist_km is not None else None
            else:
                job["match_score"] = 5.0
                job["score_breakdown"] = {}
                job["comment"] = ""
                job["llm_scored"] = False
                job["distance_km"] = None

        return jobs

    except Exception as e:
        logger.error(f"LLM scoring failed: {e}, using default scores")
        for job in jobs:
            job["match_score"] = 5.0
            job["score_breakdown"] = {}
            job["comment"] = ""
            job["llm_scored"] = False
            job["distance_km"] = None
        return jobs


# ============================================================
# API Routes
# ============================================================

@app.route("/")
def index():
    return render_template("index.html", categories=CATEGORIES)


@app.route("/api/search", methods=["POST"])
def api_search():
    """Search API — full pipeline."""
    try:
        data = request.json
        cv_data = data.get("cv_data", {})
        selected_categories = data.get("categories", [])

        if not cv_data.get("skills", "").strip():
            return jsonify({"error": "Vui lòng nhập kỹ năng hoặc mô tả bản thân"}), 400

        import time as _time
        t0 = _time.time()

        jobs, search_mode, total = search_pipeline(
            cv_data, categories=selected_categories or None,
        )

        pipeline_time = round(_time.time() - t0, 1)
        llm_time = jobs[0].get("llm_time", 0) if jobs else 0

        return jsonify({
            "jobs": jobs[:50],
            "total": len(jobs),
            "search_mode": search_mode,
            "pipeline_time": pipeline_time,
            "llm_time": llm_time,
            "message": f"Tìm thấy {len(jobs)} công việc phù hợp ({search_mode}) — {pipeline_time}s",
        })

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/job-distance", methods=["POST"])
def api_job_distance():
    """Calculate detailed distance on demand for a selected job."""
    try:
        data = request.json or {}
        cv_data = data.get("cv_data", {})
        job = data.get("job")
        job_id = data.get("job_id")

        if not job and job_id:
            job = es_helper.get_job_by_id(job_id)
        if not job:
            return jsonify({"error": "Không tìm thấy thông tin công việc"}), 400

        cv_addr = cv_data.get("address") or cv_data.get("location", "")
        if not cv_addr:
            return jsonify({"error": "Thiếu địa điểm ứng viên"}), 400

        score, dist_km = calculate_location_score(cv_addr, job, detailed=True)
        return jsonify({
            "location_score": score,
            "distance_km": round(dist_km, 1) if dist_km is not None else None,
        })

    except Exception as e:
        logger.error(f"Distance error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/parse-cv", methods=["POST"])
def api_parse_cv():
    """Parse CV file bằng Groq AI."""
    try:
        if "cv_file" not in request.files:
            return jsonify({"error": "Không tìm thấy file CV"}), 400

        file = request.files["cv_file"]
        if file.filename == "":
            return jsonify({"error": "Chưa chọn file"}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": f"Định dạng không hỗ trợ. Dùng: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(temp_path)

        try:
            from cv_parser import parse_cv_file
            result = parse_cv_file(temp_path)

            if not result.get("success"):
                return jsonify({"error": result.get("error", "Không thể đọc CV")}), 400

            return jsonify({
                "success": True,
                "technical_skills": result.get("technical_skills", ""),
                "soft_skills": result.get("soft_skills", ""),
                "languages": result.get("languages", ""),
                "certificates": result.get("certificates", ""),
                "experience": result.get("experience", ""),
                "education": result.get("education", ""),
                "gender": result.get("gender", "both"),
                "location": result.get("location", ""),
                "suggested_categories": result.get("suggested_categories", []),
                "message": "Đã trích xuất thông tin từ CV",
            })
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    except Exception as e:
        logger.error(f"CV parse error: {e}", exc_info=True)
        return jsonify({"error": f"Lỗi xử lý CV: {str(e)}"}), 500


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    # Preload model on startup → no cold-start latency
    logger.info("Preloading embedding model...")
    _init_services()
    logger.info("Ready!")
    app.run(debug=True, host="0.0.0.0", port=5000)
