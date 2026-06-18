"""
Job Matching Web Application — Production Pipeline.

Pipeline:
  CV Input → ESCO Expansion → Hybrid Search (BM25 + kNN + RRF)
  → Top-N → LLM 6-dim Scoring → WSM Final Ranking

Tách biệt rõ ràng: retrieval (ES) → scoring (LLM) → presentation (Flask).
"""

import os
import math
import re
import tempfile
import logging
from pathlib import Path

from dotenv import load_dotenv

SRC_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(SRC_ENV_PATH)
load_dotenv()  # Fallback for project-root .env if present.

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from job_matching.retrieval.elastic_helper import ElasticHelper
from job_matching.shared.vietnam_cities_data import get_city_info

logger = logging.getLogger(__name__)

# ============================================================
# Flask App
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
JOB_TERM_TAXONOMY_PATH = Path(
    os.environ.get(
        "JOB_TERM_TAXONOMY_PATH",
        str(PROJECT_ROOT / "data" / "job_term_taxonomy.json"),
    )
)
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
    static_url_path="",
)

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "bmp", "tiff"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

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
_job_term_taxonomy = None
_job_term_taxonomy_rows = 0
_job_term_taxonomy_mtime_ns = None


def _env_bool(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _init_services():
    """Initialize embedding + ESCO + SkillGraph services (chỉ gọi 1 lần)."""
    global _embedding_service, _esco_expander, _skill_graph, _use_hybrid

    if _embedding_service is not None:
        return

    # Skill Knowledge Graph is an opt-in enrichment. Experiments selected
    # ESCO-only hybrid search as the stable production default.
    if _env_bool("ENABLE_SKILL_GRAPH", default=False):
        try:
            from job_matching.enrichment.skill_graph import get_skill_graph
            _skill_graph = get_skill_graph()
            logger.info("Skill Knowledge Graph enabled")
        except Exception as e:
            logger.warning(f"SkillGraph not available: {e}")
    else:
        logger.info("Skill Knowledge Graph disabled (set ENABLE_SKILL_GRAPH=1 to enable)")

    try:
        if es_helper.has_usable_embeddings(KNN_VECTOR_FIELD):
            from job_matching.retrieval.embedding_service import get_embedding_service
            _embedding_service = get_embedding_service()
            _use_hybrid = True
            logger.info("Hybrid search (BM25 + kNN) enabled")

            # ESCO expander
            try:
                from job_matching.enrichment.esco_expander import get_esco_expander
                _esco_expander = get_esco_expander(
                    embedding_service=_embedding_service,
                    controlled_min_sim=ESCO_CONTROLLED_MIN_SIM,
                    controlled_min_margin=ESCO_CONTROLLED_MIN_MARGIN,
                    controlled_max_terms=ESCO_CONTROLLED_MAX_TERMS,
                )
                logger.info("ESCO skill expansion enabled")
            except Exception as e:
                logger.warning(f"ESCO not available: {e}")
        else:
            logger.info("No usable embeddings - BM25-only mode")
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
ENABLE_JOB_TERM_TAXONOMY = _env_bool("ENABLE_JOB_TERM_TAXONOMY", default=True)
SCORING_TOP_N = _env_int("SCORING_TOP_N", 30)
RETRIEVAL_SIZE = _env_int("RETRIEVAL_SIZE", 80)
RELAXED_RETRIEVAL_MIN_RESULTS = _env_int("RELAXED_RETRIEVAL_MIN_RESULTS", 12)
BM25_MIN_SHOULD_MATCH = os.environ.get("BM25_MIN_SHOULD_MATCH", "2")
BM25_RELAXED_MIN_SHOULD_MATCH = os.environ.get("BM25_RELAXED_MIN_SHOULD_MATCH", "1")
RRF_BM25_WEIGHT = float(os.environ.get("RRF_BM25_WEIGHT", "1.0"))
RRF_KNN_WEIGHT = float(os.environ.get("RRF_KNN_WEIGHT", "1.0"))
KNN_VECTOR_FIELD = os.environ.get("KNN_VECTOR_FIELD", "embedding").strip()
BM25_CORE_SKILL_LIMIT = _env_int("BM25_CORE_SKILL_LIMIT", 12)
KNN_SECONDARY_SKILL_LIMIT = _env_int("KNN_SECONDARY_SKILL_LIMIT", 5)
KNN_EXPANSION_TERM_LIMIT = _env_int("KNN_EXPANSION_TERM_LIMIT", 12)
ENABLE_QUERY_EXPANSION = _env_bool("ENABLE_QUERY_EXPANSION", default=False)
ENABLE_ESCO_EXPANSION = _env_bool(
    "ENABLE_ESCO_EXPANSION", default=ENABLE_QUERY_EXPANSION,
)
ENABLE_SKILL_GRAPH_EXPANSION = _env_bool(
    "ENABLE_SKILL_GRAPH_EXPANSION", default=ENABLE_QUERY_EXPANSION,
)
ESCO_CONTROLLED_MIN_SIM = _env_float("ESCO_CONTROLLED_MIN_SIM", 0.75)
ESCO_CONTROLLED_MIN_MARGIN = _env_float("ESCO_CONTROLLED_MIN_MARGIN", 0.01)
ESCO_CONTROLLED_MAX_TERMS = _env_int("ESCO_CONTROLLED_MAX_TERMS", 6)


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


def _as_text_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in re.split(r"[,;\n]", str(value)) if part.strip()]


def _load_job_term_taxonomy():
    """Load term -> taxonomy metadata from data/job_term_taxonomy.json."""
    global _job_term_taxonomy, _job_term_taxonomy_rows, _job_term_taxonomy_mtime_ns
    if not ENABLE_JOB_TERM_TAXONOMY:
        _job_term_taxonomy_rows = 0
        return {}

    current_mtime_ns = (
        JOB_TERM_TAXONOMY_PATH.stat().st_mtime_ns
        if JOB_TERM_TAXONOMY_PATH.exists()
        else None
    )
    if (
        _job_term_taxonomy is not None
        and current_mtime_ns == _job_term_taxonomy_mtime_ns
    ):
        return _job_term_taxonomy

    taxonomy = {}
    row_count = 0
    try:
        import json

        if JOB_TERM_TAXONOMY_PATH.exists():
            rows = json.loads(JOB_TERM_TAXONOMY_PATH.read_text(encoding="utf-8"))
            row_count = len(rows)
            for row in rows:
                term = str(row.get("term", "")).strip()
                label = str(row.get("normalized_label", "")).strip()
                if term:
                    taxonomy.setdefault(term.lower(), row)
                if label:
                    taxonomy.setdefault(label.lower(), row)
            logger.info("Loaded %s job term taxonomy rows", row_count)
        else:
            logger.warning("Job term taxonomy not found: %s", JOB_TERM_TAXONOMY_PATH)
    except Exception as exc:
        logger.warning("Cannot load job term taxonomy: %s", exc)

    _job_term_taxonomy = taxonomy
    _job_term_taxonomy_rows = row_count
    _job_term_taxonomy_mtime_ns = current_mtime_ns
    return _job_term_taxonomy


def _taxonomy_status():
    taxonomy = _load_job_term_taxonomy()
    return {
        "enabled": ENABLE_JOB_TERM_TAXONOMY,
        "path": str(JOB_TERM_TAXONOMY_PATH),
        "rows": _job_term_taxonomy_rows,
        "lookup_entries": len(taxonomy),
    }


def _dedupe_terms(*term_groups, limit=None):
    terms = []
    seen = set()
    for group in term_groups:
        for term in group:
            cleaned = re.sub(r"\s+", " ", str(term)).strip(" .:-")
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            terms.append(cleaned)
            seen.add(key)
            if limit and len(terms) >= limit:
                return terms
    return terms


def _join_terms(terms):
    return ", ".join([term for term in terms if term])


def _has_primary_skills(cv_data):
    return bool(
        _as_text_list(cv_data.get("target_roles"))
        or _as_text_list(cv_data.get("roles"))
        or _as_text_list(cv_data.get("skills"))
        or _as_text_list(cv_data.get("technical_skills"))
    )


def _taxonomy_surfaces(term, item):
    surfaces = [term]
    if item:
        label = str(item.get("normalized_label", "")).strip()
        if label and label.lower() != str(term).strip().lower():
            surfaces.append(label)
    return surfaces


def _split_terms_by_taxonomy(terms):
    """Return BM25/supporting terms using the offline term taxonomy."""
    taxonomy = _load_job_term_taxonomy()
    bm25_terms = []
    supporting_terms = []
    ignored_terms = []

    for term in terms:
        item = taxonomy.get(str(term).strip().lower()) if taxonomy else None
        term_type = item.get("type") if item else "unknown"
        surfaces = _taxonomy_surfaces(term, item)

        if term_type == "noise":
            ignored_terms.extend(surfaces)
        elif term_type == "soft_skill":
            ignored_terms.extend(surfaces)
        elif term_type in {
            "role", "domain", "technical_skill", "tool",
            "professional_skill", "language", "certification",
            "unknown",
        }:
            bm25_terms.extend(surfaces)
        else:
            supporting_terms.extend(surfaces)

    return bm25_terms, supporting_terms, ignored_terms


def _confirmed_expansion_terms(terms):
    """Keep only taxonomy-confirmed semantic terms as enrichment seeds."""
    taxonomy = _load_job_term_taxonomy()
    accepted_types = {"domain", "technical_skill", "tool", "professional_skill"}
    output = []
    for term in terms:
        item = taxonomy.get(str(term).strip().lower()) if taxonomy else None
        if not item or item.get("type") not in accepted_types:
            continue
        output.extend(_taxonomy_surfaces(term, item))
    return _dedupe_terms(output, limit=20)


def _expansion_seed_types(terms):
    """Return normalized seed -> controlled-vocabulary type for ESCO guards."""
    taxonomy = _load_job_term_taxonomy()
    output = {}
    for term in terms:
        item = taxonomy.get(str(term).strip().lower()) if taxonomy else None
        if not item:
            continue
        term_type = item.get("type")
        for surface in _taxonomy_surfaces(term, item):
            output[str(surface).strip().lower()] = term_type
    return output


def _build_cv_semantic_profile(query_plan, expansion_terms=None):
    """Represent a CV query in the same labeled style as job semantic profiles."""
    lines = []
    roles = query_plan.get("target_roles", [])
    core = query_plan.get("core_skills", [])
    secondary = query_plan.get("secondary_skills", [])[:KNN_SECONDARY_SKILL_LIMIT]
    languages = query_plan.get("languages", [])
    certificates = query_plan.get("certificates", [])

    if roles:
        lines.append(f"Vai trò: {', '.join(roles)}.")
    if core:
        lines.append(f"Kỹ năng cốt lõi: {', '.join(core)}.")
    if secondary:
        lines.append(f"Kỹ năng bổ trợ: {', '.join(secondary)}.")
    if languages:
        lines.append(f"Ngoại ngữ: {', '.join(languages)}.")
    if certificates:
        lines.append(f"Chứng chỉ: {', '.join(certificates)}.")
    if expansion_terms:
        lines.append(
            f"Thuật ngữ liên quan: {', '.join(expansion_terms[:KNN_EXPANSION_TERM_LIMIT])}."
        )
    return " ".join(lines).strip()


def _build_retrieval_queries(cv_data):
    """Build focused BM25/kNN query texts from parsed CV fields.

    In the current production index, TopCV chuyên môn tags are stored in
    technical_skills. Keep BM25 focused on that field's equivalent CV signal;
    Education and experience are handled later by scoring instead of being
    mixed into the search text.
    """
    target_roles = _as_text_list(cv_data.get("target_roles")) or _as_text_list(
        cv_data.get("roles")
    )
    primary_skills = _as_text_list(cv_data.get("skills")) or _as_text_list(
        cv_data.get("technical_skills")
    )
    if not target_roles and not primary_skills:
        # Generic/entry-level profiles may contain only soft skills. Keep a
        # broad role signal when available instead of returning an empty query.
        target_roles = _as_text_list(cv_data.get("category_target"))
    explicit_core = _as_text_list(cv_data.get("core_skills"))
    explicit_secondary = _as_text_list(cv_data.get("secondary_skills"))
    certificates = _as_text_list(cv_data.get("certificates"))
    languages = _as_text_list(cv_data.get("languages"))
    role_bm25, role_supporting, role_ignored = _split_terms_by_taxonomy(target_roles)
    primary_bm25, primary_supporting, ignored_terms = _split_terms_by_taxonomy(primary_skills)
    cert_bm25, cert_supporting, cert_ignored = _split_terms_by_taxonomy(certificates)
    lang_bm25, lang_supporting, lang_ignored = _split_terms_by_taxonomy(languages)
    semantic_skills = _dedupe_terms(primary_bm25, primary_supporting)
    if explicit_core:
        explicit_core_bm25, explicit_core_supporting, explicit_core_ignored = (
            _split_terms_by_taxonomy(explicit_core)
        )
        core_skills = _dedupe_terms(
            explicit_core_bm25, explicit_core_supporting,
            limit=BM25_CORE_SKILL_LIMIT,
        )
        explicit_secondary_bm25, explicit_secondary_supporting, explicit_secondary_ignored = (
            _split_terms_by_taxonomy(explicit_secondary)
        )
        secondary_skills = _dedupe_terms(
            explicit_secondary_bm25,
            explicit_secondary_supporting,
        )
    else:
        explicit_core_ignored, explicit_secondary_ignored = [], []
        core_skills = semantic_skills[:BM25_CORE_SKILL_LIMIT]
        secondary_skills = semantic_skills[BM25_CORE_SKILL_LIMIT:]
    core_bm25, core_supporting, core_ignored = _split_terms_by_taxonomy(core_skills)

    bm25_terms = _dedupe_terms(
        role_bm25, core_bm25, cert_bm25, lang_bm25,
        limit=24,
    )
    knn_terms = _dedupe_terms(
        role_bm25,
        role_supporting,
        primary_bm25,
        primary_supporting,
        cert_bm25,
        cert_supporting,
        lang_bm25,
        lang_supporting,
        limit=64,
    )

    bm25_text = _join_terms(bm25_terms)
    query_plan = {
        "target_roles": target_roles,
        "primary_skills": primary_skills,
        "core_skills": core_skills,
        "secondary_skills": secondary_skills,
        "languages": languages,
        "certificates": certificates,
        "bm25_text": bm25_text,
        "bm25_roles": _dedupe_terms(role_bm25, role_supporting),
        "bm25_evidence_terms": _dedupe_terms(core_bm25, cert_bm25, lang_bm25),
        "knn_terms": knn_terms,
        "expansion_terms": _confirmed_expansion_terms(core_skills),
        "ignored_query_terms": _dedupe_terms(
            role_ignored, core_ignored, explicit_core_ignored, explicit_secondary_ignored,
            ignored_terms, cert_ignored, lang_ignored,
        ),
        "taxonomy": _taxonomy_status(),
    }
    query_plan["expansion_text"] = _join_terms(query_plan["expansion_terms"])
    query_plan["knn_text"] = _build_cv_semantic_profile(query_plan)
    return query_plan


# ============================================================
# Search Pipeline
# ============================================================

def search_pipeline(cv_data, top_n=None):
    """
    Full pipeline: taxonomy query planning → hybrid retrieval → LLM scoring.
    
    Args:
        cv_data: dict with skills, experience, education, location, salary
        top_n: số jobs gửi cho LLM scoring
    
    Returns:
        (jobs_with_scores, search_mode, total_found)
    """
    _init_services()
    top_n = top_n or SCORING_TOP_N

    query_plan = _build_retrieval_queries(cv_data)
    bm25_text = query_plan["bm25_text"]
    if not bm25_text:
        return [], "error", 0

    cv_gender = cv_data.get("gender", "both")
    # BM25 remains taxonomy-clean. ESCO and SkillGraph enrich only the semantic
    # query used to create the kNN vector.
    expansion_seed_text = query_plan["expansion_text"]
    esco_terms = []
    graph_terms = []
    if ENABLE_ESCO_EXPANSION and expansion_seed_text and _esco_expander:
        esco_terms = _esco_expander.expand_terms_controlled(
            query_plan["expansion_terms"],
            term_types=_expansion_seed_types(query_plan["expansion_terms"]),
        )

    if ENABLE_SKILL_GRAPH_EXPANSION and expansion_seed_text and _skill_graph:
        graph_text = _skill_graph.expand_skills_text(
            expansion_seed_text, max_terms=KNN_EXPANSION_TERM_LIMIT,
        )
        graph_terms.extend(_as_text_list(graph_text))

    seed_keys = {term.lower() for term in query_plan["expansion_terms"]}
    # ESCO labels are trusted only after passing the expander's strict
    # similarity/margin/type guards. Graph terms still need confirmation by
    # the internal vocabulary because graph traversal is broader by design.
    confirmed_graph_terms = [
        term for term in _confirmed_expansion_terms(graph_terms)
        if term.lower() not in seed_keys
    ]
    expansion_terms = _dedupe_terms(
        esco_terms,
        confirmed_graph_terms,
        limit=KNN_EXPANSION_TERM_LIMIT,
    )
    semantic_query_text = _build_cv_semantic_profile(query_plan, expansion_terms)
    logger.info(
        "Structured kNN query: %s core, %s secondary, %s expansion terms",
        len(query_plan["core_skills"]),
        min(len(query_plan["secondary_skills"]), KNN_SECONDARY_SKILL_LIMIT),
        len(expansion_terms),
    )

    # --- Stage 2: Hybrid Retrieval ---
    if _use_hybrid and _embedding_service:
        cv_text = _embedding_service.build_cv_text({"skills": semantic_query_text})
        query_vector = _embedding_service.encode_single(cv_text)
        jobs, total = es_helper.search_jobs_hybrid(
            bm25_text, query_vector, size=RETRIEVAL_SIZE,
            categories=None,
            cv_gender=cv_gender, exclude_expired=True,
            bm25_min_should_match=BM25_MIN_SHOULD_MATCH,
            bm25_weight=RRF_BM25_WEIGHT,
            knn_weight=RRF_KNN_WEIGHT,
            vector_field=KNN_VECTOR_FIELD,
            role_terms=query_plan["bm25_roles"],
            evidence_terms=query_plan["bm25_evidence_terms"],
        )
        search_mode = "hybrid"
        if ENABLE_ESCO_EXPANSION and _esco_expander:
            search_mode += "+esco"
        if ENABLE_SKILL_GRAPH_EXPANSION and _skill_graph:
            search_mode += "+graph"
    else:
        jobs, total = es_helper.search_jobs_by_profile(
            bm25_text, size=RETRIEVAL_SIZE,
            categories=None,
            cv_gender=cv_gender, exclude_expired=True,
            bm25_min_should_match=BM25_MIN_SHOULD_MATCH,
            role_terms=query_plan["bm25_roles"],
            evidence_terms=query_plan["bm25_evidence_terms"],
        )
        search_mode = "bm25"

    if len(jobs) < RELAXED_RETRIEVAL_MIN_RESULTS:
        logger.info(
            "Relaxing retrieval: %s results < %s",
            len(jobs), RELAXED_RETRIEVAL_MIN_RESULTS,
        )
        if _use_hybrid and _embedding_service:
            jobs, total = es_helper.search_jobs_hybrid(
                bm25_text, query_vector, size=RETRIEVAL_SIZE,
                categories=None,
                cv_gender=cv_gender, exclude_expired=True,
                bm25_min_should_match=BM25_RELAXED_MIN_SHOULD_MATCH,
                num_candidates=RETRIEVAL_SIZE * 4,
                bm25_weight=RRF_BM25_WEIGHT,
                knn_weight=RRF_KNN_WEIGHT,
                vector_field=KNN_VECTOR_FIELD,
                role_terms=query_plan["bm25_roles"],
                evidence_terms=query_plan["bm25_evidence_terms"],
            )
        else:
            jobs, total = es_helper.search_jobs_by_profile(
                bm25_text, size=RETRIEVAL_SIZE,
                categories=None,
                cv_gender=cv_gender, exclude_expired=True,
                bm25_min_should_match=BM25_RELAXED_MIN_SHOULD_MATCH,
                role_terms=query_plan["bm25_roles"],
                evidence_terms=query_plan["bm25_evidence_terms"],
            )
        search_mode += "+relaxed"

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
    # Rank the partial AI-scored set first. Jobs that missed the scoring time
    # limit remain in retrieval order through their fallback scores.
    all_jobs.sort(
        key=lambda x: (bool(x.get("llm_scored")), x.get("match_score", 0)),
        reverse=True,
    )

    return all_jobs, search_mode, total


def _score_with_llm(cv_data, jobs):
    """Gọi LLM scorer, fallback sang heuristic nếu fail."""
    def retrieval_fallback_score(rank):
        # Keep fallback scores clearly below trusted LLM scores, while still
        # preserving retrieval order when the external scorer is unavailable.
        return round(max(3.0, 5.0 - rank * 0.12), 2)

    try:
        from job_matching.scoring.llm_scorer import score_batch, DEFAULT_WEIGHTS

        weights = cv_data.get("weights", DEFAULT_WEIGHTS)
        results = score_batch(cv_data, jobs, weights=weights)

        for i, job in enumerate(jobs):
            if i < len(results):
                score_data = results[i]
                if score_data.get("fallback"):
                    job["match_score"] = retrieval_fallback_score(i)
                    job["score_breakdown"] = {}
                    job["comment"] = ""
                    job["llm_scored"] = False
                    job["distance_km"] = None
                    continue

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
        for i, job in enumerate(jobs):
            job["match_score"] = retrieval_fallback_score(i)
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
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def api_search():
    """Search API — full pipeline."""
    try:
        data = request.get_json(silent=True) or {}
        cv_data = data.get("cv_data", {})
        if not isinstance(cv_data, dict) or not _has_primary_skills(cv_data):
            return jsonify({"error": "Vui lòng nhập kỹ năng chuyên môn"}), 400

        import time as _time
        t0 = _time.time()

        jobs, search_mode, total = search_pipeline(cv_data)

        pipeline_time = round(_time.time() - t0, 1)
        llm_time = jobs[0].get("llm_time", 0) if jobs else 0
        ai_scored_count = sum(bool(job.get("llm_scored")) for job in jobs)

        return jsonify({
            "jobs": jobs[:50],
            "total": len(jobs),
            "ai_scored_count": ai_scored_count,
            "retrieval_only_count": len(jobs) - ai_scored_count,
            "search_mode": search_mode,
            "pipeline_time": pipeline_time,
            "llm_time": llm_time,
            "message": (
                f"Tìm thấy {len(jobs)} công việc phù hợp; "
                f"AI đã chấm nhanh {ai_scored_count} job "
                f"({search_mode}) — {pipeline_time}s"
            ),
        })

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug-query", methods=["POST"])
def api_debug_query():
    """Return the query plan built from parsed CV data for demos/debugging."""
    try:
        data = request.json or {}
        cv_data = data.get("cv_data", data)
        if not isinstance(cv_data, dict):
            return jsonify({"error": "cv_data must be an object"}), 400
        query_plan = _build_retrieval_queries(cv_data)
        return jsonify(query_plan)
    except Exception as e:
        logger.error(f"Debug query error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/jobs", methods=["GET"])
def api_jobs():
    """Browse jobs currently stored in the active Elasticsearch index."""
    try:
        page = request.args.get("page", 1, type=int)
        size = request.args.get("size", 20, type=int)
        query = request.args.get("q", "", type=str).strip()

        data = es_helper.list_jobs(page=page, size=size, query=query)
        data["index_name"] = es_helper.index_name
        return jsonify(data)
    except Exception as e:
        logger.error(f"List jobs error: {e}", exc_info=True)
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


@app.route("/api/job-detail-score", methods=["POST"])
def api_job_detail_score():
    """Run a focused one-job LLM score on demand without reranking the list."""
    try:
        data = request.json or {}
        cv_data = data.get("cv_data", {})
        job = data.get("job")
        job_id = data.get("job_id")

        if not isinstance(cv_data, dict) or not _has_primary_skills(cv_data):
            return jsonify({"error": "Thiếu thông tin CV để chấm điểm"}), 400
        if not job and job_id:
            job = es_helper.get_job_by_id(job_id)
        if not job:
            return jsonify({"error": "Không tìm thấy thông tin công việc"}), 400

        from job_matching.scoring.llm_scorer import score_detail_with_evidence, DEFAULT_WEIGHTS

        weights = cv_data.get("weights", DEFAULT_WEIGHTS)
        score_data = score_detail_with_evidence(cv_data, job, weights=weights)
        if not score_data:
            return jsonify({"error": "Không chấm được công việc này"}), 502

        scores = score_data.get("scores", {})

        cv_addr = cv_data.get("address") or cv_data.get("location", "")
        goong_score, dist_km = calculate_location_score(cv_addr, job)
        scores["location"] = goong_score

        total = sum(
            float(scores.get(dim, 0)) * float(weights.get(dim, 0))
            for dim in ["relevance", "skills", "experience", "education", "location", "salary"]
        )

        return jsonify({
            "match_score": round(total, 2),
            "score_breakdown": scores,
            "comment": score_data.get("comment", ""),
            "evidence": score_data.get("evidence", {}),
            "llm_scored": True,
            "detail_scored": True,
            "llm_time": score_data.get("llm_time", 0),
            "distance_km": round(dist_km, 1) if dist_km is not None else job.get("distance_km"),
        })

    except Exception as e:
        logger.error(f"Detail scoring error: {e}", exc_info=True)
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
            from job_matching.scoring.cv_parser import parse_cv_file
            result = parse_cv_file(temp_path)

            if not result.get("success"):
                return jsonify({"error": result.get("error", "Không thể đọc CV")}), 400

            return jsonify({
                "success": True,
                "target_roles": result.get("target_roles", ""),
                "core_skills": result.get("core_skills", ""),
                "secondary_skills": result.get("secondary_skills", ""),
                "technical_skills": result.get("technical_skills", ""),
                "languages": result.get("languages", ""),
                "certificates": result.get("certificates", ""),
                "experience": result.get("experience", ""),
                "education": result.get("education", ""),
                "gender": result.get("gender", "both"),
                "location": result.get("location", ""),
                "cv_markdown": result.get("raw_text", ""),
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

