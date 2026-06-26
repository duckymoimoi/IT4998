"""Build a lightweight taxonomy from job skill/language/certificate terms.

The TopCV production index stores mixed role/domain/tool terms in
technical_skills, plus language and certificate terms in their own fields. This
script extracts unique terms from Elasticsearch across those fields and can
classify them with an LLM into a stable schema used by query planning.

Note: in production new terms reach the taxonomy through the pending queue
(``append_pending_terms`` in ``semantic_job_profile``), which already inspects
``technical_skills``, ``certificates`` and ``languages``. The ``--extract``
bootstrap below mirrors that same set of source fields for consistency.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from elasticsearch import Elasticsearch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TERMS_CSV = PROJECT_ROOT / "data" / "job_terms_unique.csv"
DEFAULT_TAXONOMY_JSON = PROJECT_ROOT / "data" / "job_term_taxonomy.json"
DEFAULT_OVERRIDES_JSON = PROJECT_ROOT / "data" / "job_term_taxonomy_overrides.json"
DEFAULT_PENDING_JSONL = PROJECT_ROOT / "data" / "job_term_taxonomy_pending.jsonl"

# Source fields mined for taxonomy terms. Kept in sync with
# ``semantic_job_profile._typed_terms`` so the bootstrap extraction and the
# production pending queue draw terms from the same fields.
DEFAULT_SOURCE_FIELDS = ["technical_skills", "languages", "certificates"]

TERM_TYPES = [
    "role",
    "domain",
    "technical_skill",
    "tool",
    "professional_skill",
    "soft_skill",
    "language",
    "certification",
    "noise",
]


def split_terms(text: str) -> List[str]:
    if not text:
        return []
    terms = []
    for part in re.split(r"[,;\n|]+", str(text)):
        # Parentheses and brackets can be meaningful parts of a normalized
        # label, for example "Kỹ sư an toàn lao động (HSE)".
        term = re.sub(r"\s+", " ", part).strip(" .:-")
        if term and term.lower() not in {"nan", "none", "null", "n/a", "na"}:
            terms.append(term)
    return terms


def extract_terms(es_host: str, index: str, fields: List[str] | None = None) -> List[Dict]:
    """Extract unique terms across one or more source fields.

    ``fields`` defaults to :data:`DEFAULT_SOURCE_FIELDS` so language and
    certificate terms are mined alongside ``technical_skills``.
    """
    fields = list(fields) if fields else list(DEFAULT_SOURCE_FIELDS)
    es = Elasticsearch(es_host)
    counter: Counter = Counter()
    examples: Dict[str, str] = {}

    for field in fields:
        page = es.search(
            index=index,
            body={
                "query": {"exists": {"field": field}},
                "_source": [field, "title"],
                "size": 500,
                "sort": ["_doc"],
            },
            scroll="2m",
        )

        scroll_id = page.get("_scroll_id")
        while True:
            hits = page["hits"]["hits"]
            if not hits:
                break
            for hit in hits:
                src = hit["_source"]
                for term in split_terms(src.get(field, "")):
                    key = term.lower()
                    counter[key] += 1
                    examples.setdefault(key, term)
            page = es.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = page.get("_scroll_id")

    rows = [
        {"term": examples[key], "frequency": freq}
        for key, freq in counter.most_common()
    ]
    return rows


def write_terms_csv(rows: Iterable[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["term", "frequency"])
        writer.writeheader()
        writer.writerows(rows)


def load_terms_csv(path: Path, limit: int = 0) -> List[Dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["frequency"] = int(float(row.get("frequency") or 0))
    if limit:
        return rows[:limit]
    return rows


def merge_pending_terms(terms_path: Path, pending_path: Path, consume: bool = True) -> List[Dict]:
    """Merge crawler-collected unknown terms into the canonical terms CSV."""
    existing = load_terms_csv(terms_path) if terms_path.exists() else []
    counter = Counter({
        str(row.get("term", "")).strip().lower(): int(row.get("frequency") or 0)
        for row in existing
        if str(row.get("term", "")).strip()
    })
    labels = {
        str(row.get("term", "")).strip().lower(): str(row.get("term", "")).strip()
        for row in existing
        if str(row.get("term", "")).strip()
    }
    pending_sources = sorted(
        pending_path.parent.glob(f"{pending_path.name}.consumed-*")
    )
    if pending_path.exists():
        if consume:
            claimed_path = pending_path.with_name(
                f"{pending_path.name}.consumed-{uuid.uuid4().hex}"
            )
            pending_path.replace(claimed_path)
            pending_sources.append(claimed_path)
        else:
            pending_sources.append(pending_path)

    for source in pending_sources:
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                term = str(item.get("term", "")).strip()
                if not term:
                    continue
                key = term.lower()
                counter[key] += 1
                labels.setdefault(key, term)

    rows = [
        {"term": labels[key], "frequency": frequency}
        for key, frequency in counter.most_common()
    ]
    write_terms_csv(rows, terms_path)
    if consume:
        for source in pending_sources:
            if source.name.startswith(f"{pending_path.name}.consumed-"):
                source.unlink(missing_ok=True)
    return rows


def load_groq_keys() -> List[str]:
    keys = []
    for i in range(1, 10):
        key = os.environ.get(f"GROQ_API_KEY_{i}")
        if key:
            keys.append(key.strip())
    single = os.environ.get("GROQ_API_KEY")
    if single and single.strip() not in keys:
        keys.append(single.strip())
    return keys


def build_prompt(rows: List[Dict]) -> str:
    compact_rows = [{"term": row["term"]} for row in rows]
    return f"""Bạn là chuyên gia chuẩn hóa taxonomy tuyển dụng Việt Nam.

Nhiệm vụ: phân loại từng term trích từ các field kỹ năng, ngoại ngữ và chứng chỉ của job TopCV.
Không cần biết toàn bộ job; chỉ phân loại ý nghĩa chính của term.

Schema type bắt buộc, chọn đúng 1:
- role: tên vai trò/chức danh/chức năng công việc, ví dụ Backend Developer, Kế toán tổng hợp, Sales Manager.
- domain: ngành/lĩnh vực kinh doanh hoặc domain nghề, ví dụ Ngân hàng, Bất động sản, IT - Phần mềm.
- technical_skill: kỹ năng kỹ thuật/chuyên môn có thể dùng để match năng lực, ví dụ Python, SQL, SEO, Thiết kế mạch.
- tool: công cụ/phần mềm/nền tảng/sản phẩm cụ thể, ví dụ Excel, MISA, AutoCAD, SAP, Facebook Ads.
- professional_skill: nghiệp vụ/chuyên môn không phải tool cụ thể, bao gồm kỹ năng và phương thức bán hàng, ví dụ Kế toán thuế, Tư vấn khách hàng, Quản lý dự án, B2B, Telesales.
- soft_skill: kỹ năng mềm, ví dụ giao tiếp, teamwork, đàm phán.
- language: ngoại ngữ hoặc mức độ ngoại ngữ.
- certification: chứng chỉ.
- noise: term nhiễu, quá chung, không hữu ích để search, hoặc placeholder/default sai.

Yêu cầu output:
- Chỉ trả về JSON array hợp lệ.
- Mỗi item giữ nguyên term đầu vào.
- confidence từ 0 đến 1.
- normalized_label ngắn gọn, chuẩn hóa chữ hoa/thường tự nhiên.
- notes rất ngắn, tiếng Việt.

Input terms:
{json.dumps(compact_rows, ensure_ascii=False)}

Output JSON array với format:
[
  {{
    "term": "...",
    "type": "role|domain|technical_skill|tool|professional_skill|soft_skill|language|certification|noise",
    "normalized_label": "...",
    "confidence": 0.0,
    "notes": "..."
  }}
]"""


def parse_json_array(text: str) -> List[Dict]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in LLM response")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("LLM response is not a JSON array")
    return data


def classify_with_groq(rows: List[Dict], model: str, batch_size: int, sleep_sec: float) -> List[Dict]:
    from groq import Groq

    keys = load_groq_keys()
    if not keys:
        raise RuntimeError("No GROQ_API_KEY/GROQ_API_KEY_N found in environment")

    classified = []
    key_idx = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompt = build_prompt(batch)
        last_error = None
        for attempt in range(len(keys)):
            key = keys[(key_idx + attempt) % len(keys)]
            client = Groq(api_key=key)
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_completion_tokens=5000,
                )
                raw = response.choices[0].message.content or ""
                items = parse_json_array(raw)
                by_term = {str(item.get("term", "")).strip().lower(): item for item in items}
                for row in batch:
                    item = by_term.get(row["term"].lower())
                    if not item:
                        item = {
                            "term": row["term"],
                            "type": "noise",
                            "normalized_label": row["term"],
                            "confidence": 0.0,
                            "notes": "missing_from_llm_output",
                        }
                    item["frequency"] = row["frequency"]
                    if item.get("type") not in TERM_TYPES:
                        item["type"] = "noise"
                        item["confidence"] = min(float(item.get("confidence") or 0), 0.5)
                    classified.append(item)
                key_idx = (key_idx + attempt + 1) % len(keys)
                print(f"classified {min(start + batch_size, len(rows))}/{len(rows)}")
                if sleep_sec:
                    time.sleep(sleep_sec)
                break
            except Exception as exc:
                last_error = exc
                continue
        else:
            raise RuntimeError(f"Failed batch starting at {start}: {last_error}")
    return classified


def write_json(rows: List[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)


def dedupe_taxonomy_rows(rows: List[Dict]) -> List[Dict]:
    """Keep one row per normalized input term, with the latest row winning."""
    deduped = {}
    order = []
    for row in rows:
        key = str(row.get("term", "")).strip().lower()
        if not key:
            continue
        if key not in deduped:
            order.append(key)
        deduped[key] = row
    return [deduped[key] for key in order]


def apply_overrides(taxonomy_path: Path, overrides_path: Path) -> List[Dict]:
    rows = dedupe_taxonomy_rows(load_existing_taxonomy(taxonomy_path))
    overrides = load_existing_taxonomy(overrides_path)
    override_by_term = {
        str(row.get("term", "")).strip().lower(): row
        for row in overrides
        if str(row.get("term", "")).strip()
    }

    cleaned = []
    for row in rows:
        # Remove unexpected keys produced by occasional malformed LLM JSON.
        item = {
            "term": row.get("term", ""),
            "type": row.get("type", "noise"),
            "normalized_label": row.get("normalized_label") or row.get("term", ""),
            "confidence": row.get("confidence", 0.0),
            "notes": row.get("notes", ""),
            "frequency": row.get("frequency", 0),
        }

        override = override_by_term.get(str(item["term"]).strip().lower())
        if override:
            item.update({
                "type": override.get("type", item["type"]),
                "normalized_label": override.get("normalized_label", item["normalized_label"]),
                "confidence": override.get("confidence", item["confidence"]),
                "notes": override.get("notes", item["notes"]),
            })

        if item["type"] not in TERM_TYPES:
            item["type"] = "noise"
            item["confidence"] = min(float(item.get("confidence") or 0), 0.5)
            item["notes"] = "invalid type normalized to noise"

        try:
            item["confidence"] = round(float(item.get("confidence") or 0), 3)
        except (TypeError, ValueError):
            item["confidence"] = 0.0
        try:
            item["frequency"] = int(float(item.get("frequency") or 0))
        except (TypeError, ValueError):
            item["frequency"] = 0

        cleaned.append(item)

    cleaned = dedupe_taxonomy_rows(cleaned)
    write_json(cleaned, taxonomy_path)
    return cleaned


def load_existing_taxonomy(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def classify_new_terms(
    terms_path: Path = DEFAULT_TERMS_CSV,
    taxonomy_path: Path = DEFAULT_TAXONOMY_JSON,
    overrides_path: Path = DEFAULT_OVERRIDES_JSON,
    *,
    minimum_terms: int = 0,
    limit: int = 0,
    batch_size: int = 40,
    sleep_sec: float = 2.0,
    model: str = "openai/gpt-oss-120b",
    apply_manual_overrides: bool = True,
    resume: bool = True,
) -> Dict:
    """Classify unrecognized terms when the configured threshold is reached."""
    rows = load_terms_csv(terms_path, limit=limit)
    existing = dedupe_taxonomy_rows(
        load_existing_taxonomy(taxonomy_path) if resume else []
    )
    existing_terms = {
        str(row.get("term", "")).strip().lower()
        for row in existing
        if str(row.get("term", "")).strip()
    }
    remaining = [
        row for row in rows
        if row["term"].strip().lower() not in existing_terms
    ]
    result = {
        "terms": len(rows),
        "existing": len(existing),
        "remaining": len(remaining),
        "classified": 0,
        "classified_terms": [],
        "triggered": False,
    }
    if len(remaining) < max(0, minimum_terms):
        return result

    result["triggered"] = bool(remaining)
    classified = list(existing)
    newly_classified = []
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start:start + batch_size]
        batch_result = classify_with_groq(
            batch, model=model, batch_size=batch_size, sleep_sec=sleep_sec,
        )
        classified = dedupe_taxonomy_rows(classified + batch_result)
        newly_classified.extend(batch_result)
        write_json(classified, taxonomy_path)
        print(f"checkpoint wrote {len(classified)} terms to {taxonomy_path}")

    affected_terms = [
        str(row.get("term", "")).strip()
        for row in newly_classified
        if str(row.get("term", "")).strip()
    ]
    if remaining and apply_manual_overrides:
        classified = apply_overrides(taxonomy_path, overrides_path)
        affected_terms.extend(
            str(row.get("term", "")).strip()
            for row in load_existing_taxonomy(overrides_path)
            if str(row.get("term", "")).strip()
        )

    result["classified"] = len(newly_classified)
    result["classified_terms"] = affected_terms
    result["taxonomy_rows"] = len(classified)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build job term taxonomy from skill/language/certificate fields")
    parser.add_argument("--es-host", default=os.environ.get("ES_HOST", "http://localhost:9200"))
    parser.add_argument("--index", default=os.environ.get("ES_INDEX", "topcv_jobs_production"))
    parser.add_argument(
        "--fields", nargs="+", default=None,
        help="Source fields to mine (default: technical_skills languages certificates)",
    )
    parser.add_argument(
        "--field", default=None,
        help="Deprecated: single source field. Use --fields instead.",
    )
    parser.add_argument("--terms-csv", type=Path, default=DEFAULT_TERMS_CSV)
    parser.add_argument("--taxonomy-json", type=Path, default=DEFAULT_TAXONOMY_JSON)
    parser.add_argument("--overrides-json", type=Path, default=DEFAULT_OVERRIDES_JSON)
    parser.add_argument("--pending-jsonl", type=Path, default=DEFAULT_PENDING_JSONL)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--merge-pending", action="store_true")
    parser.add_argument("--classify", action="store_true")
    parser.add_argument("--apply-overrides", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0 = process all terms")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--sleep-sec", type=float, default=2.0)
    parser.add_argument("--model", default=os.environ.get("TAXONOMY_MODEL", "openai/gpt-oss-120b"))
    parser.add_argument("--resume", action="store_true", help="Skip terms already present in taxonomy-json")
    args = parser.parse_args()

    if args.extract:
        if args.fields:
            source_fields = args.fields
        elif args.field:
            source_fields = [args.field]
        else:
            source_fields = DEFAULT_SOURCE_FIELDS
        rows = extract_terms(args.es_host, args.index, source_fields)
        write_terms_csv(rows, args.terms_csv)
        print(
            f"wrote {len(rows)} unique terms from {source_fields} to {args.terms_csv}"
        )

    if args.merge_pending:
        rows = merge_pending_terms(args.terms_csv, args.pending_jsonl)
        print(f"merged pending terms; wrote {len(rows)} unique terms to {args.terms_csv}")

    if args.classify:
        result = classify_new_terms(
            terms_path=args.terms_csv,
            taxonomy_path=args.taxonomy_json,
            overrides_path=args.overrides_json,
            limit=args.limit,
            batch_size=args.batch_size,
            sleep_sec=args.sleep_sec,
            model=args.model,
            apply_manual_overrides=False,
            resume=args.resume,
        )
        print(
            f"loaded {result['terms']} terms; existing={result['existing']}; "
            f"remaining={result['remaining']}"
        )
        print(
            f"wrote {result.get('taxonomy_rows', result['existing'])} "
            f"classified terms to {args.taxonomy_json}"
        )

    if args.apply_overrides:
        cleaned = apply_overrides(args.taxonomy_json, args.overrides_json)
        print(
            f"applied overrides from {args.overrides_json}; "
            f"wrote {len(cleaned)} rows to {args.taxonomy_json}"
        )


if __name__ == "__main__":
    main()
