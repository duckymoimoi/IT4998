"""Audit controlled-vocabulary terms against strict ESCO expansion rules."""

import argparse
import json
from collections import Counter
from pathlib import Path

from job_matching.enrichment.esco_expander import ESCOExpander
from job_matching.retrieval.embedding_service import get_embedding_service


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TAXONOMY = PROJECT_ROOT / "data" / "job_term_taxonomy.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "esco_controlled_mapping_audit.json"
SEMANTIC_TYPES = {"technical_skill", "professional_skill"}


def load_seeds(path, limit):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = {}
    for row in rows:
        term_type = row.get("type")
        if term_type not in SEMANTIC_TYPES:
            continue
        term = str(row.get("normalized_label") or row.get("term") or "").strip()
        if not term:
            continue
        key = term.lower()
        frequency = int(row.get("frequency") or 0)
        if key not in selected or frequency > selected[key]["frequency"]:
            selected[key] = {
                "term": term,
                "term_type": term_type,
                "frequency": frequency,
            }
    return sorted(
        selected.values(),
        key=lambda item: (-item["frequency"], item["term"].lower()),
    )[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--min-sim", type=float, default=0.82)
    parser.add_argument("--min-margin", type=float, default=0.03)
    args = parser.parse_args()

    seeds = load_seeds(args.taxonomy, args.limit)
    embedding_service = get_embedding_service()
    expander = ESCOExpander(
        embedding_service=embedding_service,
        controlled_min_sim=args.min_sim,
        controlled_min_margin=args.min_margin,
        controlled_max_terms=max(len(seeds), 1),
    )
    terms = [item["term"] for item in seeds]
    term_types = {item["term"].lower(): item["term_type"] for item in seeds}
    _, audit = expander.expand_terms_controlled(
        terms,
        term_types=term_types,
        return_audit=True,
    )

    frequency_by_term = {item["term"].lower(): item["frequency"] for item in seeds}
    for record in audit:
        record["frequency"] = frequency_by_term.get(record["seed"].lower(), 0)
    summary = {
        "parameters": {
            "min_similarity": args.min_sim,
            "min_margin": args.min_margin,
            "semantic_types": sorted(SEMANTIC_TYPES),
            "limit": args.limit,
        },
        "counts": dict(Counter(record["status"] for record in audit)),
        "mappings": audit,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["counts"], ensure_ascii=False))
    print(f"Audit written to {output}")


if __name__ == "__main__":
    main()
