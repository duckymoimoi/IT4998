"""Backfill deterministic TopCV overview fields for existing ES documents."""

from __future__ import annotations

import argparse
import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan

from job_matching.scoring.llm_cleaner import LLMCleaner


def separate_specializations_from_skills(technical_skills, specializations):
    specialization_keys = {
        value.strip().lower()
        for value in str(specializations or "").split(",")
        if value.strip()
    }
    skills = [
        value.strip()
        for value in str(technical_skills or "").split(",")
        if value.strip()
    ]
    return ", ".join(
        value for value in skills
        if value.lower() not in specialization_keys
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--es-host", default=os.getenv("ES_HOST", "http://localhost:9200"))
    parser.add_argument("--index", default=os.getenv("ES_INDEX", "topcv_jobs_production"))
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    es = Elasticsearch(args.es_host, request_timeout=180)
    cleaner = LLMCleaner()
    query = {"exists": {"field": "overview"}}
    target = es.count(index=args.index, query=query)["count"]
    print(f"index={args.index} target={target} dry_run={args.dry_run}")

    actions = []
    scanned = updated = with_specializations = 0

    def flush():
        nonlocal actions, updated
        if not actions or args.dry_run:
            actions = []
            return
        success, errors = bulk(es, actions, raise_on_error=False, refresh=False)
        if errors:
            raise RuntimeError(f"bulk errors: {len(errors)}")
        updated += success
        print(f"updated={updated}/{target}")
        actions = []

    for hit in scan(
        es,
        index=args.index,
        query={
            "query": query,
            "_source": [
                "overview", "requirements_tags", "specializations",
                "technical_skills",
            ],
        },
        size=max(args.batch_size, 100),
        request_timeout=180,
    ):
        scanned += 1
        source = hit["_source"]
        extracted = cleaner._trich_xuat_regex({
            "overview": hit["_source"].get("overview", ""),
            "requirements_tags": source.get("requirements_tags", ""),
            "specializations": source.get("specializations", ""),
            "job_details": "",
        })
        specializations = extracted.get("specializations", "")
        requirements_tags = extracted.get("requirements_tags", "")
        if specializations:
            with_specializations += 1
        update_doc = {}
        if specializations:
            update_doc["specializations"] = specializations
            existing_skills = str(source.get("technical_skills", "") or "")
            separated_skills = separate_specializations_from_skills(
                existing_skills, specializations,
            )
            if separated_skills != existing_skills:
                update_doc["technical_skills"] = separated_skills
        if requirements_tags:
            update_doc["requirements_tags"] = requirements_tags
        if not update_doc:
            continue
        actions.append({
            "_op_type": "update",
            "_index": args.index,
            "_id": hit["_id"],
            "doc": update_doc,
        })
        if len(actions) >= args.batch_size:
            flush()
    flush()

    if not args.dry_run:
        es.indices.refresh(index=args.index)
    print({
        "scanned": scanned,
        "updated": updated,
        "with_specializations": with_specializations,
        "without_specializations": scanned - with_specializations,
    })


if __name__ == "__main__":
    main()
