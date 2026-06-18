"""Backfill canonical semantic profile and embedding for existing ES jobs."""

from __future__ import annotations

import argparse
import math
import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan

from job_matching.enrichment.semantic_job_profile import SemanticJobProfileBuilder
from job_matching.retrieval.embedding_service import get_embedding_service


SOURCE_FIELDS = [
    "title", "specializations", "requirements_tags",
    "technical_skills", "certificates", "languages",
    "job_description", "job_requirements", "url",
]


def ensure_mapping(es, index):
    es.indices.put_mapping(index=index, properties={
        "semantic_text": {"type": "text"},
        "semantic_title": {"type": "text"},
        "embedding": {
            "type": "dense_vector",
            "dims": 1024,
            "index": True,
            "similarity": "cosine",
        },
    })


def nonzero(vector):
    return vector if math.sqrt(sum(x * x for x in vector)) >= 1e-8 else [1e-6] * len(vector)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--es-host", default=os.getenv("ES_HOST", "http://localhost:9200"))
    parser.add_argument("--index", default=os.getenv("ES_INDEX", "topcv_jobs_production"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    es = Elasticsearch(args.es_host, request_timeout=180)
    ensure_mapping(es, args.index)
    builder = SemanticJobProfileBuilder()
    embed = get_embedding_service(device="cpu")
    query = {"match_all": {}} if args.overwrite else {
        "bool": {
            "should": [
                {"bool": {"must_not": [{"exists": {"field": "semantic_text"}}]}},
                {"bool": {"must_not": [{"exists": {"field": "embedding"}}]}},
            ],
            "minimum_should_match": 1,
        }
    }
    total = es.count(index=args.index)["count"]
    target = es.count(index=args.index, query=query)["count"]
    print(f"total={total} target={target} overwrite={args.overwrite}")

    pending = []
    updated = 0

    def flush():
        nonlocal pending, updated
        profiles = [builder.build(hit["_source"]) for hit in pending]
        vectors = embed.encode(
            [profile["semantic_text"] for profile in profiles],
            batch_size=args.encode_batch_size,
            show_progress=False,
        )
        actions = []
        for hit, profile, vector in zip(pending, profiles, vectors):
            profile["embedding"] = nonzero(vector.tolist())
            actions.append({
                "_op_type": "update",
                "_index": args.index,
                "_id": hit["_id"],
                "script": {
                    "lang": "painless",
                    "source": (
                        "for (entry in params.profile.entrySet()) { "
                        "ctx._source[entry.getKey()] = entry.getValue(); }"
                    ),
                    "params": {"profile": profile},
                },
            })
        success, errors = bulk(es, actions, raise_on_error=False, refresh=False)
        if errors:
            raise RuntimeError(f"bulk errors: {len(errors)}")
        updated += success
        print(f"updated={updated}/{target}")
        pending = []

    for hit in scan(
        es,
        index=args.index,
        query={"query": query, "_source": SOURCE_FIELDS},
        size=max(args.batch_size, 50),
        request_timeout=180,
    ):
        pending.append(hit)
        if len(pending) >= args.batch_size:
            flush()
    if pending:
        flush()

    es.indices.refresh(index=args.index)
    embedded = es.count(index=args.index, query={"exists": {"field": "embedding"}})["count"]
    semantic = es.count(index=args.index, query={"exists": {"field": "semantic_text"}})["count"]
    print({"total": total, "updated": updated, "embedding": embedded, "semantic_text": semantic})


if __name__ == "__main__":
    main()
