
from elasticsearch import Elasticsearch
import logging
import os

logger = logging.getLogger(__name__)


class ElasticHelper:
    def __init__(self, es_host=None, index_name=None):
        self.es_host = es_host or os.environ.get("ES_HOST", "http://localhost:9200")
        self.index_name = index_name or os.environ.get("ES_INDEX", "topcv_jobs")

        try:
            self.es = Elasticsearch(self.es_host)

            if self.es.ping():
                info = self.es.info()
                logger.info(f"Connected to Elasticsearch {info['version']['number']}")
            else:
                logger.error(f"Cannot connect to {es_host}")

        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise

    def _build_filters(self, categories=None, cv_gender=None, exclude_expired=False):
        """
        Tao list ES filter clauses cho bool query.

        Args:
            categories: list[str] - danh muc job can loc (VD: ["IT", "Marketing"])
            cv_gender: str - gioi tinh ung vien ("nam", "nu", "both", None)
            exclude_expired: bool - loai bo job het han
                             (default=False: giu lai tat ca job de test voi data lon)

        Returns:
            list[dict] - ES filter clauses
        """
        filters = []

        # 1. Category filter
        if categories:
            filters.append({"terms": {"category": categories}})

        # 2. Gender filter
        if cv_gender and cv_gender.lower() != "both":
            # Job phai match gioi tinh ung vien HOAC khong yeu cau gioi tinh
            gender_lower = cv_gender.lower()
            filters.append({
                "bool": {
                    "should": [
                        {"term": {"gender_requirement": gender_lower}},
                        {"term": {"gender_requirement": "cả hai"}},
                        {"term": {"gender_requirement": "không yêu cầu"}},
                        {"term": {"gender_requirement": ""}},
                        # Job khong co truong gender_requirement
                        {"bool": {"must_not": {"exists": {"field": "gender_requirement"}}}},
                    ],
                    "minimum_should_match": 1,
                }
            })

        # 3. Expired filter
        # NOTE: Tam thoi tat de test voi full dataset (bao gom ca job het han)
        # Khi deploy production, doi lai exclude_expired=True
        if exclude_expired:
            filters.append({
                "bool": {
                    "should": [
                        {"term": {"is_expired": False}},
                        {"bool": {"must_not": {"exists": {"field": "is_expired"}}}},
                    ],
                    "minimum_should_match": 1,
                }
            })

        return filters

    def search_jobs_by_profile(self, profile_text, size=100,
                                categories=None, cv_gender=None, exclude_expired=True):
        """BM25-only search voi ES-level filters (fallback khi khong co embedding)."""
        if not profile_text or not profile_text.strip():
            return [], 0

        filters = self._build_filters(categories, cv_gender, exclude_expired)
        bm25_hits = self._search_bm25(profile_text, size=size, filters=filters)

        jobs = []
        for hit in bm25_hits:
            job = hit["_source"]
            job["_score"] = hit["_score"]
            job["_id"] = hit["_id"]
            jobs.append(job)

        return jobs, len(jobs)

    def _search_bm25(self, profile_text, size=100, filters=None):
        """BM25 search, tra ve list hits (co ho tro filters)"""
        if not profile_text or not profile_text.strip():
            return []

        # Gioi han query text de tranh "too many clauses" loi ES
        words = profile_text.split()
        if len(words) > 60:
            profile_text = " ".join(words[:60])

        multi_match_query = {
            "multi_match": {
                "query": profile_text,
                "fields": [
                    "requirements_tags^5.0",
                    "specializations^4.0",
                    "title^3.0",
                    "technical_skills^3.0",
                    "certificates^2.5",
                    "languages^2.0",
                    "soft_skills^1.0",
                    "job_requirements^0.5",
                    "job_description^0.3",
                ],
                "type": "best_fields",
                "operator": "or",
                "minimum_should_match": "30%",
            }
        }

        if filters:
            search_body = {
                "query": {
                    "bool": {
                        "must": [multi_match_query],
                        "filter": filters,
                    }
                },
                "size": size,
            }
        else:
            search_body = {
                "query": multi_match_query,
                "size": size,
            }

        try:
            result = self.es.search(index=self.index_name, body=search_body)
            return result["hits"]["hits"]
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []

    def _search_knn(self, query_vector, size=100, filters=None, num_candidates=None):
        """kNN search tren dense_vector field (co ho tro filters)"""
        try:
            if num_candidates is None:
                num_candidates = size * 2
            num_candidates = max(num_candidates, size)

            knn_body = {
                "field": "embedding",
                "query_vector": query_vector,
                "k": size,
                "num_candidates": num_candidates,
            }

            # kNN co the nhan filter rieng de pre-filter truoc khi tim kNN
            if filters:
                knn_body["filter"] = {"bool": {"filter": filters}}

            result = self.es.search(
                index=self.index_name,
                knn=knn_body,
                size=size,
            )
            return result["hits"]["hits"]
        except Exception as e:
            logger.error(f"kNN search error: {e}")
            return []

    def _rrf_fusion(self, bm25_hits, knn_hits, k=60, bm25_weight=1.0, knn_weight=1.0):
        """
        Reciprocal Rank Fusion (client-side).
        RRF score = weighted sum(1 / (k + rank)) cho moi result set.

        Args:
            bm25_hits: list hits tu BM25
            knn_hits: list hits tu kNN
            k: hang so RRF (mac dinh 60)
            bm25_weight: trong so dong gop cua BM25
            knn_weight: trong so dong gop cua kNN

        Returns:
            list[dict] - da merge va sap xep theo RRF score
        """
        doc_data = {}

        for rank, hit in enumerate(bm25_hits):
            doc_id = hit["_id"]
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "doc": hit["_source"],
                    "_id": doc_id,
                    "bm25_score": hit.get("_score", 0),
                    "knn_score": 0,
                    "rrf_score": 0,
                    "bm25_rank": rank + 1,
                    "knn_rank": None,
                }
            doc_data[doc_id]["rrf_score"] += bm25_weight * (1.0 / (k + rank + 1))

        for rank, hit in enumerate(knn_hits):
            doc_id = hit["_id"]
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "doc": hit["_source"],
                    "_id": doc_id,
                    "bm25_score": 0,
                    "knn_score": hit.get("_score", 0),
                    "rrf_score": 0,
                    "bm25_rank": None,
                    "knn_rank": rank + 1,
                }
            else:
                doc_data[doc_id]["knn_score"] = hit.get("_score", 0)
                doc_data[doc_id]["knn_rank"] = rank + 1
            doc_data[doc_id]["rrf_score"] += knn_weight * (1.0 / (k + rank + 1))

        sorted_docs = sorted(doc_data.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_docs

    def search_jobs_hybrid(self, profile_text, query_vector, size=100,
                            categories=None, cv_gender=None, exclude_expired=True,
                            rrf_k=60, bm25_weight=1.0, knn_weight=1.0,
                            num_candidates=None):
        """
        Hybrid search: BM25 + kNN + RRF fusion voi ES-level filters.

        Args:
            profile_text: text de BM25 search
            query_vector: embedding vector de kNN search
            size: so ket qua toi da
            categories: list[str] - loc theo danh muc
            cv_gender: str - gioi tinh ung vien
            exclude_expired: bool - loai bo job het han
            rrf_k: hang so RRF; gia tri nho uu tien khac biet rank dau hon
            bm25_weight: trong so BM25 trong RRF
            knn_weight: trong so kNN trong RRF
            num_candidates: so ung vien HNSW kNN; mac dinh size * 2

        Returns:
            (jobs, total) - tuong tu search_jobs_by_profile
        """
        filters = self._build_filters(categories, cv_gender, exclude_expired)

        bm25_hits = self._search_bm25(profile_text, size=size, filters=filters)
        knn_hits = self._search_knn(
            query_vector, size=size, filters=filters,
            num_candidates=num_candidates,
        )

        merged = self._rrf_fusion(
            bm25_hits, knn_hits, k=rrf_k,
            bm25_weight=bm25_weight, knn_weight=knn_weight,
        )

        jobs = []
        for item in merged[:size]:
            job = item["doc"]
            job["_id"] = item["_id"]
            job["_score"] = item.get("bm25_score", 0)
            job["_rrf_score"] = item["rrf_score"]
            job["_bm25_rank"] = item.get("bm25_rank")
            job["_knn_rank"] = item.get("knn_rank")
            job.pop("embedding", None)
            jobs.append(job)

        return jobs, len(merged)

    def has_embedding_field(self):
        """Kiem tra index co dense_vector field 'embedding' khong"""
        try:
            mapping = self.es.indices.get_mapping(index=self.index_name)
            props = mapping[self.index_name]["mappings"].get("properties", {})
            return "embedding" in props
        except Exception:
            return False

    def get_job_by_id(self, job_id):
        try:
            result = self.es.get(index=self.index_name, id=job_id)
            return result["_source"]
        except Exception as e:
            logger.error(f"Error getting job: {e}")
            return None

    def count_jobs(self):
        try:
            result = self.es.count(index=self.index_name)
            return result["count"]
        except Exception as e:
            logger.error(f"Error counting jobs: {e}")
            return 0

    def delete_index(self):
        try:
            self.es.indices.delete(index=self.index_name, ignore=[400, 404])
            return True
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            return False

    def get_index_info(self):
        try:
            count = self.es.count(index=self.index_name)
            stats = self.es.indices.stats(index=self.index_name)

            info = {
                "total_documents": count["count"],
                "size_mb": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"] / 1024 / 1024,
                "index_name": self.index_name,
                "es_host": self.es_host,
                "has_embeddings": self.has_embedding_field(),
            }
            return info
        except Exception as e:
            logger.error(f"Error getting info: {e}")
            return None
