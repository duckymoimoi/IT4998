"""
Import TopCV jobs data to Elasticsearch (LOCAL)
Ho tro dense_vector (embedding) cho hybrid search BM25 + kNN
"""

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import logging
import time
import os
import requests

from job_matching.scoring.salary_normalizer import SalaryNormalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ElasticImporter:
    def __init__(self, es_host="http://localhost:9200", use_embeddings=True):
        """
        Args:
            es_host: Elasticsearch host URL
            use_embeddings: True = tao embedding cho moi job (can bge-m3)
        """
        self.es_host = es_host
        self.es = Elasticsearch(es_host)
        self.salary_normalizer = SalaryNormalizer()
        self.use_embeddings = use_embeddings
        self.embedding_service = None

        logger.info(f"Connecting to Elasticsearch: {es_host}")

        if use_embeddings:
            try:
                from job_matching.retrieval.embedding_service import get_embedding_service
                self.embedding_service = get_embedding_service()
                logger.info("Embedding service (bge-m3) da san sang")
            except Exception as e:
                logger.warning(f"Khong the tai embedding service: {e}")
                logger.warning("Se import KHONG co embeddings")
                self.use_embeddings = False

    def check_connection(self):
        try:
            if self.es.ping():
                info = self.es.info()
                logger.info(f"Elasticsearch is running")
                logger.info(f"   Version: {info['version']['number']}")
                logger.info(f"   Cluster: {info['cluster_name']}")
                return True
            else:
                logger.error("Cannot connect to Elasticsearch")
                return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    def create_index(self, index_name="topcv_jobs", force_recreate=True):
        """Tao index voi mapping bao gom dense_vector cho kNN search"""
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "job_text_analyzer": {
                            "type": "standard",
                            "stopwords": "_none_",
                        }
                    }
                },
            },
            "mappings": {
                "properties": {
                    # --- Text fields (BM25 search) ---
                    "title": {"type": "text", "analyzer": "job_text_analyzer"},
                    "url": {"type": "keyword"},
                    "company": {"type": "text"},
                    "company_address": {"type": "text", "analyzer": "job_text_analyzer"},
                    "company_size": {"type": "keyword"},
                    # Legacy/experimental fields; production no longer populates them.
                    "company_field": {"type": "keyword"},
                    "category": {"type": "keyword"},
                    "deadline": {"type": "text"},
                    "crawled_date": {"type": "text"},

                    # --- BM25 search fields (voi boost) ---
                    "requirements_tags": {"type": "text", "analyzer": "job_text_analyzer"},
                    "specializations": {"type": "text", "analyzer": "job_text_analyzer"},
                    "technical_skills": {"type": "text", "analyzer": "job_text_analyzer"},
                    # Legacy compatibility only; new jobs no longer populate this field.
                    "soft_skills": {"type": "text", "analyzer": "job_text_analyzer"},
                    "languages": {"type": "text", "analyzer": "job_text_analyzer"},
                    "certificates": {"type": "text", "analyzer": "job_text_analyzer"},

                    # --- Job detail fields ---
                    "job_description": {"type": "text", "analyzer": "job_text_analyzer"},
                    "job_requirements": {"type": "text", "analyzer": "job_text_analyzer"},
                    "job_benefits": {"type": "text", "analyzer": "job_text_analyzer"},
                    "job_location": {"type": "text", "analyzer": "job_text_analyzer"},
                    "work_location_text": {"type": "text", "analyzer": "job_text_analyzer"},
                    "working_time": {"type": "text"},

                    # --- Salary ---
                    "job_salary": {"type": "text"},
                    "salary_min": {"type": "float"},
                    "salary_max": {"type": "float"},
                    "salary_type": {"type": "keyword"},
                    "salary_note": {"type": "text"},
                    "has_commission": {"type": "boolean"},

                    # --- Filters ---
                    "gender_requirement": {"type": "keyword"},
                    "experience": {"type": "text"},
                    "education_level": {"type": "keyword"},
                    "education_field": {"type": "text"},

                    # --- Tracking ---
                    "content_hash": {"type": "keyword"},
                    "is_expired": {"type": "boolean"},

                    # Legacy compatibility: web can reuse coordinates on existing jobs.
                    "geo_coordinates": {
                        "type": "nested",
                        "properties": {
                            "lat": {"type": "float"},
                            "lng": {"type": "float"},
                            "address": {"type": "text"},
                        }
                    },

                    # --- Dense vector cho kNN search ---
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "semantic_text": {"type": "text", "analyzer": "job_text_analyzer"},
                    "semantic_title": {"type": "text", "analyzer": "job_text_analyzer"},
                }
            },
        }

        if not self.use_embeddings:
            del mapping["mappings"]["properties"]["embedding"]

        try:
            index_exists = self.es.indices.exists(index=index_name)

            if index_exists:
                logger.warning(f"Index '{index_name}' already exists")

                if force_recreate:
                    logger.info("Deleting existing index...")
                    self.es.indices.delete(index=index_name)
                    logger.info("Deleted old index")
                else:
                    response = input("Delete and recreate? (y/n): ")
                    if response.lower() == "y":
                        self.es.indices.delete(index=index_name)
                    else:
                        logger.warning("Keeping existing index")
                        return False

            self.es.indices.create(index=index_name, body=mapping)
            embed_status = "CO embedding (kNN)" if self.use_embeddings else "KHONG co embedding"
            logger.info(f"Created fresh index: {index_name} ({embed_status})")
            return True

        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False

    def prepare_document(self, row):
        """Chuan bi document voi salary normalization"""
        doc = {}

        include_fields = [
            "title", "url", "company", "company_address",
            "company_size",
            "deadline", "crawled_date",
            "requirements_tags", "specializations",
            "technical_skills", "languages", "certificates",
            "job_description", "job_requirements", "job_benefits",
            "job_location", "work_location_text", "working_time", "job_salary",
            "gender_requirement", "experience",
            "education_level", "education_field",
            "salary_note", "content_hash",
        ]

        for field in include_fields:
            value = row.get(field, "")
            if pd.isna(value) or value is None:
                doc[field] = ""
            else:
                doc[field] = str(value).strip()

        # Boolean fields
        is_expired = row.get("is_expired", False)
        if isinstance(is_expired, str):
            is_expired = is_expired.lower() in ("true", "1", "yes")
        doc["is_expired"] = bool(is_expired) if not pd.isna(is_expired) else False

        has_commission = row.get("has_commission", False)
        if isinstance(has_commission, str):
            has_commission = has_commission.lower() in ("true", "1", "yes")
        doc["has_commission"] = bool(has_commission) if not pd.isna(has_commission) else False

        # Salary normalization
        salary_text = row.get("job_salary", "")
        salary_data = self.salary_normalizer.normalize_salary(salary_text)

        if salary_data['min'] is not None:
            doc["salary_min"] = salary_data['min'] / 1_000_000
        if salary_data['max'] is not None:
            doc["salary_max"] = salary_data['max'] / 1_000_000
        doc["salary_type"] = salary_data['type']

        # Su dung salary_min/max tu CSV neu da co (tu LLM cleaner)
        csv_sal_min = row.get("salary_min")
        csv_sal_max = row.get("salary_max")
        csv_sal_type = row.get("salary_type")

        if csv_sal_min and not pd.isna(csv_sal_min):
            try:
                val = float(csv_sal_min)
                if val > 1000:
                    doc["salary_min"] = val / 1_000_000
                else:
                    doc["salary_min"] = val
            except (ValueError, TypeError):
                pass

        if csv_sal_max and not pd.isna(csv_sal_max):
            try:
                val = float(csv_sal_max)
                if val > 1000:
                    doc["salary_max"] = val / 1_000_000
                else:
                    doc["salary_max"] = val
            except (ValueError, TypeError):
                pass

        if csv_sal_type and not pd.isna(csv_sal_type):
            doc["salary_type"] = str(csv_sal_type).strip()

        # Sanitize: ES rejects NaN in JSON - convert to None/empty
        import math
        for key, val in list(doc.items()):
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                doc[key] = None
            elif isinstance(val, str) and val.lower() == 'nan':
                doc[key] = ""

        return doc

    def bulk_index_data(self, csv_file, index_name="topcv_jobs", batch_size=100):
        """Bulk index data tu CSV, bao gom tao embedding neu duoc bat"""
        try:
            logger.info(f"Reading data from: {csv_file}")
            df = pd.read_csv(csv_file)

            logger.info(f"Total records: {len(df):,}")
            logger.info(f"Columns: {list(df.columns)}")

            if self.use_embeddings and self.embedding_service:
                logger.info("Dang tao embeddings cho tat ca jobs...")
                embed_start = time.time()
                from job_matching.enrichment.semantic_job_profile import SemanticJobProfileBuilder
                semantic_builder = SemanticJobProfileBuilder()

                semantic_profiles = []
                semantic_texts = []
                for _, row in df.iterrows():
                    job = row.to_dict()
                    profile = semantic_builder.build(job)
                    semantic_profiles.append(profile)
                    semantic_texts.append(profile["semantic_text"])

                all_embeddings = self.embedding_service.encode(
                    semantic_texts,
                    batch_size=32,
                    show_progress=True,
                )
                embed_time = time.time() - embed_start
                logger.info(f"Tao {len(all_embeddings)} embeddings trong {embed_time:.1f}s "
                            f"({embed_time/len(all_embeddings):.2f}s/job)")
            else:
                all_embeddings = None
                semantic_profiles = None

            actions = []
            success_count = 0
            error_count = 0
            salary_types = {}

            for i, (idx, row) in enumerate(df.iterrows()):
                doc = self.prepare_document(row)

                if all_embeddings is not None:
                    emb = all_embeddings[i].tolist()
                    # ES cosine similarity rejects zero-magnitude vectors
                    import math
                    magnitude = math.sqrt(sum(x*x for x in emb))
                    if magnitude < 1e-8:
                        # Jobs thiếu data → embedding gần zero → thay bằng tiny uniform vector
                        emb = [1e-6] * len(emb)
                    doc.update(semantic_profiles[i])
                    doc["embedding"] = emb

                sal_type = doc.get('salary_type', 'unknown')
                salary_types[sal_type] = salary_types.get(sal_type, 0) + 1

                # Dùng content_hash làm _id (unique per job)
                doc_id = doc.get('content_hash', str(i))
                action = {
                    "_index": index_name,
                    "_id": doc_id,
                    "_source": doc,
                }
                actions.append(action)

                if len(actions) >= batch_size:
                    success, errors = bulk(self.es, actions, raise_on_error=False)
                    success_count += success
                    if errors:
                        error_count += len(errors)
                        logger.warning(f"Errors in batch: {len(errors)}")
                        # Log first error batch for debugging
                        if error_count - len(errors) == 0:  # First batch with errors
                            for err in errors[:3]:
                                logger.error(f"  Bulk error detail: {err}")

                    logger.info(f"Indexed {success_count:,}/{len(df):,} documents")
                    actions = []

            if actions:
                success, errors = bulk(self.es, actions, raise_on_error=False)
                success_count += success
                if errors:
                    error_count += len(errors)

            logger.info(f"COMPLETED: Indexed {success_count:,} documents, {error_count} errors")

            if salary_types:
                logger.info("Salary Type Distribution:")
                for sal_type, count in sorted(salary_types.items(), key=lambda x: -x[1]):
                    if count > 0:
                        pct = count / len(df) * 100
                        logger.info(f"   {sal_type:15s}: {count:6,} ({pct:5.1f}%)")

            self.es.indices.refresh(index=index_name)

            return success_count, error_count

        except Exception as e:
            logger.error(f"Error during bulk indexing: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Import jobs to Elasticsearch")
    parser.add_argument("--csv", default="topcv_jobs.csv", help="CSV file to import")
    parser.add_argument("--index", default="topcv_jobs", help="Elasticsearch index name")
    parser.add_argument("--no-embedding", action="store_true", help="Import without embeddings")
    parser.add_argument("--es-host", default="http://localhost:9200", help="Elasticsearch host")
    args = parser.parse_args()

    importer = ElasticImporter(
        es_host=args.es_host,
        use_embeddings=not args.no_embedding,
    )

    if not importer.check_connection():
        logger.error("Cannot proceed without Elasticsearch connection")
        return

    if not importer.create_index(args.index, force_recreate=True):
        logger.error("Failed to create index")
        return

    success_count, error_count = importer.bulk_index_data(
        csv_file=args.csv,
        index_name=args.index,
        batch_size=100,
    )


if __name__ == "__main__":
    main()
