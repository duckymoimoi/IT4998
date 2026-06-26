"""Derive structured job profile fields from ESCO/O*NET files.

This module is intentionally data-driven: it does not hard-code role families
such as "AI/Data" or "Backend". It maps job text to ESCO skill concepts, then
uses the ESCO occupation-skill graph and O*NET crosswalk already present in the
project to infer occupations and related tools.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from job_matching.enrichment.skill_graph import get_skill_graph


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
SKILLS_WITH_NAMES = DATA_DIR / "skills_with_names.csv"


TECH_ALIASES = {

    "react.js": "React",
    "reactjs": "React",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "fast api": "FastAPI",
    "fastapi": "FastAPI",
    "scikit learn": "Scikit-learn",
    "scikit-learn": "Scikit-learn",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "xgboost": "XGBoost",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "spark": "Apache Spark",
    "hadoop": "Apache Hadoop",
    "nlp": "natural language processing",
}


STOP_TERMS = {
    "", "it", "ai", "ml", "dl", "b2b", "b2c", "nam", "nu", "nữ",
    "english", "vietnamese", "junior", "senior", "middle",
    "hoặc tương đương", "khác", "other", "others", "cơ bản",
}


class JobProfileEnricher:
    """Infer normalized job fields through ESCO skills and occupations."""

    def __init__(
        self,
        embedding_service=None,
        use_semantic: bool = False,
        min_semantic_sim: float = 0.70,
        source_fields: Optional[List[str]] = None,
        fallback_to_title: bool = False,
    ):
        self.graph = get_skill_graph()
        self.alias_to_uri = self._build_alias_index()
        self.embedding_service = embedding_service
        self.use_semantic = use_semantic
        self.min_semantic_sim = min_semantic_sim
        # In the production crawl pipeline, TopCV's "Chuyen mon" tags are
        # passed through the cleaner and stored in technical_skills.
        self.source_fields = source_fields or ["technical_skills"]
        self.fallback_to_title = fallback_to_title
        self._esco_expander = None
        if self.use_semantic:
            if self.embedding_service is None:
                from job_matching.retrieval.embedding_service import get_embedding_service
                self.embedding_service = get_embedding_service()
            from job_matching.enrichment.esco_expander import ESCOExpander
            self._esco_expander = ESCOExpander(embedding_service=self.embedding_service)

    def _build_alias_index(self) -> Dict[str, str]:
        """Build lowercase label/alt-label -> ESCO skill URI index."""
        aliases: Dict[str, str] = {}
        if not SKILLS_WITH_NAMES.exists():
            return aliases

        with SKILLS_WITH_NAMES.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uri = (row.get("uri") or "").strip()
                preferred = (row.get("preferred_label") or "").strip()
                if not uri or not preferred:
                    continue
                aliases.setdefault(preferred.lower(), uri)
                for alt in re.split(r"\s*\|\s*", row.get("alt_labels") or ""):
                    alt = alt.strip()
                    if alt:
                        aliases.setdefault(alt.lower(), uri)
        return aliases

    def _split_terms(self, text: str) -> List[str]:
        if not text:
            return []
        raw_terms = re.split(r"[,;/\n|]+", str(text))
        terms = []
        seen = set()
        for term in raw_terms:
            cleaned = re.sub(r"\s+", " ", term).strip(" .:-()[]{}")
            if len(cleaned) < 2:
                continue
            key = cleaned.lower()
            if key in STOP_TERMS or key in seen:
                continue
            terms.append(cleaned)
            seen.add(key)
        return terms

    def _candidate_terms(self, job: dict) -> List[str]:
        parts = [job.get(field, "") for field in self.source_fields]
        # The job title can be useful in experiments, but it is noisy enough to
        # infer wrong occupations in production. Keep the default strict and
        # return no enrichment when compact tag fields are missing.
        if self.fallback_to_title and not any(str(part or "").strip() for part in parts):
            parts.append(job.get("title", ""))

        terms = []
        seen = set()
        for part in parts:
            for term in self._split_terms(part):
                key = term.lower()
                if key not in seen:
                    terms.append(term)
                    seen.add(key)
        return terms

    def _find_skill_uri(self, term: str) -> Tuple[Optional[str], str, Optional[float]]:
        key = term.lower().strip()
        if not key or key in STOP_TERMS:
            return None, "none", None

        alias = TECH_ALIASES.get(key)
        if alias:
            key = alias.lower()

        # 1. Exact ESCO preferred/alt label from CSV.
        uri = self.alias_to_uri.get(key)
        if uri:
            return uri, "exact_alias", 1.0

        # 2. Exact graph title lookup, including SkillGraph suffix handling.
        uri = self.graph.find_uri(term)
        if uri and self.graph.nodes.get(uri, {}).get("type") == "SKILL":
            return uri, "exact_graph", 1.0

        if alias:
            uri = self.graph.find_uri(alias)
            if uri and self.graph.nodes.get(uri, {}).get("type") == "SKILL":
                return uri, "tech_alias", 1.0

        # 3. Semantic ESCO label matching through cached bge-m3 embeddings.
        if self._esco_expander and self._esco_expander.label_embeddings is not None:
            import numpy as np

            query = alias or term
            emb = self.embedding_service.encode([query])[0]
            sims = np.dot(self._esco_expander.label_embeddings, emb)
            idx = int(np.argmax(sims))
            sim = float(sims[idx])
            if sim >= self.min_semantic_sim:
                label_key = self._esco_expander.labels[idx]
                uri = self.alias_to_uri.get(label_key)
                if uri:
                    return uri, "semantic_esco", sim

        return None, "none", None

    def _score_occupations(self, skill_uris: Iterable[str]) -> Counter:
        votes: Counter = Counter()
        for skill_uri in skill_uris:
            for occ_uri in self.graph.skill_to_occs.get(skill_uri, set()):
                weight = 2.0 if skill_uri in self.graph.occ_essential.get(occ_uri, set()) else 1.0
                votes[occ_uri] += weight
        return votes

    def enrich(self, job: dict, max_skills: int = 12, max_occupations: int = 5, max_tools: int = 12) -> dict:
        """Return derived fields for a single job document."""
        terms = self._candidate_terms(job)

        matched = []
        seen_uris = set()
        for term in terms:
            uri, method, score = self._find_skill_uri(term)
            if not uri or uri in seen_uris:
                continue
            node = self.graph.nodes.get(uri, {})
            matched.append({
                "input": term,
                "esco_uri": uri,
                "label": node.get("title", term),
                "match_method": method,
                "match_score": round(score, 3) if score is not None else None,
            })
            seen_uris.add(uri)
            if len(matched) >= max_skills:
                break

        skill_uris = [m["esco_uri"] for m in matched]
        occ_votes = self._score_occupations(skill_uris)

        occupations = []
        for occ_uri, score in occ_votes.most_common(max_occupations):
            node = self.graph.nodes.get(occ_uri, {})
            occupations.append({
                "esco_uri": occ_uri,
                "label": node.get("title", ""),
                "score": round(float(score), 2),
                "onet_code": self.graph.esco_to_onet.get(occ_uri, ""),
            })

        # O*NET tools are derived from inferred occupations, not injected from a manual list.
        tools = []
        seen_tools = set()
        for occ in occupations:
            for tool_uri in self.graph.occ_tools.get(occ["esco_uri"], set()):
                if tool_uri in seen_tools:
                    continue
                node = self.graph.nodes.get(tool_uri, {})
                title = node.get("title", "")
                if title:
                    tools.append(title)
                    seen_tools.add(tool_uri)
                if len(tools) >= max_tools:
                    break
            if len(tools) >= max_tools:
                break

        confidence = 0.0
        if matched and occupations:
            top_score = occupations[0]["score"]
            confidence = min(1.0, top_score / max(3.0, len(matched) * 1.5))

        return {
            "normalized_skills": [
                {
                    "label": m["label"],
                    "esco_uri": m["esco_uri"],
                    "source_text": m["input"],
                    "match_method": m.get("match_method"),
                    "match_score": m.get("match_score"),
                }
                for m in matched
            ],
            "normalized_skill_labels": [m["label"] for m in matched],
            "normalized_skill_uris": skill_uris,
            "inferred_occupations": occupations,
            "inferred_occupation_labels": [o["label"] for o in occupations],
            "inferred_occupation_uris": [o["esco_uri"] for o in occupations],
            "inferred_onet_codes": [o["onet_code"] for o in occupations if o.get("onet_code")],
            "onet_tools_from_occupations": tools,
            "profile_enrichment_confidence": round(confidence, 2),
        }


def enrich_job_profile(job: dict) -> dict:
    return JobProfileEnricher().enrich(job)
