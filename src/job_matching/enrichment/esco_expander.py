"""
ESCO Skill Expander - Mo rong CV skills bang ESCO alt_labels.

Khi ung vien viet "lap trinh web", ESCO giup bo sung:
  "web programming, web development, HTML/CSS/JavaScript"
=> BM25 tim duoc nhieu jobs hon, kNN cung co query vector phong phu hon.

Su dung cache embedding tu data/cache/ (da tao boi test_esco_bridge.py)
"""
import os
import csv
import logging
import re
import threading
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = str(PROJECT_ROOT / "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
ESCO_CSV = os.path.join(DATA_DIR, "skills_with_names.csv")

_esco_instance = None
_esco_instance_lock = threading.Lock()


class ESCOExpander:
    """
    Map CV skills -> ESCO concepts -> expanded skills (with alt_labels).
    Uses cached bge-m3 embeddings for fast similarity lookup.
    """

    def __init__(
        self,
        embedding_service=None,
        top_k=1,
        min_sim=0.6,
        max_total_terms=30,
        controlled_min_sim=0.75,
        controlled_min_margin=0.01,
        controlled_max_terms=6,
    ):
        """
        Args:
            embedding_service: EmbeddingService instance (bge-m3)
            top_k: so ESCO concepts lay cho moi CV skill
            min_sim: nguong cosine toi thieu de chap nhan match
            max_total_terms: gioi han tong so terms sau expansion (tranh ES too many clauses)
        """
        self.top_k = top_k
        self.min_sim = min_sim
        self.max_total_terms = max_total_terms
        self.embedding_service = embedding_service
        self.controlled_min_sim = controlled_min_sim
        self.controlled_min_margin = controlled_min_margin
        self.controlled_max_terms = controlled_max_terms

        # Load ESCO database
        self.esco_skills = self._load_esco_csv()
        self.labels = list(self.esco_skills.keys())
        self.label_texts = [self.esco_skills[k]["label"] for k in self.labels]
        self.preferred_label_index = self._build_preferred_label_index()
        self.exact_label_index = self._build_exact_label_index()

        # Load cached embeddings
        self.label_embeddings = self._load_cached_embeddings()

        if self.label_embeddings is not None:
            logger.info(f"ESCO Expander ready: {len(self.labels)} skills, "
                        f"embeddings shape={self.label_embeddings.shape}")
        else:
            logger.warning("ESCO Expander: no cached embeddings, will encode on-the-fly")

    def _load_esco_csv(self):
        """Load ESCO preferred_label + alt_labels tu CSV."""
        skills = {}
        if not os.path.exists(ESCO_CSV):
            logger.warning(f"ESCO CSV not found: {ESCO_CSV}")
            return skills

        with open(ESCO_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = (row.get("preferred_label") or "").strip()
                if not label:
                    continue
                alt = (row.get("alt_labels") or "").strip()
                desc = (row.get("description_en") or "").strip()
                skills[label.lower()] = {
                    "uri": (row.get("uri") or "").strip(),
                    "label": label,
                    "alt_labels": alt,
                    "skill_type": (row.get("skill_type") or "").strip(),
                    "description": desc,
                }
        logger.info(f"Loaded {len(skills)} ESCO skills from CSV")
        return skills

    @staticmethod
    def _normalize_term(term):
        return re.sub(r"\s+", " ", str(term or "")).strip(" .,:;-").lower()

    def _build_exact_label_index(self):
        """Index preferred and alternative labels, retaining ambiguity."""
        index = {}
        for esco_key, info in self.esco_skills.items():
            surfaces = [info["label"]]
            surfaces.extend(
                alt.strip()
                for alt in str(info.get("alt_labels") or "").split("|")
                if alt.strip()
            )
            for surface in surfaces:
                normalized = self._normalize_term(surface)
                if normalized:
                    index.setdefault(normalized, []).append(esco_key)
        return index

    def _build_preferred_label_index(self):
        index = {}
        for esco_key, info in self.esco_skills.items():
            normalized = self._normalize_term(info["label"])
            if normalized:
                index.setdefault(normalized, []).append(esco_key)
        return index

    def _load_cached_embeddings(self):
        """Load cached ESCO label embeddings (tao boi test_esco_bridge.py)."""
        num_skills = len(self.labels)
        emb_cache = os.path.join(CACHE_DIR, f"esco_emb_bge-m3_{num_skills}_label.npy")
        labels_cache = os.path.join(CACHE_DIR, f"esco_labels_{num_skills}.npy")

        if os.path.exists(emb_cache) and os.path.exists(labels_cache):
            # Verify label order matches
            cached_labels = np.load(labels_cache, allow_pickle=True).tolist()
            if cached_labels == self.labels:
                embeddings = np.load(emb_cache)
                logger.info(f"Loaded cached ESCO embeddings: {embeddings.shape}")
                return embeddings
            else:
                logger.warning("ESCO label order mismatch, cache invalid")

        # Fallback: encode on-the-fly if embedding_service available
        if self.embedding_service and self.label_texts:
            logger.info(f"Encoding {len(self.label_texts)} ESCO labels (may take a while)...")
            embeddings = self.embedding_service.encode(
                self.label_texts, batch_size=256, show_progress=True
            )
            # Save cache
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.save(emb_cache, embeddings)
            np.save(labels_cache, np.array(self.labels, dtype=object))
            logger.info(f"Saved ESCO embeddings cache: {emb_cache}")
            return embeddings

        return None

    def expand_skills(self, skills_text):
        """
        Legacy experiment path: expand skills with loose semantic matching.

        Production retrieval uses expand_terms_controlled(). This method is
        retained only so historical ablation scripts remain reproducible.

        Args:
            skills_text: str - comma-separated skills (VD: "Python, quan ly du an, Excel")

        Returns:
            str - expanded skills text (original + ESCO alt_labels)
        """
        if not skills_text or not skills_text.strip():
            return skills_text

        if self.label_embeddings is None or self.embedding_service is None:
            # Khong co embedding -> fallback exact match
            return self._expand_exact_match(skills_text)

        # Parse individual skills
        cv_skills = [s.strip() for s in skills_text.split(",") if s.strip()]
        if not cv_skills:
            return skills_text

        # Encode CV skills
        cv_embeddings = self.embedding_service.encode(cv_skills)

        # Find top ESCO matches cho moi skill
        expanded_terms = []
        seen_terms = set()

        def add_term(term):
            normalized = term.strip()
            key = normalized.lower()
            if normalized and key not in seen_terms:
                expanded_terms.append(normalized)
                seen_terms.add(key)

        for skill in cv_skills:
            add_term(skill)

        esco_matches = []

        for i, skill in enumerate(cv_skills):
            # Cosine similarity voi tat ca ESCO labels
            sims = np.dot(self.label_embeddings, cv_embeddings[i])
            top_indices = np.argsort(sims)[-self.top_k:][::-1]

            for idx in top_indices:
                sim = sims[idx]
                if sim < self.min_sim:
                    break

                esco_key = self.labels[idx]
                esco_info = self.esco_skills[esco_key]

                # Them alt_labels (max 2 per concept de gioi han query size)
                if esco_info["alt_labels"]:
                    alts = [a.strip() for a in esco_info["alt_labels"].split("|")]
                    for alt in alts[:2]:
                        add_term(alt)

                esco_matches.append((skill, esco_info["label"], sim))

                # Hard limit tong so terms
                if len(expanded_terms) >= self.max_total_terms:
                    break
            if len(expanded_terms) >= self.max_total_terms:
                break

        if esco_matches:
            logger.debug(f"ESCO matches: {[(s, e, f'{sim:.3f}') for s, e, sim in esco_matches[:5]]}")

        return ", ".join(expanded_terms[:self.max_total_terms])

    def _expand_exact_match(self, skills_text):
        """Legacy fallback that matches preferred labels only."""
        cv_skills = [s.strip() for s in skills_text.split(",") if s.strip()]
        expanded = []
        seen_terms = set()

        def add_term(term):
            normalized = term.strip()
            key = normalized.lower()
            if normalized and key not in seen_terms:
                expanded.append(normalized)
                seen_terms.add(key)

        for skill in cv_skills:
            add_term(skill)

        for skill in cv_skills:
            skill_lower = skill.lower().strip()
            if skill_lower in self.esco_skills:
                esco_info = self.esco_skills[skill_lower]
                if esco_info["alt_labels"]:
                    alts = [a.strip() for a in esco_info["alt_labels"].split("|")]
                    for alt in alts[:3]:
                        add_term(alt)

        return ", ".join(expanded)

    def expand_skills_exact(self, skills_text):
        """Legacy exact expansion retained for historical scripts."""
        if not skills_text or not skills_text.strip():
            return skills_text
        return self._expand_exact_match(skills_text)

    def _controlled_match(self, skill, embedding=None, term_type=None):
        """Return one high-confidence ESCO match and an audit record."""
        normalized = self._normalize_term(skill)
        # Alternative labels can be broad or context-dependent. Only preferred
        # labels are trusted as exact matches; all other surfaces must pass the
        # semantic similarity and ambiguity checks.
        exact_candidates = self.preferred_label_index.get(normalized, [])
        if len(exact_candidates) == 1:
            esco_key = exact_candidates[0]
            return esco_key, {
                "seed": skill,
                "term_type": term_type or "unknown",
                "status": "accepted",
                "match_method": "exact",
                "esco_label": self.esco_skills[esco_key]["label"],
                "similarity": 1.0,
                "margin": 1.0,
                "reason": "unique exact preferred label",
            }
        if len(exact_candidates) > 1:
            return None, {
                "seed": skill,
                "term_type": term_type or "unknown",
                "status": "rejected",
                "match_method": "exact",
                "similarity": 1.0,
                "margin": 0.0,
                "reason": "ambiguous exact label",
            }

        # ESCO is a skill vocabulary. Broad domains and product/tool names are
        # especially prone to plausible-looking but wrong nearest neighbours.
        semantic_types = {"technical_skill", "professional_skill"}
        if term_type not in semantic_types:
            return None, {
                "seed": skill,
                "term_type": term_type,
                "status": "rejected",
                "match_method": "semantic",
                "similarity": None,
                "margin": None,
                "reason": f"semantic mapping disabled for type={term_type or 'unknown'}",
            }
        if embedding is None or self.label_embeddings is None:
            return None, {
                "seed": skill,
                "term_type": term_type or "unknown",
                "status": "rejected",
                "match_method": "semantic",
                "similarity": None,
                "margin": None,
                "reason": "embedding unavailable",
            }

        sims = np.dot(self.label_embeddings, embedding)
        top_indices = np.argsort(sims)[-2:][::-1]
        best_idx = int(top_indices[0])
        best_sim = float(sims[best_idx])
        second_sim = float(sims[int(top_indices[1])]) if len(top_indices) > 1 else 0.0
        margin = best_sim - second_sim
        esco_key = self.labels[best_idx]
        accepted = (
            best_sim >= self.controlled_min_sim
            and margin >= self.controlled_min_margin
        )
        return (esco_key if accepted else None), {
            "seed": skill,
            "term_type": term_type or "unknown",
            "status": "accepted" if accepted else "rejected",
            "match_method": "semantic",
            "esco_label": self.esco_skills[esco_key]["label"],
            "similarity": round(best_sim, 4),
            "margin": round(margin, 4),
            "reason": (
                "passes similarity and ambiguity thresholds"
                if accepted
                else "below similarity threshold or ambiguous nearest neighbour"
            ),
        }

    def expand_terms_controlled(self, skills, term_types=None, return_audit=False):
        """Return a small set of high-confidence ESCO preferred labels.

        Original terms are deliberately not returned. Callers append these
        labels only to the semantic query, never to the BM25 query.
        """
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",") if s.strip()]
        else:
            skills = [str(s).strip() for s in (skills or []) if str(s).strip()]
        if not skills:
            return ([], []) if return_audit else []

        term_types = {
            self._normalize_term(key): value
            for key, value in (term_types or {}).items()
        }
        embeddings = None
        semantic_indices = [
            idx for idx, skill in enumerate(skills)
            if len(self.preferred_label_index.get(self._normalize_term(skill), [])) != 1
            and term_types.get(self._normalize_term(skill)) in {
                "technical_skill", "professional_skill",
            }
        ]
        if semantic_indices and self.embedding_service and self.label_embeddings is not None:
            embeddings = self.embedding_service.encode(
                [skills[idx] for idx in semantic_indices]
            )
        embedding_by_index = {
            skill_idx: embeddings[pos]
            for pos, skill_idx in enumerate(semantic_indices)
        } if embeddings is not None else {}

        output, seen, audit = [], set(), []
        seed_keys = {self._normalize_term(skill) for skill in skills}
        for idx, skill in enumerate(skills):
            term_type = term_types.get(self._normalize_term(skill))
            esco_key, record = self._controlled_match(
                skill,
                embedding=embedding_by_index.get(idx),
                term_type=term_type,
            )
            audit.append(record)
            if not esco_key:
                continue
            label = self.esco_skills[esco_key]["label"].strip()
            key = self._normalize_term(label)
            if not key or key in seed_keys or key in seen:
                continue
            output.append(label)
            seen.add(key)
            if len(output) >= self.controlled_max_terms:
                break

        return (output, audit) if return_audit else output


def get_esco_expander(embedding_service=None, **kwargs):
    """Singleton getter."""
    global _esco_instance
    with _esco_instance_lock:
        if _esco_instance is None:
            _esco_instance = ESCOExpander(embedding_service=embedding_service, **kwargs)
    return _esco_instance

