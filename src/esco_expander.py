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
import numpy as np

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
ESCO_CSV = os.path.join(DATA_DIR, "skills_with_names.csv")

_esco_instance = None


class ESCOExpander:
    """
    Map CV skills -> ESCO concepts -> expanded skills (with alt_labels).
    Uses cached bge-m3 embeddings for fast similarity lookup.
    """

    def __init__(self, embedding_service=None, top_k=1, min_sim=0.6, max_total_terms=30):
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

        # Load ESCO database
        self.esco_skills = self._load_esco_csv()
        self.labels = list(self.esco_skills.keys())
        self.label_texts = [self.esco_skills[k]["label"] for k in self.labels]

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
                    "label": label,
                    "alt_labels": alt,
                    "description": desc,
                }
        logger.info(f"Loaded {len(skills)} ESCO skills from CSV")
        return skills

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
        Mo rong skills text bang ESCO alt_labels.

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
        expanded_terms = set(cv_skills)  # Giu nguyen skills goc
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
                        expanded_terms.add(alt)

                esco_matches.append((skill, esco_info["label"], sim))

                # Hard limit tong so terms
                if len(expanded_terms) >= self.max_total_terms:
                    break
            if len(expanded_terms) >= self.max_total_terms:
                break

        if esco_matches:
            logger.debug(f"ESCO matches: {[(s, e, f'{sim:.3f}') for s, e, sim in esco_matches[:5]]}")

        # Cap at max_total_terms
        result = list(expanded_terms)[:self.max_total_terms]
        return ", ".join(result)

    def _expand_exact_match(self, skills_text):
        """Fallback: exact string match voi ESCO labels."""
        cv_skills = [s.strip() for s in skills_text.split(",") if s.strip()]
        expanded = set(cv_skills)

        for skill in cv_skills:
            skill_lower = skill.lower().strip()
            if skill_lower in self.esco_skills:
                esco_info = self.esco_skills[skill_lower]
                if esco_info["alt_labels"]:
                    alts = [a.strip() for a in esco_info["alt_labels"].split("|")]
                    for alt in alts[:3]:
                        expanded.add(alt)

        return ", ".join(expanded)


def get_esco_expander(embedding_service=None, **kwargs):
    """Singleton getter."""
    global _esco_instance
    if _esco_instance is None:
        _esco_instance = ESCOExpander(embedding_service=embedding_service, **kwargs)
    return _esco_instance


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    expander = ESCOExpander()

    test_skills = [
        "Python, quản lý dự án, Excel",
        "lập trình web, React, Node.js",
        "kế toán tổng hợp, thuế, kiểm toán",
        "chăm sóc khách hàng, telesales",
    ]

    for skills in test_skills:
        expanded = expander.expand_skills(skills)
        print(f"\n  Input:    {skills}")
        print(f"  Expanded: {expanded}")
