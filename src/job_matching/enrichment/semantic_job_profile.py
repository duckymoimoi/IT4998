"""Build a clean, semantic-rich single-vector profile for a job."""

from __future__ import annotations

import json
import re
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from job_matching.enrichment.build_term_taxonomy import split_terms
from job_matching.enrichment.term_taxonomy import load_taxonomy_lookup


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "data" / "job_term_taxonomy.json"
DEFAULT_PENDING_PATH = PROJECT_ROOT / "data" / "job_term_taxonomy_pending.jsonl"
_DEFAULT_BUILDER = None
_DEFAULT_BUILDER_TAXONOMY_SIGNATURE = None
_PENDING_TERMS_LOCK = threading.Lock()

BM25_TYPES = {
    "role", "domain", "technical_skill", "tool", "professional_skill",
    "language", "certification",
}
SEMANTIC_TYPES = BM25_TYPES
TYPE_LIMITS = {
    "role": 3,
    "domain": 5,
    "professional_skill": 9,
    "technical_skill": 12,
    "tool": 10,
    "language": 4,
    "certification": 4,
}

TITLE_NOISE_PATTERNS = [
    r"\boffer\b.*$",
    r"\bthu nhập\b.*$",
    r"\blương\b.*$",
    r"\bđi làm ngay\b.*$",
    r"\bnhận việc ngay\b.*$",
    r"\bkhông yêu cầu kinh nghiệm\b.*$",
    r"\btại\s+(hà nội|hồ chí minh|tp\.?\s*hcm|đà nẵng)\b.*$",
    r"\bupto\b.*$",
    r"\blên đến\s+\d+.*$",
    r"\bđến\s+\d+\s*triệu.*$",
]
DESCRIPTION_NOISE_PATTERNS = [
    r"\b(mức lương|thu nhập|phúc lợi|quyền lợi|thưởng|được hưởng)\b",
    r"\b(địa điểm|thời gian làm việc|hạn nộp|ứng tuyển)\b",
    r"\b(chuyên nghiệp|chịu áp lực|năng động|nhiệt tình|cẩn thận)\b",
]


class SemanticJobProfileBuilder:
    def __init__(self, taxonomy_path: Path | str = DEFAULT_TAXONOMY_PATH):
        self.taxonomy_path = Path(taxonomy_path)
        self.taxonomy = self._load_taxonomy()
        self.evidence_candidates = self._build_evidence_candidates()

    def _load_taxonomy(self) -> Dict[str, Dict]:
        lookup, _ = load_taxonomy_lookup(self.taxonomy_path)
        return lookup

    def _build_evidence_candidates(self):
        """Compile high-precision taxonomy terms that can be recovered from JD/JR."""
        candidates = []
        seen = set()
        for item in self.taxonomy.values():
            term_type = item.get("type")
            if term_type not in BM25_TYPES:
                continue
            try:
                confidence = float(item.get("confidence") or 0)
            except (TypeError, ValueError):
                confidence = 0
            if confidence < 0.85:
                continue

            label = str(item.get("normalized_label") or item.get("term") or "").strip()
            surfaces = self._dedupe([item.get("term", ""), label])
            key = (term_type, label.lower())
            if not label or key in seen:
                continue

            safe_surfaces = []
            for surface in surfaces:
                # Vietnamese single-word terms are often ambiguous in prose.
                # Multi-word terms and ASCII identifiers are safer exact
                # matches for vocabulary-backed evidence extraction.
                if " " not in surface:
                    try:
                        surface.encode("ascii")
                    except UnicodeEncodeError:
                        continue
                if len(surface) >= 3 or term_type in {"language", "certification"}:
                    safe_surfaces.append(surface)
            if not safe_surfaces:
                continue

            needles = [
                re.sub(r"\s+", " ", surface).lower()
                for surface in safe_surfaces
            ]
            patterns = [
                re.compile(
                    r"(?<!\w)" + re.escape(surface).replace(r"\ ", r"\s+") + r"(?!\w)",
                    flags=re.I | re.UNICODE,
                )
                for surface in safe_surfaces
            ]
            candidates.append((term_type, label, needles, patterns))
            seen.add(key)
        return candidates

    @staticmethod
    def _dedupe(values: Iterable[str], limit: int | None = None) -> List[str]:
        output, seen = [], set()
        for value in values:
            cleaned = re.sub(r"\s+", " ", str(value)).strip(" .:-()[]{}")
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            output.append(cleaned)
            seen.add(key)
            if limit and len(output) >= limit:
                break
        return output

    @staticmethod
    def clean_title(title: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(title or "")).strip()
        cleaned = re.sub(r"\[[^\]]{1,40}\]", " ", cleaned)
        cleaned = re.sub(r"\([^)]*(lương|thu nhập|đi làm|tại hà nội|tại hồ chí minh)[^)]*\)", " ", cleaned, flags=re.I)
        for pattern in TITLE_NOISE_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.I)
        return re.sub(r"\s+", " ", cleaned).strip(" ,-")

    @staticmethod
    def _task_summary(text: str, max_sentences: int = 3, max_chars: int = 420) -> str:
        text = re.sub(r"^[•\-–—*+\d.)\s]+", "", str(text or ""), flags=re.M)
        candidates = re.split(r"[\n\r]+|(?<=[.!?;])\s+", text)
        selected = []
        for candidate in candidates:
            sentence = re.sub(r"\s+", " ", candidate).strip(" .:-")
            if len(sentence) < 12:
                continue
            if any(re.search(pattern, sentence, flags=re.I) for pattern in DESCRIPTION_NOISE_PATTERNS):
                continue
            selected.append(sentence)
            if len(selected) >= max_sentences:
                break
        summary = ". ".join(selected)
        return summary[:max_chars].rsplit(" ", 1)[0] if len(summary) > max_chars else summary

    @staticmethod
    def _evidence_text(job: dict) -> str:
        return re.sub(
            r"\s+", " ",
            " ".join(str(job.get(field, "") or "") for field in [
                "title", "job_description", "job_requirements",
            ]),
        ).lower()

    @staticmethod
    def _has_evidence(evidence: str, value: str) -> bool:
        """Match a complete term/phrase instead of an arbitrary substring."""
        evidence = str(evidence or "").lower()
        value = re.sub(r"\s+", " ", str(value or "")).strip().lower()
        if not value:
            return False
        pattern = r"(?<!\w)" + re.escape(value).replace(r"\ ", r"\s+") + r"(?!\w)"
        return bool(re.search(pattern, evidence, flags=re.UNICODE))

    def _typed_terms(self, job: dict):
        grouped = defaultdict(list)
        unknown = []
        evidence = self._evidence_text(job)
        fields = [
            "specializations",
            "technical_skills",
            "certificates",
            "languages",
        ]
        for field in fields:
            for term in split_terms(job.get(field, "")):
                item = self.taxonomy.get(term.lower())
                if not item:
                    unknown.append(term)
                    continue
                term_type = item.get("type", "unknown")
                label = item.get("normalized_label") or term
                # Valid tools/technical terms can still be wrong for a specific
                # job due to noisy extraction. Require contextual evidence.
                if term_type in {"technical_skill", "tool"}:
                    if not self._has_evidence(evidence, term) and not self._has_evidence(evidence, label):
                        continue
                if term_type in BM25_TYPES:
                    grouped[term_type].append(label)

        # Recover known taxonomy terms that the small cleaner model missed.
        # This is exact evidence matching only: no ESCO and no semantic guess.
        for term_type, label, needles, patterns in self.evidence_candidates:
            if any(needle in evidence for needle in needles) and any(
                pattern.search(evidence) for pattern in patterns
            ):
                grouped[term_type].append(label)

        return {
            key: self._dedupe(values, TYPE_LIMITS.get(key))
            for key, values in grouped.items()
        }, self._dedupe(unknown)

    def build_searchable_fields(self, job: dict) -> dict[str, str]:
        """Build normalized BM25 fields from taxonomy terms with text evidence."""
        typed, _ = self._typed_terms(job)
        technical = []
        for term_type in ["professional_skill", "technical_skill", "tool", "domain"]:
            technical.extend(typed.get(term_type, []))

        requirements = []
        for term_type in [
            "role", "domain", "professional_skill", "technical_skill",
            "tool", "language", "certification",
        ]:
            requirements.extend(typed.get(term_type, []))

        specializations = []
        for term_type in ["role", "domain", "professional_skill"]:
            specializations.extend(typed.get(term_type, []))

        fields = {
            "technical_skills": self._dedupe(technical, 32),
            "requirements_tags": self._dedupe(requirements, 40),
            "specializations": self._dedupe(specializations, 24),
            "languages": self._dedupe(typed.get("language", []), 8),
            "certificates": self._dedupe(typed.get("certification", []), 10),
        }
        return {
            field: ", ".join(values)
            for field, values in fields.items()
            if values
        }

    def build(self, job: dict, include_searchable_fields: bool = False) -> dict:
        typed, unknown = self._typed_terms(job)
        title = self.clean_title(job.get("title", ""))
        task_summary = self._task_summary(job.get("job_description", ""))
        # TopCV specialization labels can contain "/" and parentheses as part
        # of one normalized label. Preserve their surface form exactly.
        source_specializations = []
        seen_specializations = set()
        for raw_value in str(job.get("specializations", "") or "").split(","):
            value = re.sub(r"\s+", " ", raw_value).strip()
            key = value.lower()
            if value and key not in seen_specializations:
                source_specializations.append(value)
                seen_specializations.add(key)
            if len(source_specializations) >= 8:
                break
        specialization_keys = {value.lower() for value in source_specializations}
        unknown = [
            value for value in unknown
            if value.lower() not in specialization_keys
        ]

        lines = []
        if title:
            lines.append(f"Vai trò: {title}.")
        if source_specializations:
            lines.append(f"Chuyên môn: {', '.join(source_specializations)}.")
        labels = [
            ("Lĩnh vực", "domain"),
            ("Nghiệp vụ", "professional_skill"),
            ("Kỹ năng kỹ thuật", "technical_skill"),
            ("Công cụ", "tool"),
            ("Ngoại ngữ", "language"),
            ("Chứng chỉ", "certification"),
        ]
        for display, term_type in labels:
            values = typed.get(term_type, [])
            if values and term_type in SEMANTIC_TYPES:
                lines.append(f"{display}: {', '.join(values)}.")
        if task_summary:
            lines.append(f"Nhiệm vụ chính: {task_summary}.")

        # Unknown compact tags are safer than raw requirements and help new
        # crawl terms remain searchable before the next taxonomy batch update.
        if unknown:
            lines.append(f"Thuật ngữ chưa chuẩn hóa: {', '.join(unknown[:6])}.")

        profile = {
            "semantic_text": " ".join(lines).strip(),
            "semantic_title": title,
        }
        if include_searchable_fields:
            profile.update(self.build_searchable_fields(job))
        return profile


def append_pending_terms(
    job: dict,
    pending_path: Path | str = DEFAULT_PENDING_PATH,
    taxonomy_path: Path | str = DEFAULT_TAXONOMY_PATH,
) -> int:
    """Append unknown crawl terms for later batch classification."""
    global _DEFAULT_BUILDER, _DEFAULT_BUILDER_TAXONOMY_SIGNATURE
    with _PENDING_TERMS_LOCK:
        taxonomy_path = Path(taxonomy_path)
        taxonomy_stat = taxonomy_path.stat() if taxonomy_path.exists() else None
        taxonomy_signature = (
            (taxonomy_stat.st_mtime_ns, taxonomy_stat.st_size)
            if taxonomy_stat
            else None
        )
        if (
            _DEFAULT_BUILDER is None
            or _DEFAULT_BUILDER.taxonomy_path != taxonomy_path
            or _DEFAULT_BUILDER_TAXONOMY_SIGNATURE != taxonomy_signature
        ):
            _DEFAULT_BUILDER = SemanticJobProfileBuilder(taxonomy_path)
            _DEFAULT_BUILDER_TAXONOMY_SIGNATURE = taxonomy_signature
        builder = _DEFAULT_BUILDER
        _, unknown = builder._typed_terms(job)
        if not unknown:
            return 0

        pending_path = Path(pending_path)
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        with pending_path.open("a", encoding="utf-8") as handle:
            for term in unknown:
                handle.write(json.dumps({
                    "term": term,
                    "source_url": job.get("url", ""),
                    "source_title": job.get("title", ""),
                    "collected_at": now,
                }, ensure_ascii=False) + "\n")
        return len(unknown)
