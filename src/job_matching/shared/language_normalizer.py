"""Normalize language names and language certificates consistently."""

from __future__ import annotations

import re


LANGUAGE_CERT_PATTERN = re.compile(
    r"(?i)\b(?:"
    r"IELTS|TOEIC|TOEFL(?:\s*iBT|\s*ITP)?|JLPT|HSK|TOPIK|VSTEP|APTIS|"
    r"Cambridge|PET|KET|FCE|CAE|CPE"
    r")\b"
)
LANGUAGE_NAMES = [
    "Tiếng Anh",
    "Tiếng Nhật",
    "Tiếng Trung",
    "Tiếng Hàn",
    "Tiếng Pháp",
    "Tiếng Đức",
    "Tiếng Nga",
    "Tiếng Tây Ban Nha",
    "Tiếng Việt",
]
CERTIFICATE_PATTERNS = [
    r"(?i)\bIELTS\s*\d+(?:[.,]\d+)?\+?\b",
    r"(?i)\bTOEIC\s*\d+\+?\b",
    r"(?i)\bTOEFL(?:\s*iBT|\s*ITP)?\s*\d+\+?\b",
    r"(?i)\bJLPT\s*N?[1-5]\b",
    r"(?i)\bHSK\s*[1-6]\b",
    r"(?i)\bTOPIK\s*[1-6]\b",
    r"(?i)\bVSTEP\s*[A-C][1-2]\b",
    r"(?i)\bAPTIS\s*[A-C][1-2]\b",
    r"(?i)\b(?:PET|KET|FCE|CAE|CPE)\b",
]


def split_items(value) -> list[str]:
    if not value:
        return []
    raw_items = value if isinstance(value, list) else re.split(r"[,;\n]", str(value))
    return [
        re.sub(r"\s+", " ", str(item)).strip(" .:-")
        for item in raw_items
        if str(item).strip()
    ]


def dedupe_items(items) -> list[str]:
    output = []
    seen = set()
    for item in items:
        cleaned = re.sub(r"\s+", " ", str(item)).strip(" .:-")
        key = cleaned.lower()
        if cleaned and key not in seen:
            output.append(cleaned)
            seen.add(key)
    return output


def language_name_from_text(text: str) -> str:
    folded = str(text).lower()
    return next((name for name in LANGUAGE_NAMES if name.lower() in folded), "")


def extract_language_certificate(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return ""

    for pattern in CERTIFICATE_PATTERNS:
        match = re.search(pattern, value)
        if match:
            return re.sub(r"\s+", " ", match.group(0)).strip()

    if re.search(r"(?i)\btiếng\s+nhật\b", value):
        match = re.search(r"(?i)\bN[1-5]\b", value)
        if match:
            return f"JLPT {match.group(0).upper()}"
    if re.search(r"(?i)\btiếng\s+trung\b", value):
        match = re.search(r"(?i)\bHSK\s*[1-6]\b", value)
        if match:
            return re.sub(r"\s+", "", match.group(0).upper())
    return ""


def normalize_language_certificates(languages, certificates) -> tuple[list[str], list[str]]:
    clean_languages = []
    clean_certificates = split_items(certificates)

    for item in split_items(languages):
        certificate = extract_language_certificate(item)
        language = language_name_from_text(item)
        if certificate:
            clean_certificates.append(certificate)
            if language:
                clean_languages.append(language)
        else:
            clean_languages.append(item)

    final_certificates = []
    for item in clean_certificates:
        certificate = extract_language_certificate(item)
        if certificate:
            final_certificates.append(certificate)
            language = language_name_from_text(item)
            if language:
                clean_languages.append(language)
        elif language_name_from_text(item) and not LANGUAGE_CERT_PATTERN.search(item):
            clean_languages.append(item)
        else:
            final_certificates.append(item)

    return dedupe_items(clean_languages), dedupe_items(final_certificates)
