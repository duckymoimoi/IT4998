"""Shared conversion helpers for CSV rows stored in Elasticsearch."""

from __future__ import annotations

import pandas as pd


def is_missing(value) -> bool:
    return value is None or bool(pd.isna(value))


def clean_text(value) -> str:
    return "" if is_missing(value) else str(value).strip()


def text_fields(row, fields) -> dict[str, str]:
    return {field: clean_text(row.get(field, "")) for field in fields}


def boolean_value(value) -> bool:
    if is_missing(value):
        return False
    if isinstance(value, str):
        value = value.lower() in {"true", "1", "yes"}
    return bool(value)


def salary_millions(value) -> float | None:
    if is_missing(value) or not value:
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    return amount / 1_000_000 if amount > 1000 else amount
