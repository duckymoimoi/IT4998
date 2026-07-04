"""Small, dependency-free helpers for environment configuration."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import TypeVar


T = TypeVar("T")
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def numbered_env_values(
    prefixes: str | Iterable[str],
    *,
    max_index: int = 9,
    include_single: bool = True,
    single_only_as_fallback: bool = False,
    strip: bool = False,
    deduplicate: bool = False,
) -> list[str]:
    """Load PREFIX_1..N values and optional unnumbered fallbacks in stable order."""
    if isinstance(prefixes, str):
        prefixes = (prefixes,)

    values: list[str] = []
    for prefix in prefixes:
        numbered = []
        for index in range(1, max_index + 1):
            value = os.environ.get(f"{prefix}_{index}")
            if value:
                value = value.strip() if strip else value
                if value and (not deduplicate or value not in numbered):
                    numbered.append(value)
        values.extend(
            value
            for value in numbered
            if not deduplicate or value not in values
        )

        single = os.environ.get(prefix) if include_single else None
        if single and (not single_only_as_fallback or not numbered):
            single = single.strip() if strip else single
            if single and single not in values:
                values.append(single)
    return values
