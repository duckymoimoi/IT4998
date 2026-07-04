"""Read the project's term taxonomy into a case-insensitive lookup."""

from __future__ import annotations

import json
from pathlib import Path


def load_taxonomy_lookup(path: Path | str) -> tuple[dict[str, dict], int]:
    taxonomy_path = Path(path)
    if not taxonomy_path.exists():
        return {}, 0

    rows = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    lookup = {}
    for row in rows:
        for value in (row.get("term", ""), row.get("normalized_label", "")):
            value = str(value).strip()
            if value:
                lookup.setdefault(value.lower(), row)
    return lookup, len(rows)
