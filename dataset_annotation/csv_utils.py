from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def find_cols(headers: List[str], candidates: List[str]) -> Optional[str]:
    lowered = [header.lower() for header in headers]
    for candidate in candidates:
        if candidate.lower() in lowered:
            return headers[lowered.index(candidate.lower())]
    return None
