"""Portable column identifiers shared by input contracts and Spark readers."""

from __future__ import annotations

import re
from collections.abc import Iterable


def validate_column_names(names: Iterable[object]) -> None:
    """Require ASCII identifiers; dots and backticks have expression meaning in Spark."""
    seen: dict[str, str] = {}
    for name in names:
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ValueError(
                f"unsupported column name {name!r}; rename headers to ASCII identifiers "
                "matching [A-Za-z_][A-Za-z0-9_]*"
            )
        folded = name.casefold()
        if folded in seen and seen[folded] != name:
            raise ValueError(
                f"column names {seen[folded]!r} and {name!r} differ only by case; "
                "rename headers so names are distinct ignoring case"
            )
        seen[folded] = name


def fresh_internal_names(existing: Iterable[str], stems: Iterable[str]) -> tuple[str, ...]:
    """Return deterministic Spark column names distinct from caller columns ignoring case."""
    used = {name.casefold() for name in existing}
    names = []
    for stem in stems:
        candidate = stem
        suffix = 2
        while candidate.casefold() in used:
            candidate = f"{stem}_{suffix}"
            suffix += 1
        names.append(candidate)
        used.add(candidate.casefold())
    return tuple(names)
