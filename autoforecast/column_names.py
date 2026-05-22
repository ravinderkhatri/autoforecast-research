"""Portable column identifiers shared by input contracts and Spark readers."""

from __future__ import annotations

import re
from collections.abc import Iterable


def validate_column_names(names: Iterable[object]) -> None:
    """Require ASCII identifiers; dots and backticks have expression meaning in Spark."""
    for name in names:
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ValueError(
                f"unsupported column name {name!r}; rename headers to ASCII identifiers "
                "matching [A-Za-z_][A-Za-z0-9_]*"
            )
