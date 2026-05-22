"""Portable storage and MLMD-shaped lineage, with no dependency on any tracking service.

**Deliberately project-agnostic.** Nothing here imports `autoforecast` or `forecast`; the
vocabulary is Artifact, Execution, Context and Event, which is MLMD's, and the storage type knows
only that a location is a path or an object-store prefix. Another project can copy this directory
and use it unchanged.

Nothing is imported at package level, so a caller that only needs `storage` does not pay for
anything else.
"""

from __future__ import annotations

__all__ = ["lineage", "storage"]
