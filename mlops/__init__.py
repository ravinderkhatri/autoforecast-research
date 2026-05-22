"""Small, task-agnostic records for reproducible ML training runs.

Forecasting, supervised learning and LLM fine-tuning own their execution code.  This package
only records the immutable identities every one of those runs needs to be recovered.
"""

from mlops.provenance import (
    CheckpointCapability,
    DataSnapshot,
    ProvenanceError,
    RuntimeIdentity,
    SourceIdentity,
    capture_runtime_identity,
    capture_source_identity,
    local_data_reference,
    sanitize_web_url,
    validate_data_reference,
    verify_retained_ref,
    write_run_provenance,
)
from mlops.release import RetainedRun, verify_retained_run
from mlops.run_bundle import RunBundle, load_completed_run_bundle, open_run_bundle

__all__ = [
    "CheckpointCapability",
    "DataSnapshot",
    "ProvenanceError",
    "RuntimeIdentity",
    "SourceIdentity",
    "capture_runtime_identity",
    "capture_source_identity",
    "local_data_reference",
    "sanitize_web_url",
    "verify_retained_ref",
    "validate_data_reference",
    "write_run_provenance",
    "RunBundle",
    "load_completed_run_bundle",
    "open_run_bundle",
    "RetainedRun",
    "verify_retained_run",
]
