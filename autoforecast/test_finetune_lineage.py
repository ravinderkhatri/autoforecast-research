"""Functional local child-fine-tuning input validation."""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from scripts.run_autoforecast import _verified_parent_checkpoint

from autoforecast.train_workflow import TrainWorkflowError


def test_local_parent_checkpoint_is_hash_bound_before_child_training(
    tmp_path: pathlib.Path,
) -> None:
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"parent weights")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert _verified_parent_checkpoint(checkpoint, digest) == (checkpoint.resolve(), digest)
    with pytest.raises(TrainWorkflowError, match="digest differs"):
        _verified_parent_checkpoint(checkpoint, "0" * 64)

    link = tmp_path / "parent-link.pt"
    link.symlink_to(checkpoint)
    with pytest.raises(TrainWorkflowError, match="regular local file"):
        _verified_parent_checkpoint(link, digest)
