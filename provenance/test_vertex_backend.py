"""Tests for the Vertex mirror -- of the MAPPING only, which is the part that is ours.

**The API contract is NOT tested and cannot be here.** There is no GCP project, the SDK is not
installed, and nothing in `vertex_backend` has run against a real Vertex ML Metadata store.
Argument names, resource-id rules and every permission question are unverified, so the first real
run should be expected to need corrections in the call sites.

What IS testable without a cloud: which graph node becomes which call, in which order, carrying
which fields, and that a failure degrades to a report rather than killing a run whose local record
is already complete.

Run: python3 provenance/test_vertex_backend.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from provenance.lineage import Artifact, Context, Execution, Graph
from provenance.vertex_backend import (
    MirrorReport,
    VertexBackendError,
    mirror,
    plan,
)

PROJECT, LOCATION = "a-project", "us-central1"


def graph() -> Graph:
    g = Graph(context=Context(id="ladder", type_name="Experiment", properties={"dataset": "m5"}))
    g.add_artifact(Artifact(id="panel", type_name="Dataset", uri="gs://b/panel",
                            content_hash="abc123"))
    g.add_artifact(Artifact(id="fc", type_name="Forecast", uri="gs://b/fc"))
    g.add_execution(Execution(id="rung3", type_name="Train",
                              parameters={"encoder_days": 364}))
    g.consumed("rung3", "panel")
    g.produced("rung3", "fc")
    return g


class FakeNode:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.inputs: list = []
        self.outputs: list = []

    def assign_input_artifacts(self, artifacts): self.inputs.extend(artifacts)
    def assign_output_artifacts(self, artifacts): self.outputs.extend(artifacts)


class FakeFactory:
    def __init__(self, log, kind, fail_on=None):
        self.log, self.kind, self.fail_on = log, kind, fail_on

    def create(self, **kwargs):
        self.log.append((self.kind, kwargs))
        if self.fail_on and kwargs.get("resource_id") == self.fail_on:
            raise RuntimeError(f"quota exceeded for {self.fail_on}")
        return FakeNode(**kwargs)


class FakeClient:
    """Stands in for `google.cloud.aiplatform`. Records what was asked of it."""

    def __init__(self, fail_on=None):
        self.log: list = []
        self.Context = FakeFactory(self.log, "Context")
        self.Artifact = FakeFactory(self.log, "Artifact", fail_on)
        self.Execution = FakeFactory(self.log, "Execution")


def test_the_plan_creates_ends_before_the_edges_that_reference_them() -> None:
    """Order is not incidental. Vertex cannot assign an artifact to an execution before both
    exist, exactly as `lineage.Graph` refuses a dangling edge."""
    steps = [c["step"] for c in plan(graph(), project=PROJECT, location=LOCATION)]
    assert steps.index("context:ladder") == 0
    assert steps.index("artifact:panel") < steps.index("inputs:rung3")
    assert steps.index("execution:rung3") < steps.index("inputs:rung3")
    assert steps.index("artifact:fc") < steps.index("outputs:rung3")


def test_our_type_names_map_onto_vertex_system_schemas() -> None:
    """So the console shows a dataset as a dataset rather than as an opaque custom type."""
    calls = {c["step"]: c["kwargs"] for c in plan(graph(), project=PROJECT, location=LOCATION)}
    assert calls["artifact:panel"]["schema_title"] == "system.Dataset"
    assert calls["execution:rung3"]["schema_title"] == "system.ContainerExecution"
    assert calls["context:ladder"]["schema_title"] == "system.Experiment"


def test_the_content_hash_travels_as_metadata_not_as_the_uri() -> None:
    """Two artifacts at the same uri with different bytes must stay distinguishable, which is the
    entire point of hashing them."""
    calls = {c["step"]: c["kwargs"] for c in plan(graph(), project=PROJECT, location=LOCATION)}
    assert calls["artifact:panel"]["metadata"]["content_hash"] == "abc123"
    assert calls["artifact:panel"]["uri"] == "gs://b/panel"
    assert "content_hash" not in calls["artifact:fc"]["metadata"], \
        "an artifact with no hash should not gain an empty one"


def test_every_call_carries_the_project_and_region() -> None:
    """A Vertex resource name embeds both; without them the graph lands somewhere unaddressable."""
    for call in plan(graph(), project=PROJECT, location=LOCATION):
        if call["call"].startswith("aiplatform."):
            assert call["kwargs"]["project"] == PROJECT
            assert call["kwargs"]["location"] == LOCATION


def test_a_plan_without_a_project_is_refused() -> None:
    for bad in (("", LOCATION), (PROJECT, "")):
        try:
            plan(graph(), project=bad[0], location=bad[1])
        except VertexBackendError as e:
            assert "cannot be addressed" in str(e)
        else:
            raise AssertionError(f"planned with project/location {bad}")


def test_a_mirror_writes_every_node_and_wires_the_edges() -> None:
    client = FakeClient()
    report = mirror(graph(), project=PROJECT, location=LOCATION, client=client)
    assert report.ok, str(report)
    assert report.written == report.attempted == 6
    kinds = [kind for kind, _ in client.log]
    assert kinds.count("Artifact") == 2 and kinds.count("Execution") == 1


def test_a_failure_is_reported_and_does_not_raise() -> None:
    """A run whose local record is complete is a valid run. Raising would let an unreachable
    tracking service fail a training job that had already produced its result."""
    report = mirror(graph(), project=PROJECT, location=LOCATION,
                    client=FakeClient(fail_on="panel"))
    assert not report.ok
    assert report.written < report.attempted
    assert any("quota exceeded" in msg for _, msg in report.errors)
    assert "INCOMPLETE" in str(report)


def test_an_absent_sdk_is_skipped_not_failed() -> None:
    """`google-cloud-aiplatform` is not installed here, which is the ordinary local case."""
    report = mirror(graph(), project=PROJECT, location=LOCATION)
    assert report.skipped_reason is not None
    assert "not installed" in report.skipped_reason
    assert not report.ok, "a mirror that recorded nothing is not ok"
    assert "SKIPPED" in str(report)


def test_a_skipped_mirror_is_distinguishable_from_a_clean_one() -> None:
    """An empty error list alone would read as success, which is how a tracking gap goes
    unnoticed for weeks."""
    skipped = MirrorReport(skipped_reason="no project")
    clean = MirrorReport(attempted=3, written=3)
    assert not skipped.ok and clean.ok
    assert skipped.errors == clean.errors == []


def test_an_unmapped_type_keeps_its_own_name_as_a_custom_schema() -> None:
    g = Graph(context=Context(id="c", type_name="Backtest"))
    g.add_artifact(Artifact(id="x", type_name="Ledger", uri="/x"))
    calls = {c["step"]: c["kwargs"] for c in plan(g, project=PROJECT, location=LOCATION)}
    assert calls["artifact:x"]["schema_title"] == "Ledger"
    assert calls["context:c"]["schema_title"] == "Backtest"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)
