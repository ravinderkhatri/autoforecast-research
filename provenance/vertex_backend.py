"""Mirror a lineage graph into Vertex ML Metadata.

**THIS MODULE HAS NEVER RUN AGAINST GOOGLE CLOUD.** There is no GCP project attached to this
repository, `google-cloud-aiplatform` is not installed, and nothing here has been executed against
a real Vertex ML Metadata store. It is a hypothesis about an API, written so the mapping is
reviewable and so the first cloud run is a test of a thing that exists rather than a thing being
invented under time pressure.

What that means concretely, and it is not a formality:

  * The MAPPING is tested -- which graph node becomes which MLMD call, in which order, with which
    fields -- against a fake client, because that part is ours and can be wrong.
  * The API CONTRACT is not tested. Argument names, resource-id rules, whether
    `aiplatform.Artifact.create` accepts `schema_title` the way this assumes, and every quota and
    permission question are unverified. Expect the first real run to need corrections HERE, in the
    call sites, not in `lineage.py`.

WHY IT MIRRORS AND NEVER OWNS. `json_backend` is the source of truth because gate records are
claim-bearing and must be reconstructible from the repository alone. So every failure in this
module is a WARNING: a run that cannot reach Vertex is still a valid run with a complete local
record. `mirror` therefore returns a report instead of raising, and the caller decides.

WHY MLMD AND NOT A RUN LOG. The graph is already MLMD-shaped -- Artifact, Execution, Context,
Event -- so this is a writer, not a translator. That was the whole reason for choosing this
vocabulary over MLflow's experiment/run/params/metrics: lineage a run log never captured cannot be
reconstructed from it, so a project that tracks through one arrives at Vertex with runs and no
provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from provenance.lineage import Graph

__all__ = ["MirrorReport", "VertexBackendError", "mirror", "plan"]

SCHEMA_TITLE = {
    # Our type names -> Vertex's system schema titles. Anything unmapped keeps its own name, which
    # Vertex accepts as a custom schema; mapping the common ones means the console shows them as
    # datasets and models rather than as opaque custom types.
    "Dataset": "system.Dataset",
    "Model": "system.Model",
    "Metrics": "system.Metrics",
    "Forecast": "system.Artifact",
    "Train": "system.ContainerExecution",
    "Score": "system.ContainerExecution",
    "Build": "system.ContainerExecution",
    "Experiment": "system.Experiment",
}


class VertexBackendError(ValueError):
    """The graph cannot be expressed in Vertex ML Metadata."""


@dataclass
class MirrorReport:
    """What the mirror managed. **Never raises upward** -- see the module docstring.

    Attributes:
        attempted: Calls the plan contained.
        written: Calls that succeeded.
        errors: `(step, message)` for each failure, in order.
        skipped_reason: Why nothing was attempted, when that is the case -- an absent SDK, no
            project. Distinct from an empty error list, which would otherwise read as success.
    """

    attempted: int = 0
    written: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    skipped_reason: str | None = None

    @property
    def ok(self) -> bool:
        """Whether every attempted call landed. A skipped mirror is NOT ok -- nothing was
        recorded -- but it is also not a failed run."""
        return self.skipped_reason is None and not self.errors and self.attempted > 0

    def __str__(self) -> str:
        if self.skipped_reason:
            return f"vertex mirror SKIPPED: {self.skipped_reason}"
        if self.errors:
            first = self.errors[0]
            return (f"vertex mirror INCOMPLETE: {self.written}/{self.attempted} written; "
                    f"first error at {first[0]}: {first[1]}")
        return f"vertex mirror: {self.written}/{self.attempted} written"


def plan(graph: Graph, *, project: str, location: str) -> list[dict[str, object]]:
    """The ordered calls that would reproduce `graph` in Vertex ML Metadata.

    Separated from execution so the MAPPING can be tested without a cloud project -- which is the
    only part of this module that is ours to get right. Order matters and is not incidental:
    contexts and artifacts must exist before the events that reference them, exactly as
    `lineage.Graph` refuses a dangling edge.

    Args:
        graph: What to mirror.
        project: GCP project id.
        location: Vertex region, e.g. `us-central1`.

    Returns:
        A list of `{"step", "call", "kwargs"}` records.

    Raises:
        VertexBackendError: no project or location. Vertex resource names embed both, so a graph
            written without them would land somewhere unaddressable.
    """
    if not project or not location:
        raise VertexBackendError(
            "project and location are required: a Vertex resource name embeds both, and a graph "
            "written without them cannot be addressed afterwards")

    calls: list[dict[str, object]] = [{
        "step": f"context:{graph.context.id}",
        "call": "aiplatform.Context.create",
        "kwargs": {"resource_id": graph.context.id,
                   "schema_title": SCHEMA_TITLE.get(graph.context.type_name,
                                                    graph.context.type_name),
                   "metadata": dict(graph.context.properties),
                   "project": project, "location": location},
    }]
    for artifact in graph.artifacts.values():
        metadata = dict(artifact.properties)
        # The content hash travels as metadata rather than as the uri, so two artifacts at the
        # same uri with different bytes stay distinguishable -- which is the point of hashing.
        if artifact.content_hash:
            metadata["content_hash"] = artifact.content_hash
        calls.append({
            "step": f"artifact:{artifact.id}",
            "call": "aiplatform.Artifact.create",
            "kwargs": {"resource_id": artifact.id, "uri": artifact.uri,
                       "schema_title": SCHEMA_TITLE.get(artifact.type_name, artifact.type_name),
                       "metadata": metadata, "project": project, "location": location},
        })
    for execution in graph.executions.values():
        calls.append({
            "step": f"execution:{execution.id}",
            "call": "aiplatform.Execution.create",
            "kwargs": {"resource_id": execution.id,
                       "schema_title": SCHEMA_TITLE.get(execution.type_name,
                                                        execution.type_name),
                       "state": execution.state,
                       "metadata": dict(execution.parameters),
                       "project": project, "location": location},
        })
    # Edges last, because both ends must exist. Vertex expresses them as assign_input_artifacts
    # and assign_output_artifacts on the execution rather than as standalone objects.
    for execution_id in graph.executions:
        inputs = [a.id for a in graph.inputs_of(execution_id)]
        outputs = [a.id for a in graph.outputs_of(execution_id)]
        if inputs:
            calls.append({"step": f"inputs:{execution_id}",
                          "call": "Execution.assign_input_artifacts",
                          "kwargs": {"execution": execution_id, "artifacts": inputs}})
        if outputs:
            calls.append({"step": f"outputs:{execution_id}",
                          "call": "Execution.assign_output_artifacts",
                          "kwargs": {"execution": execution_id, "artifacts": outputs}})
    return calls


def mirror(graph: Graph, *, project: str, location: str, client=None) -> MirrorReport:
    """Write `graph` to Vertex ML Metadata. **Returns a report; does not raise.**

    A run whose local record is complete is a valid run whether or not the mirror succeeded, so
    every failure here is a warning the caller decides about. Raising would let an unreachable
    tracking service fail a training job that had already produced its result.

    Args:
        graph: What to mirror.
        project: GCP project id.
        location: Vertex region.
        client: An object exposing the `plan` call names, for testing. `None` imports
            `google.cloud.aiplatform`, and an absent SDK is reported as skipped rather than
            raised.

    Returns:
        A `MirrorReport`.
    """
    if client is None:
        try:
            from google.cloud import aiplatform  # noqa: PLC0415
            client = aiplatform
        except ImportError as exc:
            return MirrorReport(skipped_reason=f"google-cloud-aiplatform not installed ({exc})")

    try:
        calls = plan(graph, project=project, location=location)
    except VertexBackendError as exc:
        return MirrorReport(skipped_reason=str(exc))

    report = MirrorReport(attempted=len(calls))
    created: dict[str, object] = {}
    for call in calls:
        step, name, kwargs = call["step"], call["call"], dict(call["kwargs"])
        try:
            if name.startswith("aiplatform."):
                factory = getattr(client, name.split(".", 1)[1].split(".")[0])
                created[step] = factory.create(**kwargs)
            else:
                execution = created.get(f"execution:{kwargs['execution']}")
                if execution is None:
                    raise VertexBackendError(
                        f"execution {kwargs['execution']} was not created, so its edges cannot "
                        "be assigned")
                artifacts = [created[f"artifact:{a}"] for a in kwargs["artifacts"]]
                getattr(execution, name.split(".", 1)[1])(artifacts)
            report.written += 1
        except Exception as exc:                                    # noqa: BLE001
            # Broad on purpose: this module has never run against the real SDK, so the exception
            # types it raises are not known here. Recording the message beats guessing a hierarchy.
            report.errors.append((step, f"{type(exc).__name__}: {exc}"))
    return report
