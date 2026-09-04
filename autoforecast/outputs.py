"""Show where a study's inputs and a workflow's outputs are, and what each one holds.

The map is built only from records AutoForecast already keeps: the registration and
admission records, the workflow result and its MLflow links. Building it scans no table;
source table descriptions come from catalog metadata. On Databricks, items link to Catalog
Explorer and MLflow; elsewhere they show their locations only.
"""

from __future__ import annotations

import dataclasses
import html
import json
import os
import pathlib
from typing import Any

SECTIONS = ("data", "training")
_TITLES = {"data": "Study data", "training": "Training results"}


@dataclasses.dataclass(frozen=True)
class OutputItem:
    """One input or output: where it is and what it holds."""

    section: str
    group: str
    name: str
    location: str
    description: str
    link: str | None = None


@dataclasses.dataclass(frozen=True)
class OutputsMap:
    """Headline facts and described locations, for one or both sections."""

    facts: tuple[tuple[str, str, str], ...] = ()
    items: tuple[OutputItem, ...] = ()

    def select(self, section: str = "all") -> OutputsMap:
        """Keep one section ("data" or "training"), or both with "all"."""
        if section not in {"all", *SECTIONS}:
            raise ValueError("section must be 'data', 'training' or 'all'")
        keep = SECTIONS if section == "all" else (section,)
        return OutputsMap(tuple(f for f in self.facts if f[0] in keep),
                          tuple(i for i in self.items if i.section in keep))

    def merge(self, other: OutputsMap) -> OutputsMap:
        return OutputsMap(self.facts + other.facts, self.items + other.items)

    def _sections(self):
        for section in SECTIONS:
            facts = [f for f in self.facts if f[0] == section]
            items = [i for i in self.items if i.section == section]
            if facts or items:
                yield section, facts, items

    def __str__(self) -> str:
        lines = []
        for section, facts, items in self._sections():
            lines += ["", f"== {_TITLES[section]} =="]
            lines += [f"{label}: {value}" for _, label, value in facts]
            group = None
            for item in items:
                if item.group != group:
                    group = item.group
                    lines += ["", f"{group}"]
                lines.append(f"  {item.name}: {item.location}")
                lines.append(f"      {item.description}")
                if item.link:
                    lines.append(f"      {item.link}")
        return "\n".join(lines).strip() or "No study data or training results yet."

    def _repr_html_(self) -> str:
        parts = []
        for section, facts, items in self._sections():
            parts.append(f"<h3>{html.escape(_TITLES[section])}</h3>")
            if facts:
                parts.append("<p>" + "<br>".join(
                    f"<b>{html.escape(label)}</b>: {html.escape(value)}"
                    for _, label, value in facts) + "</p>")
            rows = []
            for item in items:
                where = html.escape(item.location)
                if item.link:
                    where = f'<a href="{html.escape(item.link)}" target="_blank">{where}</a>'
                rows.append(f"<tr><td>{html.escape(item.group)}</td>"
                            f"<td>{html.escape(item.name)}</td><td>{where}</td>"
                            f"<td>{html.escape(item.description)}</td></tr>")
            if rows:
                parts.append("<table><tr><th>Group</th><th>What</th><th>Where</th>"
                             "<th>Contains</th></tr>" + "".join(rows) + "</table>")
        return "".join(parts) or "<p>No study data or training results yet.</p>"

    def to_dict(self) -> dict[str, Any]:
        return {"facts": [{"section": s, "label": label, "value": value}
                          for s, label, value in self.facts],
                "items": [dataclasses.asdict(item) for item in self.items]}

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame([dataclasses.asdict(item) for item in self.items],
                            columns=[f.name for f in dataclasses.fields(OutputItem)])


def show(outputs: OutputsMap, logger=None) -> None:
    """Render in a notebook's cell output when there is one; otherwise log plain text."""
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display

        if getattr(get_ipython(), "kernel", None) is not None:
            display(HTML(outputs._repr_html_()))
            return
    except ImportError:
        pass
    if logger is not None:
        logger.info("%s", outputs)


def _workspace_host() -> str | None:
    """The Databricks workspace URL when running there; links are omitted elsewhere."""
    if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
        return None
    try:
        from mlops import _volume_files

        host = _volume_files._client().config.host
    except Exception:
        return None
    return host.rstrip("/") if isinstance(host, str) and host.startswith("https://") else None


def _table_link(host, name):
    parts = name.replace("`", "").split(".")
    return f"{host}/explore/data/{'/'.join(parts)}" if host and len(parts) == 3 else None


def _volume_link(host, path):
    parts = pathlib.PurePosixPath(str(path)).parts
    if host and len(parts) >= 5 and parts[1] == "Volumes":
        return f"{host}/explore/data/volumes/{parts[2]}/{parts[3]}/{parts[4]}"
    return None


def mlflow_run_link(host, experiment_id, run_id):
    return f"{host}/ml/experiments/{experiment_id}/runs/{run_id}" if host else None


def _table_comment(spark, name):
    if spark is None:
        return None
    try:
        return spark.catalog.getTable(name).description or None
    except Exception:
        return None


def _count(value):
    return f"{value:,}" if type(value) is int else None


def study_outputs(registry, study_id: str, *, registration: dict | None = None) -> OutputsMap:
    """Describe a registered study's inputs and saved data, from its saved records."""
    from autoforecast.study_registration import _read_saved, _verify_admission

    if registration is None:
        registration = _read_saved(registry._registration_path(study_id))
    definition = registration["definition"]
    host, root = _workspace_host(), pathlib.Path(registry.root)
    spark = getattr(registry, "spark", None)
    source = definition.get("source") or {}
    admission = None
    if isinstance(definition.get("admission_evidence"), dict):
        admission = _verify_admission(root, definition["admission_evidence"],
                                      inspect_artifacts=False)
    facts = [("data", "Study ID", study_id)]
    series = [f"{_count(source.get('starting_series'))} found"
              if _count(source.get("starting_series")) else None,
              f"{_count(source.get('eligible_series'))} after business rules"
              if _count(source.get("eligible_series")) else None,
              f"{_count(definition.get('effective_series'))} forecast"
              if _count(definition.get("effective_series")) else None]
    if any(series):
        facts.append(("data", "Series", " · ".join(part for part in series if part)))
    items = []
    queries = {"primary": source}
    if admission is not None:
        queries.update({name: value for name, value in admission["identity"]["sources"].items()
                        if name != "primary" and isinstance(value, dict)})
    seen = set()
    for query, record in queries.items():
        tables = record.get("tables") or {}
        for alias, pin in sorted(tables.items()):
            name = pin.get("table") if isinstance(pin, dict) else None
            if not name or name in seen:
                continue
            seen.add(name)
            version = pin.get("version")
            items.append(OutputItem(
                "data", "Read (pinned versions)", alias,
                f"{name} (version {version})" if version is not None else name,
                _table_comment(spark, name) or "Source table read when the study was registered",
                _table_link(host, name)))
        rows = _count(record.get("rows"))
        if "sql" in record:
            items.append(OutputItem(
                "data", "Read (pinned versions)", f"{query} query",
                "saved in the study definition",
                f"{rows} rows returned" if rows else "Query run against the tables above"))
    store = getattr(registry, "store", None)
    tables = getattr(store, "tables", None) or {}
    if tables:
        from autoforecast.integrations.delta import DESCRIPTIONS

        for kind in ("studies", "study_data_issues"):
            description = DESCRIPTIONS[kind][0]
            if kind == "studies":
                description += f"; this study's row has study_id {study_id}"
            elif admission is not None:
                # source_findings already includes the population findings.
                findings = sum(
                    (record.get("logical_hash") or record.get("snapshot") or {}).get("rows", 0)
                    for name, record in admission["artifacts"].items()
                    if name.endswith("_findings") and name != "population_findings")
                description += f"; {findings:,} rows for this study (filter on study_id)"
            items.append(OutputItem("data", "Written by AutoForecast", kind, tables[kind],
                                    description, _table_link(host, tables[kind])))
    frozen = root / "benchmarks" / study_id
    items.append(OutputItem(
        "data", "Saved study", "Frozen study data", str(frozen),
        "Training, validation and test data and scoring grids, exactly as scored",
        _volume_link(host, frozen)))
    items.append(OutputItem(
        "data", "Saved study", "Registration record",
        str(registry._registration_path(study_id)),
        "Study definition: sources with pinned versions, business rules and windows",
        _volume_link(host, root)))
    if admission is not None:
        outcome = admission.get("outcome") or {}
        counts = ", ".join(f"{label} {_count(outcome[key])}" for key, label in (
            ("starting", "found"), ("application_retained", "kept by business rules"),
            ("final_retained", "kept for forecasting")) if _count(outcome.get(key)))
        items.append(OutputItem(
            "data", "Saved study", "Data checks and exclusions",
            str(root / definition["admission_evidence"]["path"]),
            "Series excluded and why, plus every data-check finding"
            + (f": {counts}" if counts else ""),
            _volume_link(host, root)))
    return OutputsMap(tuple(facts), tuple(items))


def _metrics(metrics: dict | None) -> str | None:
    if not metrics:
        return None
    return " · ".join(f"{name} {value:.4g}" if isinstance(value, int | float) else f"{name} n/a"
                      for name, value in metrics.items())


def workflow_outputs(result) -> OutputsMap:
    """Describe a completed workflow's search, models, forecasts, scores and MLflow runs."""
    host = _workspace_host()
    facts = [("training", "Run", f"{result.run_id} · {result.status}")]
    if result.parameters:
        facts.append(("training", "Selected settings", ", ".join(
            f"{name.removeprefix('model.')}={value}"
            for name, value in sorted(result.parameters.items()))))
    for label, metrics in (("Validation scores", result.validation_metrics),
                           ("Test scores", result.testing_metrics)):
        if _metrics(metrics):
            facts.append(("training", label, _metrics(metrics)))
    items = []
    if result.tuning_path is not None:
        search = pathlib.Path(result.tuning_path)
        items.append(OutputItem(
            "training", "Parameter search", "Cross-validation results", str(search),
            "Every trial's settings and its score in each tuning window",
            _volume_link(host, search)))
        items.append(OutputItem(
            "training", "Parameter search", "Search report", str(search / "report.html"),
            "Trials compared side by side, with the selected one marked",
            _volume_link(host, search)))
    items.append(OutputItem(
        "training", "Parameter search", "Selected settings", str(result.selected_config),
        "The selected parameters and the search profile they came from",
        _volume_link(host, result.selected_config)))
    for stage, evidence in (("validation", result.validation), ("test", result.testing)):
        if evidence is None:
            continue
        title = stage.capitalize()
        forecasts = evidence["forecasts"]
        items.append(OutputItem(
            "training", title, "Trained model", str(evidence["model_artifacts"]),
            f"Saved model ({evidence['model_format']})",
            _volume_link(host, evidence["model_artifacts"])))
        items.append(OutputItem(
            "training", title, "Forecasts", str(forecasts.path),
            f"{forecasts.n_series:,} series × {forecasts.horizon} periods from {forecasts.origin}",
            _volume_link(host, forecasts.path)))
        items.append(OutputItem(
            "training", title, "Scores", str(pathlib.Path(result.path) / stage / "scores"),
            "Scores overall, per series and per forecast period",
            _volume_link(host, result.path)))
        link = (result.tracking or {}).get(stage)
        if link:
            items.append(OutputItem(
                "training", title, "MLflow run", link.get("run_id", ""),
                "Metrics, settings, model, forecasts and scores, with the input tables",
                link.get("url")))
    items.append(OutputItem(
        "training", "Workflow", "All saved evidence", str(result.path),
        "Everything this run saved; RUN.md lists every file with its hash",
        _volume_link(host, result.path)))
    publication = result.publication or {}
    if publication.get("status") not in (None, "not_requested"):
        items.append(OutputItem(
            "training", "Workflow", "Result tables", str(publication.get("status")),
            "Scores and forecasts published to the study_results and forecasts tables"))
    return OutputsMap(tuple(facts), tuple(items))


def tracking_records(links: dict | None) -> dict[str, dict[str, str | None]]:
    """MLflow run IDs and, on Databricks, run page links for each workflow stage."""
    host = _workspace_host()
    return {stage: {"experiment_id": link.experiment_id, "run_id": link.run_id,
                    "url": mlflow_run_link(host, link.experiment_id, link.run_id)}
            for stage, link in (links or {}).items()}


def as_json(outputs: OutputsMap) -> str:
    return json.dumps(outputs.to_dict(), indent=2, sort_keys=True)
