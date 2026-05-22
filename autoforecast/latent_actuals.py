"""Governed latent-demand actuals: the only door to scoring `unconstrained_demand`.

The scorers already fail closed -- a latent-demand forecast against observed sales is refused,
because sales are not the quantity. What was missing is the DOOR: latent "actuals" are not
observations, they are estimates from somewhere (a control-store experiment, distributor
sell-in, a stock-tracking study), and scoring against an unexplained parquet of numbers is the
old leaderboard failure wearing an estimand. This loader is the governance:

  * the artifact carries a MANIFEST stating **how** the latents were measured, from what
    source, when it was registered -- the claims a reader must be able to interrogate;
  * the manifest binds the parquet's CONTENT hash, so an edited artifact stops loading rather
    than quietly scoring;
  * the loader's only success output is a validated `Panel` with
    `target_estimand='unconstrained_demand'` -- which is exactly what the existing scorers'
    estimand-equality check demands, so no scorer grows a bypass: the governance IS the only
    way to construct the panel they will accept.
"""

from __future__ import annotations

import pathlib

from autoforecast._exact_json import ExactJSONError, load_exact_json
from autoforecast.panel import Panel
from provenance.lineage import directory_digest

__all__ = ["MANIFEST_NAME", "REQUIRED_MANIFEST_FIELDS", "LatentActualsError", "load"]

MANIFEST_NAME = "latent_manifest.json"

REQUIRED_MANIFEST_FIELDS = ("method", "source", "estimand", "content_sha256", "registered_on")
"""What a latent-actuals artifact must state about itself.

`method` and `source` are the interrogable claims -- "how were these latents measured" is the
first question any number scored against them must survive. `content_sha256` binds the bytes.
`registered_on` dates the governance decision. A field left blank is a claim left unmade.
"""


class LatentActualsError(ValueError):
    """The artifact cannot support claim-bearing latent-demand evaluation as presented."""


def load(artifact_dir: str | pathlib.Path, *, series_col: str = "series_id",
         time_col: str = "date", value_col: str = "latent_demand") -> Panel:
    """Verify a latent-actuals artifact and return the ONLY panel latent scoring accepts.

    Args:
        artifact_dir: Artifact ROOT: `latent_manifest.json` beside an `actuals/` directory of
            parquet fragments. Beside, not inside -- Spark reads a parquet directory whole,
            and a manifest inside it would be read as data.
        series_col: Series identifier column in the artifact.
        time_col: Date column.
        value_col: The latent-demand estimate column -- the target the panel declares.

    Returns:
        A validated `Panel` with `target_estimand='unconstrained_demand'` pointing at the
        verified artifact. Handing this to `score_forecast` or `wrmsse` is what makes a
        latent-demand score claim-bearing; nothing else constructs such a panel.

    Raises:
        LatentActualsError: no manifest, missing or blank manifest fields, a wrong estimand,
            or artifact bytes that do not match the registered content hash.
    """
    root = pathlib.Path(artifact_dir)
    data = root / "actuals"
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        raise LatentActualsError(
            f"{root} has no {MANIFEST_NAME}. Latent actuals are estimates, not observations; "
            "an artifact that does not state how they were measured cannot support a "
            "claim-bearing score")
    try:
        manifest = load_exact_json(manifest_path)
    except ExactJSONError as exc:
        raise LatentActualsError(f"{manifest_path} is not valid JSON: {exc}") from exc
    if type(manifest) is not dict:
        raise LatentActualsError("latent manifest must be one JSON object")
    missing = [f for f in REQUIRED_MANIFEST_FIELDS
               if type(manifest.get(f)) is not str or not manifest[f].strip()]
    if missing:
        raise LatentActualsError(
            f"latent manifest lacks {missing}; a field left blank is a claim left unmade")
    if manifest["estimand"] != "unconstrained_demand":
        raise LatentActualsError(
            f"latent manifest declares estimand {manifest['estimand']!r}; this loader exists "
            "for unconstrained_demand and nothing else")
    if not data.exists():
        raise LatentActualsError(f"{root} has no actuals/ parquet directory")
    digest = directory_digest(data, suffixes=(".parquet",))
    if digest != manifest["content_sha256"]:
        raise LatentActualsError(
            f"artifact bytes {digest[:12]} do not match the registered content "
            f"{str(manifest['content_sha256'])[:12]}; an edited artifact must stop loading "
            "rather than quietly scoring")
    return _validated_latent_panel(data, series_col, time_col, value_col)


def _validated_latent_panel(root: pathlib.Path, series_col: str, time_col: str,
                            value_col: str) -> Panel:
    """Construct and schema-validate the latent panel.

    The estimand check in `Panel.validate` requires censoring evidence for
    unconstrained_demand -- correct for TRAINING panels, where the latents must be recovered
    through the observation contract. A governed ACTUALS artifact is the other direction: the
    latents are already recovered, the manifest is their provenance, and demanding an
    observation column here would demand evidence of a recovery that already happened. The
    panel is therefore built as sales-estimand for schema validation and re-labelled, with
    this function as the single, documented place that happens.
    """
    checked = Panel(path=str(root), series_col=series_col, time_col=time_col,
                    target_col=value_col, season_length=7).validate()
    import dataclasses
    return dataclasses.replace(checked, target_estimand="unconstrained_demand")
