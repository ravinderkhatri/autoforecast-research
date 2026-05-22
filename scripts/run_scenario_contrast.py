"""Prospective scenario contrast from a saved checkpoint -- the runnable product surface.

WHAT THIS ANSWERS: "holding the rest of the plan fixed, setting the commercial bundle to its
reference moves expected sales by X" -- a MODELLED DIFFERENCE under a fitted model, not a causal
effect, and every number ships with the campaign-bootstrap spread and the overlap check that say
how much evidence stands behind it (`forecast/scenario/contrast.py` refuses to produce one
without them).

Until the checkpoint existed this machinery had no entry point: the model lived and died inside
`fit_predict`'s process, so `attribution: IMPLEMENTED` was true of the library and false of the
product. This script is the wiring -- load the fitted model, rebuild the inference assembly from
the panel at the checkpoint's origin, and contrast the commercial bundle for a query-scoped set
of series (`attribution_scope='query'`: the series asked about, never the catalogue).

WHAT IT REFUSES, deliberately:
  * a panel with no campaign labels -- the bootstrap resamples EPISODES, and M5 has none, so on
    M5 this script demonstrates the refusal rather than fabricating an interval;
  * an origin at or after the campaign start (pre-treatment is checked in `contrast()`);
  * a run without an observed driver range -- the overlap check needs to know what discount
    depths campaigns have actually reached, and that is knowledge about the client's history,
    not something to silently derive from the plan being asked about.

Run:
  JAVA_HOME=... PYSPARK_PYTHON=... python3 -u scripts/run_scenario_contrast.py \
      --checkpoint gates/experiments/3_yearly_stride/model.pt \
      --panel data/m5_parquet --price-col sell_price \
      --campaign-col <episode column> --campaign-start 2016-05-01 \
      --observed-depth-low 0.0 --observed-depth-high 0.25 \
      --planned-depth 0.10 --out /tmp/contrast.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from autoforecast import resources
from autoforecast.adapters import mft as ours
from autoforecast.panel import Panel
from forecast.scenario.contrast import contrast
from pyspark import SparkContext
from pyspark.sql import SparkSession

from provenance.lineage import directory_digest


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--panel", required=True)
    p.add_argument("--series-col", default=None,
                   help="optional assertion against the checkpoint's saved panel role")
    p.add_argument("--time-col", default=None,
                   help="optional assertion against the checkpoint's saved panel role")
    p.add_argument("--target-col", default=None,
                   help="optional assertion against the checkpoint's saved panel role")
    p.add_argument("--exposure-col", default=None,
                   help="optional assertion against the checkpoint's saved panel role")
    p.add_argument("--oos-col", default=None,
                   help="optional assertion against the checkpoint's saved raw OOS role")
    p.add_argument("--price-col", default=None,
                   help="optional assertion against the checkpoint's saved price role")
    p.add_argument("--promo-col", default=None)
    p.add_argument("--campaign-col", default=None,
                   help="column labelling campaign EPISODES; the bootstrap resamples these. "
                        "Without one the script refuses: an interval needs replications, and "
                        "days of one campaign are not replications of it")
    p.add_argument("--campaign-start", required=True,
                   help="ISO date the campaign begins; must be after the checkpoint's origin")
    p.add_argument("--observed-depth-low", type=float, required=True)
    p.add_argument("--observed-depth-high", type=float, required=True,
                   help="discount-depth range campaigns have ACTUALLY reached -- measured from "
                        "the client's history, because overlap against it is what separates "
                        "evidence from extrapolation")
    p.add_argument("--planned-depth", type=float, default=0.10)
    p.add_argument("--window", default="event_support",
                   choices=("event_support", "horizon"),
                   help="which days the contrast integrates; event_support uses the "
                        "checkpoint's event ceiling around --campaign-start and labels the "
                        "result GROSS when the support outruns the horizon")
    p.add_argument("--n-series", type=int, default=500,
                   help="query scope: how many series (sorted ids) the contrast covers")
    p.add_argument("--n-draws", type=int, default=8)
    p.add_argument("--gate6-evidence", default=None,
                   help="path to a forecast.scenario.gate6 evidence record (JSON). Its "
                        "episode_validated verdict -- the ONLY sanctioned producer -- flows "
                        "into the contrast; without it the contrast stays unquotable")
    p.add_argument("--out", default=None, help="where to write the JSON record")
    return p.parse_args(argv)


def _commercial_scenario_bundle(
    known: torch.Tensor, layout: ours.KnownFuture,
    price_response_mode: str = "soft_penalty_v1",
):
    """Build the live commercial intervention and its aligned no-promotion references."""
    live = tuple(c for c in ours.COMMERCIAL_BUNDLE if c in layout.declared_drivers)
    if not {"price", "discount_depth"}.issubset(live):
        raise SystemExit(
            "the checkpoint needs declared price and discount_depth drivers; without historical "
            "price, --planned-depth has no list-price reference and would leave the price "
            "counterfactual unchanged")
    price = known[..., layout.index("price")]
    depth = known[..., layout.index("discount_depth")]
    have_price = known[..., layout.index("price_available")] > 0.5
    have_depth = known[..., layout.index("discount_depth_available")] > 0.5
    if price_response_mode == "monotone_raw_v1":
        reference_supported = (
            have_price & have_depth & torch.isfinite(price) & (price > 0.0)
            & torch.isfinite(depth) & (depth < 1.0)
        )
        if not bool(reference_supported.all()):
            unsupported = int((~reference_supported).sum())
            raise SystemExit(
                "monotone_raw_v1 scenario refused: "
                f"{unsupported} cell-step(s) lack a recoverable frozen historical price "
                "reference. Synthetic list prices are not substituted for missing support")
    list_price = torch.where(have_depth & (depth < 1.0),
                             price / (1.0 - depth).clamp_min(1e-6), price)
    zero = torch.zeros_like(depth)
    reference_by_driver = {"price": list_price, "discount_depth": zero, "promo": zero}
    reference = torch.stack([reference_by_driver[driver] for driver in live], dim=-1)
    return live, ours.intervention_bundle(
        layout, "commercial", live, reference_values=reference)


def main(argv: list[str]) -> int:
    args = parse(argv)
    if args.campaign_col is None:
        print("REFUSED: no --campaign-col. The spread resamples whole campaigns because the "
              "effective sample size is EPISODES; without labels the alternative is a "
              "paired-draw spread, which is narrow by construction and is not evidence. "
              "(M5 carries no campaign structure -- this refusal is the honest M5 outcome.)")
        return 2

    model, record = ours.load_checkpoint(args.checkpoint)
    run = record["run"]
    print(f"checkpoint: origin {run['origin']}, horizon {run['horizon']}, "
          f"{len(run['series_ids']):,} series", flush=True)

    spark, owned = resources._acquire_embedded_spark_runtime(
        SparkSession, SparkContext, app_name="contrast",
        profile_factory=lambda: resources.spark_profile(python_udfs=True))
    try:
        if owned:
            spark.sparkContext.setLogLevel("ERROR")
        return _run_with_spark(args, model, record, run, spark)
    finally:
        if owned:
            spark.stop()


def _run_with_spark(args, model, record, run, spark: SparkSession) -> int:
    """Execute scenario reconstruction while respecting the caller's Spark ownership."""
    saved_panel = run.get("panel")
    if not isinstance(saved_panel, dict):
        raise SystemExit(
            "checkpoint predates the saved panel-role contract; retrain it before scenario "
            "inference. Guessing roles can rebuild a different encoder under the same weights")
    expected_panel_hash = run.get("panel_parquet_sha256")
    if not isinstance(expected_panel_hash, str) or len(expected_panel_hash) != 64:
        raise SystemExit(
            "checkpoint predates panel-content binding; retrain it before scenario inference. "
            "Matching column roles do not prove the encoder reads the data the weights saw")
    actual_panel_hash = directory_digest(args.panel, suffixes=(".parquet",))
    if actual_panel_hash != expected_panel_hash:
        raise SystemExit(
            f"scenario panel content hash {actual_panel_hash[:12]} differs from checkpoint "
            f"{expected_panel_hash[:12]}. This entry point reconstructs the checkpoint origin; "
            "using refreshed history or realised future plans would change its inputs under the "
            "same weights. Retrain on that snapshot or use the exact training panel")
    panel_record = dict(saved_panel)
    panel_record["path"] = args.panel
    # Column flags are assertions/overrides only for old command lines. The fitted checkpoint is
    # authoritative: accepting a different OOS/exposure/price role here would apply the weights
    # to a different feature meaning while all tensor shapes still matched.
    for arg_name, role in (
        ("series_col", "series_col"), ("time_col", "time_col"),
        ("target_col", "target_col"), ("exposure_col", "exposure_col"),
        ("oos_col", "oos_col"), ("price_col", "price_col"), ("promo_col", "promo_col"),
    ):
        asserted = getattr(args, arg_name)
        if asserted is not None and asserted != panel_record.get(role):
            raise SystemExit(
                f"--{arg_name.replace('_', '-')}={asserted!r} disagrees with checkpoint "
                f"{role}={panel_record.get(role)!r}; use the training data contract")
    # fit_predict permits an explicit category override. Preserve the effective role rather than
    # the Panel's unused default so the cold-start prior is reconstructed identically.
    panel_record["category_col"] = run["category_col"]
    panel = Panel.from_record(panel_record).validate()

    work = pathlib.Path(args.out or "/tmp/contrast").parent / "_contrast_features"
    category_of = (ours._series_levels(panel, spark, panel.category_col)
                   if panel.category_col else None)
    assembly = ours._assemble(
        panel, spark, run["origin"], run["horizon"], run["encoder_days"],
        [tuple(e) for e in run["events"]], run["event_max_support"], work,
        tuple(tuple(c) for c in run["cycles"]), category_of=category_of)
    if list(assembly.known.names) != list(run["known_names"]):
        raise SystemExit(
            f"the panel now yields known-future columns {assembly.known.names} but the model "
            f"trained on {tuple(run['known_names'])}; a contrast on shifted columns intervenes "
            "on the wrong driver")

    # Query scope: the first --n-series ids present in BOTH the checkpoint and the assembly.
    trained = {sid: i for i, sid in enumerate(run["series_ids"])}
    usable = [sid for sid in assembly.series_ids if sid in trained][:args.n_series]
    if not usable:
        raise SystemExit("no overlap between the checkpoint's series and the panel's")
    take = torch.tensor([assembly.series_ids.index(sid) for sid in usable], dtype=torch.long)
    rows = torch.tensor([trained[sid] for sid in usable], dtype=torch.long)
    t = record["run"]["tensors"]
    if t is None:
        raise SystemExit(
            "this checkpoint was trained SHARDED and carries no per-series conditioning "
            "tensors (the bounded-memory contract). Rebuild the query shard's inputs by "
            "predicate before contrasting; a contrast on tensors that were never persisted "
            "cannot be run from this file alone")
    inputs = dict(
        history=assembly.history[take], observed=assembly.observed[take],
        censored=assembly.censored[take],
        attributes=t["attributes"][rows], series_index=rows,
        text=None if t["text"] is None else t["text"][rows],
        known_future=assembly.known.values[take],
        baseline_logit=assembly.baseline_logit[take],
        baseline_log_size=assembly.baseline_log_size[take],
        group_index=t["group_index"][rows], store_index=t["store_index"][rows],
        product_index=t["product_index"][rows], n_draws=args.n_draws,
        commercial=None if assembly.commercial is None else assembly.commercial[take],
        commercial_mask=(None if assembly.commercial_mask is None
                         else assembly.commercial_mask[take]),
        event_distance=(None if assembly.event_distance is None
                        else assembly.event_distance[take]),
        event_present=(None if assembly.event_present is None
                       else assembly.event_present[take]))

    known = inputs["known_future"]
    layout = assembly.known
    # "No promotion" per cell: LIST price recovered from the plan (price / (1 - depth) where the
    # depth is known), depth zero, promo zero. A scalar reference cannot say this -- zero would
    # price the catalogue at free.
    live, bundle = _commercial_scenario_bundle(
        known, layout, price_response_mode=model.price_response_mode)

    campaign_of = ours._series_levels(panel, spark, args.campaign_col)
    labels = sorted({v for sid in usable if (v := campaign_of.get(sid)) is not None})
    at = {v: i for i, v in enumerate(labels)}
    campaign_index = torch.tensor([at.get(campaign_of.get(sid), -1) for sid in usable])
    if bool((campaign_index < 0).any()):
        raise SystemExit("some queried series carry no campaign label; label them or narrow "
                         "the query -- an unlabeled cell cannot be resampled honestly")

    # Checkpoint ids live in the full training index. A small query can therefore contain labels
    # such as [0, 100000] while owning only two groups. Draw only those two paths and pass the
    # matching dense ids onward; max+1 allocates every absent group in between.
    inputs["group_index"], group_labels = ours._dense_group_index(inputs["group_index"])
    n_groups = len(group_labels)
    # Factor models reuse the same sampled systemic paths in both arms.  A checkpoint resolved to
    # plain bottom-up has no shared latent variable; `draw_factor_paths` returns a canonical empty
    # tensor which the same contrast API can pass through without manufacturing dependence.
    path = model.draw_factor_paths(
        n_groups, run["horizon"], args.n_draws,
        generator=torch.Generator().manual_seed(run["seed"]))
    fn = ours.expected_demand_fn(model, inputs)
    episode_validated = False
    if args.gate6_evidence:
        from forecast.scenario.gate6 import load_evidence
        evidence = load_evidence(args.gate6_evidence)
        episode_validated = bool(evidence["episode_validated"])
        print(f"gate6 evidence: {len(evidence['campaign_ids'])} held-out campaigns, "
              f"validated={episode_validated}"
              + (f", notes: {evidence['notes']}" if evidence["notes"] else ""), flush=True)
    result = contrast(
        fn, known, bundle, path,
        episode_validated=episode_validated,
        origin=run["origin"], campaign_start=args.campaign_start,
        campaign_index=campaign_index,
        observed_driver_range=(args.observed_depth_low, args.observed_depth_high),
        planned_driver_range=(0.0, args.planned_depth),
        seed=run["seed"],
        window=args.window,
        event_support=((-run["event_max_support"], run["event_max_support"])
                       if args.window == "event_support" else None))
    print(result)
    for note in result.notes:
        print(f"  NOTE: {note}")
    if args.out:
        output = result.as_record()
        output["drivers"] = list(live)
        pathlib.Path(args.out).write_text(json.dumps(output, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
