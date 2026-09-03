# AutoForecast

Run reproducible forecasting experiments on hourly, daily, weekly, monthly, or yearly time series.
Daily is the default. AutoForecast validates
input data, compares models on validation periods, refits the selected model, and scores a
separate test period. Each run retains its data identities, configuration, forecasts, and metrics.

Start with CSV or Parquet. Column names and series keys are configurable; M5 is an optional
benchmark, not a required data format. Execution is local by default. Shared storage, an existing
remote cluster and external tracking are used only when explicitly configured.

![AutoForecast workflow: bring time-series data, validate and freeze a benchmark, compare models, refit, score a separate test period, and inspect verified results.](assets/autoforecast-overview.svg)

## Requirements and installation

- Python 3.12 or 3.13 on Linux (x86_64) or macOS (Apple Silicon).
  Intel macOS is not supported by the bundled Ray version.
- Java 17 for the local Spark preparation and evaluation runtime.
- Enough local disk and memory for your dataset. The quickstart uses two small series.

```bash
git clone https://github.com/ravinderkhatri/autoforecast-research.git
cd autoforecast-research
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Set `JAVA_HOME` to your Java 17 installation if it is not already configured. For a small local
run, limit Spark to two cores and a 1 GB JVM heap:

```bash
export FORECAST_SPARK_CORES=2
export FORECAST_SPARK_MEMORY=1g
```

The standard install includes data preparation, result plots, the model catalog, HPO with
Optuna and Ray, and MLflow tracking libraries. It also includes seasonal-naive and last-value
baselines plus AutoETS, AutoTheta and Croston. These models need neither a GPU nor a model
download. External tracking is used only when you configure it.

Backward-compatibility shims are not provided. Saved records must satisfy their current
contract; obsolete formats and missing required fields are rejected. Recreate unsupported
runs with the current API rather than modifying their saved evidence.

### Choose additional models

The standard install is the lighter option and already includes basic and statistical models.
Use Python package extras to add selected model families or the full compatible stack;
every option keeps the same workflow, HPO and tracking capabilities. Run these commands from
the cloned repository:

| Installation | Command |
|---|---|
| Standard: basic and statistical models, HPO and tracking | `python -m pip install .` |
| All compatible model families | `python -m pip install '.[all-models]'` |
| Add global LightGBM | `python -m pip install '.[shallow]'` |
| Add MFT, SFT, HMFNet and PACT | `python -m pip install '.[custom-torch]'` |
| Add TFT, TFT-Q and TFT-HNB | `python -m pip install '.[tft]'` |
| Add the AutoGluon model collection | `python -m pip install '.[autogluon-zoo]'` |
| Add TimesFM-HNB and TimesFM-V8 | `python -m pip install '.[timesfm3]'` |

On macOS, the LightGBM-based options (`shallow` and `autogluon-zoo`) also need OpenMP:
run `brew install libomp`, as described in the
[LightGBM installation instructions](https://github.com/lightgbm-org/LightGBM/blob/master/python-package/README.rst#install-from-pypi).

Install all compatible model families together:

```bash
python -m pip install '.[all-models]'
```

The `all-models` extra combines `shallow`, `custom-torch`, `tft`, `autogluon-zoo`, and `timesfm3`.
Full catalog coverage also needs a second environment for standalone TimesFM 2.5. It and the
TimesFM 3 runtime use different versions of the same Python distribution and cannot be
installed together. From the same checkout, create that separate environment:

```bash
python3.12 -m venv ../autoforecast-timesfm2-env
../autoforecast-timesfm2-env/bin/python -m pip install '.[timesfm]'
```

Installing a runtime does not select a model or download its pretrained weights. Choose the
model in the experiment configuration; its data, frequency, device and weight requirements
still apply. The TimesFM extras install pinned source revisions and require Git.

These are standard [Python packaging extras](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#dependencies-optional-dependencies),
not custom installer flags. With uv, use `uv sync --extra all-models`; do not use
`--all-extras`, which would request both incompatible TimesFM versions.

## Run the example

Create an experiment outside this checkout:

```bash
cd ..
autoforecast init-example my-experiment
autoforecast freeze --project-root my-experiment --source data
```

The default example creates 84 observations: two `(shop, sku)` series across 42 days. Preparation retains
both series, with no exclusions. Each seven-day forecast scores the same 14 cells. `freeze`
prints an absolute path ending in `benchmark.json`; use that exact path:

```bash
autoforecast run --project-root my-experiment \
  --benchmark /absolute/path/printed/by/freeze/benchmark.json
```

The run compares seasonal-naive and last-value forecasts at two validation origins. It selects
a model using validation scores, refits through February 4, and tests on February 5–11. Test
labels are supplied only to the final scorer. Running the same command again resumes the same
campaign. Changed inputs or settings create a different identity.

Use `--frequency` to create the same small experiment at another cadence:

```bash
autoforecast init-example hourly-experiment --frequency H
autoforecast init-example monthly-experiment --frequency MS
autoforecast init-example yearly-experiment --frequency YS
```

Each example contains two series and 42 periods. Its generated configuration uses a synthetic
seven-period season and seven-period horizons; those numbers count observations, not days.

## Choose a frequency

Set `dataset.panel.frequency` explicitly for non-daily data. The same declaration applies to
the input keys, validation origins, final cutoff, forecast keys, scoring, and result export.

| Frequency | Input labels | One forecast period |
|---|---|---|
| `H` | Exact-hour timestamps, such as `2024-01-01T06:00:00` | One hour |
| `D` (default) | ISO dates, such as `2024-01-01` | One day |
| `W` | ISO dates on a consistent weekday | Seven days, preserving that weekday |
| `M` or `ME` | Month-end dates | Next calendar month end |
| `MS` | First day of each month | Next calendar month start |
| `Y` or `YE` | December 31 | Next calendar year end |
| `YS` | January 1 | Next calendar year start |

Monthly and yearly boundaries use the calendar. For example, a month-end forecast after
January 31, 2024 targets February 29 and then March 31. Off-boundary dates are rejected instead
of shifted. Hourly timestamps must be timezone-naive and have zero minutes, seconds, and
fractional seconds. Normalize timezone-aware data to your intended timeline before importing it.
The workflow does not infer a frequency, resample observations, or change their labels.

Horizons and `season_length` always count periods. For example, hourly data with a daily seasonal
pattern uses `season_length: 24`; monthly data with an annual pattern uses `season_length: 12`.

## Use your own data

Create the example project as a template, then replace its data and edit these files:

| File | What to change |
|---|---|
| `dataset.yaml` | Dataset ID; series, time, and target columns; frequency; season length; optional feature roles. |
| `recipe.json` | Source keys, history starts, origins, horizons, cutoff, target unit, missing-value policies, and metric. |
| `campaign.yaml` | Matching dates, candidates, scoring metric, compute limits, and training budgets. |
| `models/*/model.yaml` | Model profiles selected by the campaign. |

For example, a CSV can use these columns:

```csv
store,product,date,units
north,001,2024-01-01,0
north,001,2024-01-02,3
south,002,2024-01-01,8
south,002,2024-01-02,5
```

Convert your complete history to typed Parquet:

```bash
autoforecast import-csv observations.csv my-experiment/custom-data/observations.parquet \
  --id-columns store product --date-column date --target-column units
```

CSV keys remain strings, so `001` stays distinct from `1`. Non-hourly dates must be `YYYY-MM-DD`.
For hourly CSV data, pass `--frequency H` and use timestamps such as `2024-01-01T06:00:00`.
For other non-daily data, pass the matching frequency, such as `--frequency MS`.
The importer selects the mapped columns and records excluded columns, source checksum,
date range, row count, and series count in the Parquet file's metadata. Preparation records the
output checksum as part of its source identity. The importer refuses
conflicting output files. Use typed Parquet directly when you need additional features or
integral keys. The CSV convenience importer reads the selected observations into memory.

Mapped CSV headers and all Parquet column names must match `[A-Za-z_][A-Za-z0-9_]*`.
For example, rename `meter.id` to `meter_id` and `daily usage` to `daily_usage` before import.
Column names must be distinct ignoring case; names beginning with `__af_` in any case are
reserved for prepared data. Series key values must be nonblank;
string IDs such as `001` and integral IDs such as `0` are valid.

Map this example in `dataset.yaml`:

```yaml
schema_version: 1
dataset:
  id: my-timeseries-data
  panel:
    series_col: store
    time_col: date
    target_col: units
    frequency: D
    target_estimand: observed_nonnegative
    season_length: 7
```

In `recipe.json`, set `source_keys` to `["store", "product"]` and `target_unit` to `units`.
`source_keys` defines the complete series identity; preparation creates a reversible composite
ID. Set `training_start`, `validation_origins`, `validation_horizon`, `final_train_cutoff`, and
`final_test_horizon` to time points and period counts supported by your history. Copy those temporal values
into `campaign.yaml`. Validation windows must end by the final training cutoff. The test window
starts at the next period. Update the dataset source labels in the campaign to describe your data.

```bash
autoforecast freeze --project-root my-experiment --source custom-data
```

Use the newly printed benchmark path with `autoforecast run`. The two-row-per-series excerpt
above only illustrates the format; training requires complete history at the declared frequency over your chosen
training, validation, and test windows.

### Data rules

- The release workflow supports finite, non-negative, observed values at the declared frequency. Fractional values
  work with the baselines; count models can impose integer-target requirements.
- Hourly Parquet keys use timezone-naive timestamps. Other cadences use dates; midnight
  timestamps are accepted only when conversion preserves the full time key. Each series/time
  must appear once. Hourly values on the same day remain distinct observations. Legacy INT96
  timestamps require an embedded Arrow schema declaring timezone-naive values; otherwise,
  convert the source to typed dates or timezone-naive timestamps before importing it.
- The cohort is fixed from series present by the first validation origin in the selection
  source. The manifest reconciles all source series, exclusions, and the retained cohort.
- New studies use recipe version 2 and frozen manifest version 3. `actual_missing` defaults
  to `error`: even one absent row or null actual in a scoring window rejects preparation
  and scoring. Correct the source data before importing it.
- `history_missing` is `preserve`. Preparation preserves null targets and records original
  missingness; it never fills targets with zero or another value. Models that need complete
  history require caller-prepared data. An explicitly observed zero is valid. NaN, infinity,
  negative targets, duplicate rows, and invalid keys are always errors.
- Set `actual_missing` to `mask` explicitly to score only observed actuals. Forecasts must
  still cover every expected cell. Coverage reports include expected, originally missing,
  excluded, and scored counts for every series; `zero_filled` is always zero.
- With `mask`, the weighted score is labeled **masked WRMSSE** and keeps every series' frozen
  weight. If any retained series has no observed actuals, WRMSSE raises an error. Missing or
  zero historical scales remain unavailable; they are never replaced with invented scales.
- The same frozen policy controls tuning, checkpoint selection, validation, and final scoring.
  Old version-2 manifests and `zero` policies are rejected. Re-register from source data;
  old results retain their original meaning and are not silently migrated.
- Every model must meet its minimum history and window requirements for the whole frozen
  cohort before fitting. A model cannot silently drop short series.
- Only recipe version 2 and study manifest version 3 are supported.
- New series default to an error. The explicit `exclude_after_first_origin` recipe policy
  excludes series absent from the initial cohort and records the count.
- `history_policy: fixed` requires overlapping history to agree across snapshots.
  `revised_snapshot` permits declared corrections and creates a new benchmark identity.
- Historical features use `past_only`. Prepared future-known features require declared
  availability evidence and an observation-state column. Unknown future availability is refused.

Separate snapshots can be supplied explicitly:

```bash
autoforecast freeze --project-root my-experiment \
  --selection data/selection --final-training data/training \
  --validation-score data/validation-actuals --final-score data/test-actuals
```

Parquet directories must contain `.parquet` parts, optionally under Hive partition directories.
Hidden completion markers are ignored. Visible sidecars, summary metadata, streaming logs,
links, and special files are refused so every reader-visible observation is authenticated.

Signed targets, learned imputation and raw warehouse feeds are outside this CLI release
workflow. This CLI route does not provision or execute cloud jobs. The custom-model campaign
API described below can connect to an explicitly supplied existing Ray cluster. Use `observed_nonnegative` for measured quantities,
including fractional values such as energy use. The two built-in baselines support this target.
Use `sales_given_availability` for observed retail sales. Optional retail models declare their
supported targets and reject incompatible inputs; none of these observed-target contracts
infers unobserved demand.

## Freeze a reusable study

The four logical data roles distinguish fitting history from scoring actuals. Validation uses
`validation_score`, not the refit target. Test labels stay outside fitting and model selection.
`--source` assigns one input snapshot to all four roles. Identical prepared roles share storage;
the manifest records their logical identities and physical paths.

A version-2 recipe must declare `metric` (`rmse`, `mae`, `wape`, or `wrmsse`). To give retained
series different usable history starts, add an exact source-key map and a derivation rule:

```json
"series_starts": {
  "rule": "Usable history declared from records available at the first validation origin",
  "values": [
    {"keys": {"shop": "a|b", "sku": 1}, "start": "2024-01-01"},
    {"keys": {"shop": "a", "sku": 2}, "start": "2024-01-08"}
  ]
}
```

Declare every retained series exactly once. Starts must lie between `training_start` and the
first validation origin and align with the frequency. Preparation never pads before them.
Without this map, every series uses `training_start`. The first observed row is not treated as
proof of a launch date. Feed completeness, launch/delist rules, and missingness reasons remain
caller responsibilities.

The manifest's `study_id` binds the cohort, declared starts, prepared values, original missing
flags, windows, policies, and metric definition. It is separate from model-run, HPO-study, and
MLflow identifiers. Studies support profile defaults, manual parameters, inner HPO, and verified
HPO reuse. Reuse a verified study by ID:

```bash
autoforecast run --project-root my-experiment --study-id STUDY_ID
```

IDs resolve existing manifests under `.autoforecast/benchmarks`; they never create or infer a
study. Creation requires explicit input data and a recipe. Repeating `freeze` with
`--study-id STUDY_ID` verifies that the inputs reproduce that contract. Changed labels or rules
require a new identity; existing artifacts are never overwritten.

RMSE, MAE, and WAPE aggregate the available cells under the declared policy. A series with no
scoreable cells has an unavailable score, and an empty aggregate is also unavailable. WAPE is
unavailable when total actual volume is zero. An unavailable validation score cannot select a
model; an unavailable held-out test score is reported without changing the selection.

Study `wrmsse` uses a single-level weighted mean of per-series RMSSE. It does not use M5's
hierarchy. Supply `--scoring-references references.parquet` with one row for every retained
series and every tuning, validation, and test boundary:

| Column | Value |
|---|---|
| `keys` | Struct containing the recipe's source-key names and exact types |
| `stage` | `tuning`, `validation`, or `test` |
| `origin` | Canonical origin string at the declared frequency |
| `weight` | Finite non-negative weight; positive total at each boundary |
| `scale` | Mean squared naive-change scale; zero or null means undefined |

Weights and scales are supplied by the caller and frozen with the study. The score is
`sum(weight * sqrt(MSE / scale)) / sum(weight)`. Under masking, each series' MSE uses its
observed cells. If any retained series has no scoreable actuals, WRMSSE raises an error.
An undefined scale makes WRMSSE unavailable. Weights are never redistributed.

The `sft`, `hmfnet`, `pact`, `tft_q`, `tft_hnb`, `timesfm_hnb`, and `timesfm_v8`
adapters select checkpoints with the study metric, including its original missing-value mask
and any supplied weights and scales. Training loss remains model-specific. Final refit uses
the selected update budget without test labels. These native adapters retain their daily-data,
target, feature, and history requirements; use a compatible adapter for other frequencies.

### Use a study from Python

Install the package and start a supported Spark session. The study interface
works in notebooks, scripts and jobs. It borrows that session and never stops it. Set the
experiment project once; its existing `.autoforecast/benchmarks` directory is the local
study registry:

```python
from autoforecast.study import load_study
from autoforecast.project import experiment_project

with experiment_project("/absolute/path/to/my-experiment"):
    with load_study(study_id) as study:
        training = study.training_data
        validation_actuals = study.validation_data
        refit = study.refit_data
        testing_actuals = study.testing_data

        # Fit your existing model on training, then refit on refit.
        # Keep testing_actuals outside all fitting and model selection.
        # Produce two Forecast objects as shown in the runnable example below.
        result = study.evaluate_forecasts(
            validation_forecasts=validation_forecasts,
            testing_forecasts=testing_forecasts,
            model_id="my-model",
            forecast_accuracy_breakdown="per_step",  # Default: accuracy at each forecast step.
            output_dir="outputs/demand-baseline/scoring",  # Parent for a fresh evaluation run.
        )
        print(result["validation"]["score"])
        print(result["test"]["score"])
        print(result["result_path"])
```

`forecast_accuracy_breakdown` controls accuracy diagnostics for each stage:

| Setting | Metrics reported alongside the full-horizon overall score |
|---|---|
| `"per_step"` (default) | Each forecast step separately: H1, H2, ..., Hn |
| `"cumulative"` | Each prefix: H1–H1, H1–H2, ..., H1–Hn |
| `"both"` | Both separately labelled breakdowns |
| `"none"` | Overall metrics only |

Hn follows each forecast's horizon and declared frequency. For hourly data H3 is three hours
ahead; for weekly data it is three weeks ahead. All supported study frequencies use the same
calendar rules as the forecasts. Validation and test horizons can differ.

Read `result["test"]["score"]["forecast_accuracy_breakdown"]["per_step"]` for the default
test breakdown. Each row identifies its lead range, target-time range, metrics and coverage.
The breakdown reports MAE, RMSE, WAPE and bias, plus WRMSSE for studies configured with its
frozen weights and scales. Cumulative scores use the underlying cell errors through that lead;
they do not average per-step scores or score summed demand. The final cumulative score equals
the overall score. The overall metric definitions and missing-actual policy stay fixed.

Strict studies require complete actuals. With explicit masking, a diagnostic WRMSSE is
unavailable when any retained series has no observed actual in that lead range; its status,
reason and coverage are returned. Undefined scales also yield unavailable WRMSSE. Weights are
never redistributed. The existing full-horizon error for an entirely unobserved series remains.
All requested breakdowns and their supporting statistics are saved with the evaluation result.
Each row's status and unavailable reason describe the study's selected metric. Other metrics
can be unavailable independently; for example, WAPE is unavailable when actual volume is zero.

For a standalone registry, pass `registry_root=...` to `load_study`. Pass `spark=spark` when
the caller's session is not active. An absent ID in an existing registry raises
`FileNotFoundError`; access failures and changed artifacts remain separate errors.
Loading never creates a study.

For setup shared across notebook cells or functions, bind these arguments once with standard Python:

```python
from functools import partial
from autoforecast.study import load_study as _load_study

load_study = partial(_load_study, registry_root="/path/to/registry", spark=spark)
study = load_study(study_id)  # Later cells need only the ID; close study when finished.
```

All four data properties return Spark DataFrames over verified private snapshots, sliced to
the intended stage. They keep source key names, the frozen internal series ID, and
`__af_original_missing`. `study.panel` exposes the column names, cadence and target estimand;
`study.series_mapping` and `study.series_starts` provide exact source keys, internal IDs,
and `__af_history_start`. `study.policies`, `study.metric_definition`, and
`study.scoring_references` expose the frozen missingness and scoring settings.

| Inclusive range | Meaning |
|---|---|
| `training_date_range` | Overall history start through the initial training cutoff |
| `validation_date_range` | Validation forecast and actual comparison window |
| `refit_date_range` | Overall history start through the final refit cutoff |
| `testing_date_range` | Full planned final evaluation window |
| `forecasting_date_range` | Alias of `testing_date_range` |

Ranges contain Python dates, or datetimes for hourly studies. Each series also retains its
own history start. Missing actuals never shorten a range. This interface requires exactly
one validation origin; it rejects multi-origin studies explicitly. Study settings cannot be
edited in place. Use the dataframes inside the study's lifetime; `close()` or leaving its
context releases the private snapshots.

Predictions use the existing `autoforecast.panel.Forecast` contract. Its Parquet rows must
contain `series_id`, `date`, and finite nonnegative numeric `yhat`. For hourly predictions,
use timezone-naive `TIMESTAMP_NTZ`; other cadences use `DATE`. Map original keys through
`study.series_mapping`; do not construct internal IDs by concatenating keys. Supply the
correct origin, horizon, cadence, target estimand, series count and emitted-cell count in
each Forecast. Every frozen series/date cell is required, even where actuals are missing.
Duplicate, missing, extra, or wrong-stage predictions are rejected.

Run the complete [study example](scripts/study_example.py) from outside the checkout:

```bash
python -m scripts.study_example /absolute/path/to/new-study-example
```

It creates two tiny series with unequal history starts, freezes them using `freeze_benchmark`,
loads the generated ID, accesses all roles and ranges, converts ordinary Spark predictions
to Forecast objects, and scores both stages. It performs no model training or paid compute.

By default, each check creates `.autoforecast/study-runs/<study-id>/evaluation-<UTC>-<UUID>/`
beside the benchmark registry. `output_dir` sets a different parent folder, and
`result["result_path"]` identifies the saved result. Each run retains
forecasts, per-series scores, coverage, study/model/run IDs and artifact hashes. The existing
scorer controls unavailable scores and WRMSSE weights. The result records verified input
identities; it cannot prove which data external model code trained on. The result records
`training_provenance` as `unverified_external`. This forecast check does not register a model
or create a managed campaign result for `read_results`.

### Tune within a study

Declare inner tuning windows in `recipe.json` **before** freezing the study. For example,
when the first outer validation origin is April 21:

```json
"tuning_windows": [{"origin": "2024-03-31", "horizon": 7}]
```

Every tuning target must precede the first outer validation target. The complete frozen cohort
must have usable history by the tuning origin. The tuning window uses the selection snapshot;
outer validation and test labels never select HPO parameters. Missingness and metric rules
are the same at every boundary.

Install the model's extra if needed; both HPO backends are included by default. Copy its profile from
`configs/models/` into your project's `models/` catalog. If the profile declares an
`implementation.config`, copy that configuration to the same relative path in your project.
Its SHA-256 must still match the profile. Choose a history length and training budget that fit
your data and hardware.

Set the candidate in `campaign.yaml` to:

```yaml
models:
  candidates:
    - model_id: sft
      parameters:
        mode: tune
        study_config: hpo.yaml
```

Create `hpo.yaml` with explicit data ranges and a trial budget. This template uses two series
whose selection snapshot covers January 1 through April 21; replace its IDs, hashes, and
ranges with those from your own study:

```yaml
schema_version: 1
study:
  id: sft-inner-search
  model_config: models/sft/model.yaml
  backend: optuna
  direction: minimize
  objective: validation/objective
  max_trials: 10
  top_k: 1
  seed: 7
  comparability_fingerprint: STUDY_ID
  evaluator: builtin
  data:
    id: selection-snapshot
    fingerprint: SELECTION_ARTIFACT_SHA256
    observed_date_range: ['2024-01-01', '2024-04-21']
    training_date_range: ['2024-01-01', '2024-03-31']
    validation_date_range: ['2024-01-01', '2024-03-31']
    validation_target_date_range: ['2024-04-01', '2024-04-07']
    test_date_range: ['2024-04-08', '2024-04-14']
    test_target_date_range: ['2024-04-08', '2024-04-14']
    training_series: 2
    validation_series: 2
    test: reserved_not_evaluated
```

`fingerprint` is the frozen manifest's `artifacts.selection`. Both series counts must equal
the frozen cohort. The training range ends at the inner origin; validation targets match the
declared tuning window. The reserved inner test range is not evaluated by HPO. The model
profile defines the search space. Run the same `autoforecast run --study-id ...` command after
configuring the campaign.

To reuse its retained `selection.json`, change the candidate mode to `reuse_tuned`, keep
`study_config`, and add `selection: /path/to/selection.json`. Reuse verifies the study boundary,
data, model, training protocol, and executable provenance. Changing those inputs requires a
new tuning run. Defaults and manual parameters do not require tuning windows.

Campaign requests use `schema_version: 6`; standalone run requests use `schema_version: 3`.
The checked-in examples declare `precision: 32-true` to preserve their FP32 settings. New
requests without an explicit precision use `16-mixed`; checkpoint continuation remains FP32.
Use a current example as a starting point. Older request and resolved-artifact formats are
rejected rather than translated.

### Tune a model supplied by Python

Keep your custom model in your application and pass its fitting function to `Study.tune`.
No built-in adapter registration or remote service is required. AutoForecast searches settings,
checks forecasts, scores the selected frozen windows, and saves the selection. Your
function fits the model, generates forecasts and saves its model files.

With one registered inner window it is selected automatically. With several, choose
`tuning_origin`, `selection.origins: [origin, ...]`, or `selection.origins: all`.
Alternatively, `selection.stage: validation` searches on the existing outer validation boundary
and does not require inner windows. Final testing never selects parameters.
The initial external-model route supports sequential Optuna; built-in Ray execution is unchanged.

An external tuning configuration reuses the existing parameter domains and constraints:

```yaml
schema_version: 2
model:
  id: custom-prophet
  adapter: external
hpo:
  backend: optuna
  max_trials: 20
  seed: 7
hyperparameters:
  model.changepoint_prior_scale:
    default: 0.05
    valid: {kind: float, low: 0.001, high: 0.5, log: true}
    search: {kind: float, low: 0.01, high: 0.1, log: true}
```

Pass an ordinary Python function; configuration files never load executable code:

```python
from autoforecast.study_tuning import FitPredictResult, load_tuning_result

def fit_predict(request):
    # Implement these model-specific operations in your application.
    model = fit_my_model(
        request.training_data,
        model_parameters=request.model_parameters,
        training_parameters=request.training_parameters,
        seed=request.seed,
    )
    # Return the existing Forecast type, with adapter=request.model_id and
    # origin=request.origin. Save the complete series/date grid under output_dir.
    forecasts = predict_and_save(
        model, request.series_mapping, request.forecast_dates,
        output_dir=request.output_dir / "forecasts",
        model_id=request.model_id, origin=request.origin,
        frequency=request.panel.frequency,
    )
    save_my_model(model, request.output_dir / "model")
    return FitPredictResult(forecasts=forecasts, model_format="my-model-format-v1")

selection = study.tune(
    fit_predict=fit_predict,
    config="prophet_tuning.yaml",
    implementation={
        "name": "my-project.custom-prophet",
        "version": "1.0.0",
        "source_sha256": source_sha256,  # SHA-256 of your retained implementation source
        "dependencies": dependency_versions,  # e.g. {"prophet": installed_version}
    },
)

print(selection.parameters)    # Full profile names, e.g. model.changepoint_prior_scale
print(selection.tuning_score)
selected_again = load_tuning_result(selection.path)
# Your application loads selected_again.model_artifacts using its own model loader.
```

The example's fitting, prediction and serialization functions belong to your application.
`request.panel` describes the physically sliced training Parquet and its column roles.
`request.series_mapping` contains the retained IDs and source keys. `forecast_dates` follows
the study frequency, including hourly and calendar frequencies. Each trial receives the same
seed; the callback must apply it. Model dependencies belong to the caller's environment.

The callback receives no target actuals or Study object. It is trusted Python running in your
process, not a sandbox: AutoForecast cannot attest what other data that code reads. Code identity
and dependency versions are caller declarations, recorded unchanged for traceability. Inputs
and returned artifacts are checked for changes. Run data identity uses the full frozen study
manifest; the sliced training input has a separate digest. Missing predictions fail; missing actuals use
the frozen study policy. No values are imputed by tuning.

Working training/mapping copies are removed when tuning finishes or fails; callbacks must not
retain their dataframes or input paths for later use. The original frozen study remains the
data source for another run.

Each completed trial retains forecasts, scores, model files and provenance under the existing
HPO attempt manifests. The result exposes `trial_id`, `model_id`, `parameters`, `tuning_score`,
`forecasts`, `model_artifacts`, `model_format`, `trial_history`, `provenance` and `path`.
Reload checks all ranked attempts and their recorded hashes; it never executes a model loader.
Failed trials abort the search without publishing a selection. Calls create new runs unless
`reuse_tuning=selection.path` explicitly reopens a completed search. Reuse checks the study,
profile and implementation identities and performs no new trials. Resuming an incomplete
search is not part of this interface. With multiple windows, `forecasts` and `model_artifacts`
are `None`; use `selection.windows` for each boundary’s separate artifacts.

### Several windows, configurable metrics and preflight

Add these settings to the external profile above:

```yaml
selection:
  stage: tuning             # or validation
  origins: all              # omit when stage is validation
  metric: mae               # mae, rmse, wape, or wrmsse; defaults to study metric
  aggregation: mean
reporting:
  metrics: [mae, rmse, wape, bias, wrmsse]
  forecast_accuracy_breakdown: both  # none, per_step, cumulative, both
series_policy: error        # default; also exclude_unavailable or use_available
minimum_history: 1          # usable observations required by your model
```

Every trial fits a fresh model at every chosen window using identical sampled settings and
seed. Its objective is the equal-weight mean of those window scores. A mean of three RMSEs
or WRMSSEs is **not** a score pooled over all forecast dates. Every required window must succeed.
Bias is a diagnostic, not a selection objective. Requested WRMSSE requires registered weights
and scales, even when MAE selects the winner. Optional undefined diagnostics retain a reason;
an undefined selection objective blocks the search.

When registering a **new** study, `tuning_schedule: {horizon: 14, stride: 14}` generates three
inner windows ending at the outer training cutoff. Add `count: 5` for five windows. These
numbers count frequency periods (hours, days, weeks or calendar periods). An explicit
`tuning_windows` list conflicts with the schedule; existing frozen studies never expand.
To register optional WRMSSE references for a study whose primary metric is MAE, supply
`scoring_references` at registration. Tuning does not invent or replace references.

`study.preflight(config=...)` checks all registered stages and saves a report.
This standalone entry point requires a profile with an `hpo` block and resolves its
selection windows. Fixed-only profiles without `hpo` use the workflow’s automatic preflight. Tuning and the
full workflow call it automatically before the first fit. The original registered IDs are
the population; every required prediction must cover its entire resolved date grid.

| Series policy | What the run does |
| --- | --- |
| `error` | Retains the original cohort; stops on insufficient history or actuals forbidden by the frozen policy. An explicitly registered actuals mask remains valid. |
| `exclude_unavailable` | Removes one union of unavailable series across all stages, freezes a derived study and normalizes its retained weights once. |
| `use_available` | Freezes trainable IDs and actual masks per window; reports original and available populations separately. |

Availability policies never excuse corrupt data, changed hashes, unexpected IDs or missing
predictions. Supply the same combined history requirement and effective study when comparing
models. Availability determined using later stages is labelled retrospective.
In `use_available`, WRMSSE uses only dates with actuals for **every** series in the fixed
weighted cohort. For A/B with B missing day 2, WRMSSE uses days 1 and 3 for both; MAE can use
all five observed cells. An unfittable weighted member or no common dates makes WRMSSE
unavailable. No API here fills missing values or fabricates pre-start history.

### Prepare additive data and scoring references

`autoforecast.preparation.prepare_time_series` accepts Spark data, original ID/date/value
column names, and explicit `fill_missing="na"` or `"zero"` and `duplicate_policy="error"`
or `"sum"`. It supports period-start H/D/W/MS/YS, additive aggregation and calendar completion.
An explicit null propagates to that value column in the aggregated period; absent source
rows allow a partial sum. Calendar completion runs from each supplied series’ first through
last observation; `series_end_date` can extend the end. It does not invent earlier periods
or wholly absent IDs. For example, observations on January 2 and 4 insert January 3 only.
`series_end_date` is also an inclusive source cutoff, aligned to the source frequency. Rows are checked for valid IDs, dates, finite values and source-period alignment before filtering, so the cutoff cannot hide invalid rows. Rows after it are then discarded; duplicate checks apply to the retained rows. A series with no rows on or before it is absent from the output; if no rows remain anywhere, preparation raises. Reconcile input and output ID populations when using it.
Timestamp input requires a UTC Spark session. Weeks start on Monday; weekly-to-monthly/yearly
aggregation is rejected. These are this helper’s restrictions, separate from study registration.
Set `observed_col="source_observed"` to distinguish inserted periods from periods with at least
one source row; true does not mean a complete feed. This does not decide business eligibility.

`autoforecast.study_references.prepare_wrmsse_references` builds daily historical weights and
scales with explicit source columns, forecast origins, missing-history policy, lookback,
scale lag and scale start. It returns the existing study reference format. It uses only history
through each origin and never excludes invalid series silently. The ID population comes from
the complete supplied frame: a future-only new ID has invalid earlier references, so validation
raises (or `validate=False` retains those rows for inspection). Callers must supply the intended
population. `validate=False` bypasses reference-value validation, not input/preparation validation. At each requested origin, at least one row must exist somewhere in the supplied population on or before that origin. An individual future-only ID needs no earlier row: with `validate=False`, it remains with invalid references. If the entire population has no historical rows at a boundary, the helper raises even with `validate=False`. See the synthetic examples in
`autoforecast/test_preparation.py`.

### Full workflow and reuse on future data

```python
run = study.run_forecast_workflow(
    fit_predict=fit_predict,
    config="prophet_tuning.yaml",
    implementation=implementation_identity,
    output_dir="/my/experiment-results",  # optional; defaults beside the study registry
)
print(run.parameters)
print(run.validation["score"])
print(run.testing["score"])
print(run.selected_config)                # winning settings: skip HPO
print(run.search_config)                  # search settings: run HPO again

# Later: new data and new dates, same winning model settings, no HPO.
next_run = next_study.run_forecast_workflow(
    fit_predict=fit_predict,
    selected_config=run.selected_config,
    implementation=implementation_identity,
    output_dir="/my/next-experiment",
)
```

For a first run with fixed settings, pass `fixed_parameters={}` to use profile defaults,
or explicit overrides such as `fixed_parameters={"model.level": 4.0}`. This requires `config`
and makes two fitting calls with **zero optimizer trials**. The profile's `hpo` block is optional
in this mode; its seed is used when supplied, otherwise the recorded seed is 42. It cannot be
combined with `selected_config`, `reuse_tuning` or `tuning_origin`. The result records
`mode="fixed_configuration"`; it does not claim the manual settings won a search.

The callback receives `request.purpose`: `"selection"`, `"validation"` or `"refit"`.
Outer-validation tuning still uses `"selection"`. The orchestration supplies this value;
it is not a tunable model parameter.

Use `validation_only=True` to retain a checked validation checkpoint before choosing among
models. Its `testing` field is `None`. After choosing, pass `validation_run=checkpoint.path`
to run on the **same study and implementation**: the validation evidence is reused without
rescoring, and only the final refit is performed. A checkpoint is not a completed test run
and cannot be exported as one. Reusing settings on new data still uses `selected_config`.

Inner selection with T trials and N windows performs N×T callback fits, one outer validation
fit and one final refit. Outer-validation selection performs T+1 fits: its selected validation
model/forecasts are reused, then the final model is fitted separately. The work estimate is
logged before tuning. Horizon reports use schema `autoforecast.forecast-accuracy-breakdown.v2`; each metric has its
own status and reason, while the row status describes the primary metric. Saved reports must satisfy the current supported contract; obsolete formats are rejected. The full workflow defaults to per-step reporting; tune-only defaults to
overall reporting. Returned models remain in the caller’s serialization format.

Saved-settings execution performs two fresh fits and no parameter search. It allows different
data and dates, checks parameter names/types/domains/constraints and the declared implementation,
and records the parameter digest and source identity. Search-derived settings retain search,
trial and attempt identity; fixed-derived settings retain their source workflow and study. Both fits keep the saved
seed, including when a current reporting configuration is supplied. It reuses settings, not
the earlier model’s learned state.
To retune later, pass `config=previous_run.search_config` and omit `selected_config` if that
profile contains a valid HPO configuration. A fixed-only profile may omit `hpo`; add a valid
search configuration before using it to tune.
`Study.tune` also exposes `selection.search_config`. Explicit old origin dates remain explicit;
use `selection.origins: all` for a search profile intended for different registered schedules.
You can run HPO again whenever you choose; no automatic retuning schedule is imposed. An optional
current `config` supplies reporting/availability settings and validates runtime parameter validity,
without forcing a saved value into the current HPO search range.

Keep `metadata/winner-config.json` inside its retained trial or workflow directory: the artifact
manifest verifies it. The workflow copies it into each new run, so later fixed-settings runs
need their immediate predecessor’s bundle, not the original study’s data. Use
`load_selected_configuration(path)` from `autoforecast.study_tuning` to inspect it without
loading any model code. Use `load_forecast_workflow(path)` from `autoforecast.study_workflow`
to verify and reload a workflow. `Study.tune` also accepts `output_dir` and exposes `winner_config`.
Output directories cannot sit inside frozen input data or an existing retained run.

Workflow `output_dir` controls retained evidence. Training snapshots and the callback’s
`request.output_dir` use separate temporary directories. All Spark readers/writers must be able
to access those temporary paths; a shared retained-output path does not configure this access.
For distributed Spark, configure an accessible temporary filesystem before the process starts
(for example, `TMPDIR` on each participating process) and verify access in that environment.
Campaign workers use local Spark; their local tests do not certify a distributed Spark notebook.

### Find and reuse saved outputs

Use `output_dir` as an experiment folder, for example `outputs/demand-baseline`.
Each normal workflow call creates a child such as
`workflow-20261028t081530z-<full-random-uuid>`. The timestamp is UTC. The full random
ID separates concurrent runs, and exclusive directory creation refuses overwrites.
Repeating the same settings creates a separate execution; parameter and input hashes
in the metadata explain which settings or data changed.

Without an explicit destination, workflow outputs go under
`.autoforecast/study-workflows/<study-id>/` beside the registry. Direct tuning uses
`study-tuning/<study-id>/studies/<tune-id>/`; standalone `evaluate_forecasts` uses
`study-runs/<study-id>/`. All three return their exact saved paths. `Study.preflight`
and `Study.run_campaign` also accept an optional destination. Their default parents are
`.autoforecast/study-preflight/<study-id>/` and
`.autoforecast/study-campaigns/<study-id>/`. Preflight creates `check-…/report.json`;
the campaign returns its exact `campaign_path`. Campaigns preserve their recovery contract: the same name, request and destination reopens verified
work. Use a different experiment folder or campaign name for an independent repeat.

Open `RUN.md` in a run or attempt folder for its status, actual files and their
purposes. It is sealed along with the artifacts. Keep JSON metadata, YAML configuration,
Parquet forecasts/scores and native model files in their original formats. The
outer search and campaign guides are snapshots; their verified child bundles and
selection/journal records remain authoritative. A completed execution may still have
unavailable metrics; inspect the recorded evaluation status and coverage.

A callback can save optional diagnostics under `request.output_dir / "artifacts"`.
For example, write `artifacts/feature-importance.parquet` or `artifacts/training-log.jsonl`.
Those files are retained for each window, including when a callback raises a normal
Python exception after run allocation. Invalid arguments can fail before a run exists.
Other callback scratch files are discarded. Abrupt process loss
can leave an incomplete attempt; it cannot promise a finalized report.

```python
from autoforecast.results import read_workflow_results, read_tuning_results, export_results

tables = read_workflow_results(run.path)  # Includes this run's search, if it performed/reused one.
export_results(tables, run.path.parent / (run.path.name + "-analysis"))
# For a successful standalone search: tables = read_tuning_results(selection.path)
# For a failed search: tables = read_tuning_results(search_path_from_failure_record)
```

`read_workflow_results` reads completed workflows or validation checkpoints.
`read_tuning_results` also reads a failed search when it contains a sealed failed attempt.
It returns failure metadata and any windows completed before the failure; it does not
invent a winning trial or forecast rows for unfinished windows. Unsealed attempts are
listed separately and contribute no rows. A search with no sealed evidence cannot be
exported this way. A failed workflow records its search path in `failure.json` and
`metadata/tuning-reference.json` when a search was allocated.

Campaigns keep the original supplied YAML text at
`<campaign_path>/profiles/candidate-<index>.yaml`. Worker `profile.json` files contain
normalized campaign profiles. Workflow `metadata/config.yaml` contains the supplied
profile or a reconstructed parameter profile; it is not a complete execution specification.
Read `metadata/winner-config.json` for effective parameter values and
`metadata/provenance.json` for resolved reporting, selection, availability and seed settings.
`metadata/search-config.yaml` retains the source profile associated with those parameters.

The result readers verify retained evidence before returning analysis tables, preserving
stage, origin and trial identities. `export_results` writes the supplied tables and metadata
as Parquet, JSON and a Markdown file guide, and rejects conflicting export retries. It does
not independently reverify the original run. Use the verified readers to obtain the tables.
Neither step loads or retrains models. Write exports outside completed bundles. A saved-settings or checkpoint
continuation records its earlier configuration source; it does not silently combine
an older experiment's search rows with the new experiment.

Keep the referenced frozen study and search directories: training slices are temporary,
and a workflow folder alone is not necessarily a portable archive.

### Several external models and optional parallel execution

```python
from autoforecast.study_campaign import StudyModel

result = study.run_campaign(
    models=[
        StudyModel("my_models.baselines:fit", baseline_identity, "baseline.yaml",
                   fixed_parameters={}, cpus=1, memory_gib=4),
        StudyModel("my_models.forecaster:fit", model_identity, "search.yaml",
                   cpus=2, memory_gib=6),
    ],
    output_dir="/shared/experiments/comparison",
    execution="local",                 # Opt into Ray processes; the API default is sequential.
    max_parallel_models=2,
    final_evaluation="validation_winner",  # Or all_models.
)
```

Model IDs in profiles must be distinct. Candidates share an objective, availability policy,
and compatible tuning windows. Before fitting, preflight resolves the strictest declared
history requirement once for the common study. Every candidate then uses that effective
study and its frozen eligibility/masks. Comparison rejects different evaluation contracts.
Unavailable objectives stop selection. Caught model-job errors appear in phase summaries;
phase-level rejection or timeout can occur before a summary is written and is recorded by
the existing supervisor as a failed attempt.

The existing campaign runner owns phase order, attempts and recovery. Its validation phase
runs candidate jobs; its final phase selects from validation and evaluates the requested
models. Test scores never choose the winner. Repeating an identical request verifies saved
outputs and resumes unfinished work instead of refitting completed models.

Local mode uses the existing Ray dependency for separate model processes. Cluster mode
requires `execution="cluster", ray_address="..."` for an existing cluster; the library does
not provision machines. All workers need the same installed model modules/dependencies and
shared study/output paths. Requests carry paths and hashes, not live Spark objects or notebook
closures. Workers check input/code identity, start their own local Spark session, and retain
separate model artifacts. A node must satisfy each job's CPU/GPU/memory declaration.
For cluster-only placement, a `StudyModel` can add `node_labels={"pool": "forecast"}`.
Labels are exact key/value matches. Admission requires one existing live node to satisfy every
label and CPU/GPU/memory request together; an impossible request fails before job submission.
The labels must already be configured on the cluster. AutoForecast delegates placement through
[Ray label selectors](https://docs.ray.io/en/latest/ray-core/scheduling/labels.html); it does not
provision a pool or expose Ray's selector-expression syntax through `node_labels`.
Shared paths must support the filesystem and locking operations used by retained bundles and
campaign recovery; this interface does not accept object-store URLs as output directories.

`max_parallel_models` bounds concurrent model jobs. Spark worker counts and standard numerical
thread settings are bounded within jobs; memory is monitored for each process tree. Custom
model code remains trusted and must respect its resource declaration. Trials and windows
remain sequential within a model: this avoids multiplying model, trial and Spark parallelism.
Scheduling separate model jobs does not automatically distribute one model's training algorithm.

Local parallel execution has synthetic acceptance tests. Multi-machine operation requires
an existing shared environment; the real two-machine acceptance test is explicitly deferred.

### Export checked external-model workflows

```python
from autoforecast.run_mlflow import export_forecast_workflow

links = export_forecast_workflow(
    run.path,
    tracking_uri="sqlite:////shared/tracking/mlflow.db",
    experiment_id="1",  # Must already exist and be active.
)
```

This uses the existing MLflow integration, creates separate validation/test run links and
uploads retained models, forecasts, exact scores and horizon reports without retraining or
rescoring. It verifies a private snapshot before remote writes and checks uploaded model bytes
and score JSON. Retrying the same workflow/experiment reuses remote run identities, including
a partial upload. Publication must have one shared export-lock location; this is not a
cross-storage distributed lock. Customer table publication remains caller-owned.

`study.run_campaign` also accepts `tracking_uri` and `experiment_id` together to export its
completed workflows and return `mlflow_links`. Omitting them performs no remote tracking writes.
Export happens after successful completion; unfinished validation evidence remains local.

Configuration/implementation errors can reject a request before a run opens. Failures after
a workflow opens seal a failed bundle; available report references depend on the failure point.
Failed bundles cannot reload as successful workflows. `status="partial"` means only some original cells were scored;
`status="pending_actuals"` means testing forecasts exist but no testing actuals were scored.
Neither status claims a complete evaluation. Final testing scores never feed back into HPO.

## Inspect results

`run` prints the campaign directory. It contains stage logs, run bundles, selection information,
forecasts, metrics, and source identities. Read and export verified results:

```python
from pathlib import Path
from autoforecast.results import read_results, export_results, comparison_plot

results = read_results('/path/printed/by/run')
print(results.rows.to_pandas())
print(results.metadata['frequency'])
print(results.metadata['final_score'])
export_results(results, Path('comparison-export'))
comparison_plot(results, series_id=results.rows['series_id'][0].as_py(),
                output=Path('comparison.png'))
```

All candidates at a scoring boundary must return the same forecast keys. The study freezes
actual coverage independently of model output. Exports include `series-scores.parquet` with
per-run scores and coverage, including zero-coverage series.
Result reading and resume reject changed retained artifacts. The synthetic example checks the
pipeline, not real-world predictive accuracy.

The exported `target_date` column retains timestamps for hourly forecasts and dates for the
other cadences. Plots use those same forecast periods, including calendar month and year boundaries.

To score existing forecasts against revised actuals, freeze a new study with matching forecast
boundaries and cohort, then write a separate post-hoc result:

```bash
autoforecast rescore --project-root my-experiment \
  --campaign /path/to/original/campaign --study-id NEW_STUDY_ID \
  --output rescored-results
```

The output retains original run/model/forecast identities and both study identities. It does
not refit models or replace the original selection and scores. Changed forecast grids are
refused. Use a fresh output directory for each rescoring operation.

## Models and M5

The maintained model catalog is in `configs/models/`. Additional model extras include
`shallow`, `custom-torch`, `tft`, `autogluon-zoo`, `timesfm`, and `timesfm3`. The two TimesFM
extras require separate environments. Install only the runtime for your selected model.

The custom model families include MFT, SFT, HMFNet, PACT, TFT-Q, TFT-HNB, and TimesFM-HNB.
Each adapter declares its input, horizon, device, and parameter requirements. A model being
listed does not mean it supports every dataset or device. The installed release check exercises
the two built-in baselines across all five frequency families. Optional models explicitly declare
their supported frequencies and reject incompatible inputs before execution; their existing
daily-model contracts do not become hourly or monthly support automatically.

M5 dataset and model requests remain in `configs/datasets/` and `configs/autoforecast/` for
source-checkout workflows. Acquire M5 separately under its dataset terms. Use the prepared
Parquet workflow for ordinary time-series comparisons. Competition WRMSSE needs the M5 hierarchy,
weights, and benchmark-specific evaluation contract; generic RMSE is not that competition score.

To add a maintained model, implement an adapter under `autoforecast/adapters/`, register its
`AdapterDescriptor` in `adapter_contracts.py`, declare its runtime and parameter resolution,
and build a new package. Configuration selects registered code; it cannot load arbitrary plugins.

## Development and verification

| Directory | Responsibility |
|---|---|
| `autoforecast/` | Input contracts, preparation, model adapters, orchestration, scoring, results. |
| `forecast/` | Forecasting models, feature preparation, and scenario calculations. |
| `configs/` | Dataset contracts, model catalog, and example requests. |
| `mlops/`, `provenance/`, `observability/` | Artifact integrity, run provenance, and diagnostics. |
| `spec/` | Data and model specifications; reference data rules live in `forecast/data/`. |
| `scripts/` | Reusable command-line operations and installed-package verification. |

Build a wheel with `python -m pip wheel --no-deps --wheel-dir dist .`, install it in a fresh
virtual environment, then run from outside the checkout:

```bash
python -m scripts.verify_timeseries_release /absolute/path/to/new-daily-check
python -m scripts.verify_timeseries_release /absolute/path/to/new-hourly-check --frequency H --csv
python -m scripts.verify_timeseries_release /absolute/path/to/new-monthly-check --frequency MS
python -m scripts.verify_timeseries_release /absolute/path/to/new-yearly-check --frequency YS
```

These checks cover preparation, isolated workers, validation selection, refit, final scoring,
export, plotting, resume, and tamper rejection. They assert exact forecast keys and reconcile
84 source observations, two retained series, no exclusions, and 14 cells per evaluation.
CI repeats CSV and Parquet workflows for every listed cadence on Python 3.12, with additional
daily checks on Python 3.13. Source tests live beside their modules; optional-model tests require their corresponding extras. Dataset-backed
tests report skips when the external dataset is absent.

Install the `ci` development dependency group and the model extras required by the chosen
tests, then run `python scripts/run_tests.py`. Each module runs
in a separate process because Spark sessions belong to individual test modules. The runner
also uses the same Python interpreter for Spark workers and the test driver.

No datasets, trained weights, experiment outputs, or credentials are distributed in this
repository. Third-party libraries and downloaded models retain their own licenses and terms.

Repository examples and tests must remain organisation-neutral. Use synthetic series and generic paths; keep customer names, source-table identifiers and deployment handoffs outside this repository.
