# AutoForecast

Compare forecasting models, choose their settings, and save the results for later analysis.
AutoForecast handles data checks, time-based evaluation, refitting, and result files.
You can use the included models or bring your own Python model.

![From input data to checked forecasts and saved results](assets/autoforecast-overview.svg)

## Install

You need **Python 3.12 or 3.13** and **Java 17**. Supported platforms are Linux x86_64
and macOS Apple Silicon.

```bash
git clone https://github.com/ravinderkhatri/autoforecast-research.git
cd autoforecast-research
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .

# Small local Spark runtime. Set JAVA_HOME to Java 17 if needed.
export FORECAST_SPARK_CORES=2
export FORECAST_SPARK_MEMORY=1g
```

This includes basic and statistical models, parameter search, and result tools.
Add other models only when needed:

| Models | Install from this checkout |
|---|---|
| LightGBM | `python -m pip install '.[shallow]'` |
| MFT, SFT, HMFNet, PACT | `python -m pip install '.[custom-torch]'` |
| TFT models | `python -m pip install '.[tft]'` |
| AutoGluon | `python -m pip install '.[autogluon-zoo]'` |
| TimesFM 3 | `python -m pip install '.[timesfm3]'` |
| All of the above | `python -m pip install '.[all-models]'` |

LightGBM on macOS also needs `brew install libomp`. Standalone TimesFM 2.5 uses
`.[timesfm]` in a **separate environment**; it cannot share the TimesFM 3 runtime.
Installing a model does not download its weights or guarantee support for every frequency.
See the [model profiles](configs/models/) for model settings.

## Try a small example

Create the experiment outside the source checkout:

```bash
cd ..
autoforecast init-example my-experiment
autoforecast freeze --project-root my-experiment --source data
```

The example has two product series and 42 days: **84 observations, no excluded series**.
Each seven-day forecast contains 14 values. `freeze` checks and saves the data and evaluation
rules as a **study**. Copy the `benchmark.json` path it prints into this command:

```bash
autoforecast run --project-root my-experiment \
  --benchmark /absolute/path/printed/by/freeze/benchmark.json
```

The run compares two baseline models on earlier periods, selects one, and evaluates it on
later data. It prints the results folder. Repeating this command resumes the same campaign.
This small example checks the workflow; it does not measure real-world forecasting quality.

## Use your data

Start from the example project. CSV and Parquet are supported; M5 data is optional.
A CSV might contain `store,product,date,units`, where store and product together identify a series.

| Project file | What you set |
|---|---|
| `dataset.yaml` | `series_col: store`, `time_col: date`, `target_col: units`, frequency, and season length |
| `recipe.json` | `source_keys: ["store", "product"]`, matching frequency, dates, horizons, and scoring rules |
| `campaign.yaml` | Models, compute limits, and temporal settings matching the recipe |
| `models/*/model.yaml` | Each model's settings and search ranges |

Make these edits before freezing. CSV import preserves your column names; it does not
update the configuration files. Then convert your data:

```bash
autoforecast import-csv sales.csv my-experiment/custom-data/sales.parquet \
  --id-columns store product --date-column date --target-column units
autoforecast freeze --project-root my-experiment --source custom-data/sales.parquet
```

For non-daily CSV data, also pass `--frequency` to `import-csv`.
Set both `dataset.panel.frequency` and `recipe.frequency` to `H` (hourly), `D` (daily), `W` (weekly),
`M`/`ME` (month end), `MS` (month start), `Y`/`YE` (year end), or `YS` (year start).
Horizons count periods: a horizon of 7 means seven hours for hourly data and seven months
for monthly data. Dates must match the chosen frequency; hourly timestamps must have no timezone.

Observed targets must be finite and non-negative. Missing history is not silently filled with
zero. Missing evaluation values are errors by default; explicit masking records the coverage.
Choose the intended series population and date rules before freezing. A new dataset or changed
rules require a new study identity.

## Register once and run by study ID

Use the package API when another application owns the data query and model code:

```python
import logging
from autoforecast import register_study, run_study
from autoforecast.study_registration import StudyRegistry

logging.basicConfig(level=logging.INFO)
registry = StudyRegistry.from_config("study.yaml", spark=spark)
study_id = register_study("study.yaml", registry=registry)
result = run_study(study_id, "seasonal_naive", registry=registry)
print(result.summary())
```

Registration checks and freezes data; it does not fit a model. Runs load the frozen ID and
delegate to the existing preflight, tuning, fitting and scoring workflow. The advanced
`load_study()` and `Study.run_forecast_workflow()` interfaces remain available.
The built-in convenience route currently supports `seasonal_naive` and retains its history
and panel settings as the fitted model. Other models supply the existing `FitPredictRequest`
to `FitPredictResult` callback or a `StudyModel` declaration:

```python
result = run_study(
    study_id, my_fit_predict,
    registry=registry,
    model_config="model.yaml",
    implementation={
        "name": "my-model", "version": "1",
        "source_sha256": code_sha256,
        "dependencies": {"my-model-package": "1.0.0"},
    },
    tracking="none", publish=False,
)
```

The callback receives its fit history, panel, source-key mapping, origin, forecast dates,
parameters, seed and output directory. It needs no global Study object. It saves predictions
and model files using the existing callback contract. An explicit `hpo` block in the model
profile enables tuning; without one, profile defaults run without HPO. `fixed_parameters={}`
also explicitly skips tuning. Model settings cannot override registered dates or scoring rules.

`study.yaml` combines the existing dataset and recipe formats with source/storage settings.
All dates below must be quoted YAML strings. The selected source must contain the declared
population, one row per key/date, and only the declared input date range.

```yaml
schema_version: 1
dataset:
  schema_version: 1
  dataset:
    id: example
    panel:
      series_col: item
      time_col: day
      target_col: quantity
      frequency: D
      season_length: 7
      target_estimand: observed_nonnegative
recipe:
  schema_version: 2
  source_keys: [item]
  frequency: D
  training_start: "2024-01-01"
  validation_origins: ["2024-01-28"]
  validation_horizon: 7
  final_train_cutoff: "2024-02-04"
  final_test_horizon: 7
  history_policy: fixed
  new_series_policy: error
  target_unit: units
  feature_availability: {}
  metric: mae
  history_missing: preserve
  actual_missing: mask
source:
  kind: parquet
  path: input.parquet
storage:
  root: saved-studies
runtime:
  tracking: none
  publish: false
  log_level: INFO
```

Paths resolve relative to the config file. Duplicate YAML keys, aliases and unknown settings
are rejected. Registration infers each series' available start from its first observation
through the validation origin, or uses explicit `recipe.series_starts` declarations. It never
invents history before that start. `history_missing: zero` fills only each model's fit slice;
raw actuals remain unchanged. `actual_missing: zero` explicitly scores missing actuals as zero
and records the original missing and filled counts. `mask` excludes them from the score;
`error` refuses missing actuals. Static features fill only from evidence available by the fit
cutoff; past-only features are not filled from future observations.

For WRMSSE, add `references: {path: references.parquet}` or supply `weight_col`,
`missing_history`, `duplicate_policy` and optional `weight_window_days`, `scale_lag`,
`scale_start` for the existing daily reference calculator. `invalid_references` defaults to
`error`; explicit `exclude` removes affected keys across every window and rechecks the retained
weight total. It records the keys, reasons and excluded reference weights/scales. This is
separate from model-profile availability exclusion. Generic source findings remain under
`checks/`, including failed registrations; they do not impose application business rules.
Registration reference reporting currently requires daily studies.
With Delta storage, successful source and reference reports are linked to the study before its
registration is published. Failed attempts keep their findings without a successful study link.

For a fresh process, configure `AUTOFORECAST_REGISTRY_ROOT` and optional
`AUTOFORECAST_SCRATCH_ROOT` / `AUTOFORECAST_DELTA_NAMESPACE`, or set `AUTOFORECAST_CONFIG`.
Environment storage settings take precedence over YAML in both registration and loading;
an explicit `registry=` overrides both. Config-based loading reads storage/runtime settings
only, never the original source SQL.
There is no process-global active registry. Separate explicit registries can run concurrently.

### Application preparation and admission evidence

Pass an explicit preparation object to the same registration call when your application
owns eligibility or transformation decisions. Its `validate` method checks domain settings
before managed source access. Its `prepare` method receives captured settings, resolved
windows, materialized observations, named audit inputs, exact population keys, generic
findings and a temporary workspace. Configuration never names Python code to import.

For example, a daily sensor application can explicitly exclude sensors with too few
observed values before the first validation origin. This is an application policy, not a
library default. Add these settings to the study configuration:

```yaml
preparation:
  minimum_observations: 12
records:
  format: json
```

```python
from pyspark.sql import functions as F
from autoforecast import register_study
from autoforecast.study_registration import EXCLUSION_REASON_COLUMN, PreparationResult

class SensorPreparation:
    def validate(self, settings):
        minimum = settings.parameters.get("minimum_observations")
        if type(minimum) is not int or minimum < 1:
            raise ValueError("minimum_observations must be a positive integer")
        if settings.recipe["frequency"] != "D":
            raise ValueError("This sensor preparation policy uses daily observations")

    def prepare(self, context):
        recipe = context.settings.recipe
        panel = context.settings.dataset["dataset"]["panel"]
        keys = recipe["source_keys"]
        minimum = context.settings.parameters["minimum_observations"]
        history = context.observations.filter(
            F.col(panel["time_col"]).between(
                F.lit(recipe["training_start"]).cast("date"),
                F.lit(recipe["validation_origins"][0]).cast("date"),
            )
        )
        counts = history.groupBy(*keys).agg(
            F.count(panel["target_col"]).alias("__af_observed_count")
        )
        excluded = context.starting_keys.join(counts, keys, "left").filter(
            F.coalesce(F.col("__af_observed_count"), F.lit(0)) < minimum
        ).select(*keys)
        retained = context.starting_keys.join(excluded, keys, "left_anti")
        return PreparationResult(
            observations=context.observations.join(retained, keys, "inner"),
            starting_keys=context.starting_keys,
            retained_keys=retained,
            exclusions=excluded.withColumn(EXCLUSION_REASON_COLUMN, F.lit("SHORT_SENSOR_HISTORY")),
            identifier="sensor-history", version="1",
            policy={"minimum_observations": minimum},
            actions=("exclude sensors below the declared history requirement",),
        )

study_id = register_study("sensors.yaml", registry=registry, prepare=SensorPreparation())
```

Run models by this ID using the existing `run_study` API. The application does not hash
data, write issue tables, link findings or author generated admission records.
`PreparationSettings`, `PreparationContext`, `PreparationResult` and `StudyPreparation`
are public contracts in `autoforecast.study_registration`.

Starting keys must equal the disjoint union of retained and excluded keys, with the same
types and a nonblank `EXCLUSION_REASON_COLUMN` for every exclusion. Equal counts with different IDs fail.
Prepared observations must contain exactly the declared retained keys. Optional `starts`
contains those keys plus a canonical string `START_COLUMN`; optional `findings` uses the supplied
findings schema. `actions` records actions taken; `planned_actions` records declared intent.
Neither can replace native structural failures or managed source facts.
Preparation parameters and policies are mappings of named JSON scalars, not nested objects.
The metadata columns are `__af_exclusion_reason` and `__af_history_start`, using the existing
reserved prefix so ordinary keys such as `reason` or `start` remain valid. Import both column
constants from `autoforecast.study_registration`. Existing configured starts are retained for
surviving keys when the hook excludes series without supplying replacement starts.

To calculate references from a separate prepared frame, return `reference_input` with the
existing reference calculation settings. To supply already computed references, configure
`references: {prepared: true}` and return `scoring_references`. These routes are mutually
exclusive and use the same reference schema, coverage and numeric checks as native registration.
Key, stage and origin types remain strict; reference origins are canonical strings.

For managed Delta audit inputs, extend the source section with named SQL files:

```yaml
source:
  kind: delta
  sql: observations.sql
  tables:
    measurements: catalog.schema.sensor_readings
  audit_queries:
    original: population.sql
  population: original
  require_matching_population: true
```

For this example, both queries read the declared `measurements` alias. The primary query
selects the configured observation columns; the population query selects distinct source
keys. Every query is checked before any source-version lookup. Each physical table is pinned
once across aliases and queries. The existing read-only SQL operator restrictions apply;
CTEs and undeclared relations are not enabled by this feature. Audit queries require an
explicit preparation hook.

`population` names the query supplying the starting IDs, defaulting to the primary query.
SQL date conditions and pinned Delta versions are recorded separately. Primary and population
membership must match unless `require_matching_population: false` is explicit. Differences
then remain separate population findings and key artifacts; the hook must still account for
every retained or excluded population key. Choosing the correct business population remains
the application's responsibility. The primary input must satisfy the native observation
contract before preparation; transaction-level or attribute-only data belongs in audit inputs.

Registration retains exact membership, exclusions, findings, policies and final cohort evidence.
Their logical identity participates in the existing study identity. Changed material decisions
produce a different ID even when prepared observations are identical. Returned lazy frames
are materialized before checks, and captured settings/source baselines cannot be replaced by
hook output. Native exclusions are reconciled again, for example 100 starting sensors,
95 after application eligibility, and 93 after two native reference exclusions.

Configuration accepts `.yaml`, `.yml` and `.json` through one validation path. Nested duplicate
keys, unsafe YAML tags, anchors/aliases, non-string mapping keys and non-finite numbers fail.
Quote ISO dates. Equivalent validated configuration and resolved paths have the same logical
identity across input encodings. Generated registration and admission metadata default to JSON;
`records: {format: yaml}` selects YAML. Readers authenticate either format, reject conflicting
copies, and keep the first published bytes and location on identical retries. Benchmark and
tracking manifests, Parquet artifacts and JSON table columns retain their existing formats.
Current no-hook registrations and current-format study IDs remain supported unchanged.

Load verifies admission records and retained artifacts before exposing model inputs. It needs
neither the original source/configuration nor the preparation object. Source SQL, versions,
schemas and input hashes attest to the inputs read during registration; discarded raw/audit
rows are not replayable from those hashes. Full raw/audit-row archival is not enabled by default,
and Delta versions alone do not guarantee replay after vacuum. Frozen model inputs and durable
admission evidence are retained through the existing stores.

Preparation is trusted Python, not a sandbox. Only managed reads receive library-verified
provenance. Storage baselines require the same protected shared POSIX filesystem as existing
registrations. Configuration, source, admission, freeze and publication stages log START/PASS/FAIL
without routine row contents. Failed attempts retain available findings and failure records.
Evidence and issue linking must succeed before completed registration publication; the sequence
of file and table writes is not one transaction.

### Optional Delta storage and publication

Supply a Delta-enabled Spark session and credentials through your runtime. Core file workflows
do not require Delta. The integration tests use Spark 4.2 with Delta 4.4.1. Add
`storage.namespace: catalog.schema` and optionally a `storage.tables` mapping for the five roles:
`studies`, `frozen_observations`, `study_data_issues`, `study_results`, `forecasts`.
Only explicit registration provisions these tables; reads and runs never do.
Initial table creation is serialized using the existing POSIX file-lock primitive under the
registry root. All processes provisioning the same tables must share that root and working
file-lock semantics. Row publication then uses native Delta transactions concurrently.

For Delta source queries, replace `source` with:

```yaml
source:
  kind: delta
  sql: source.sql
  tables:
    observations: catalog.schema.observations
```

`source.sql` contains a read-only query over the declared aliases, for example
`SELECT item, day, quantity FROM observations`. The integration pins each table version once,
runs the query in an isolated Spark session and materializes its result before checking it.
Undeclared tables, commands and unsupported query operators are rejected. Source SQL, versions,
typed schema, input counts and admission decisions are retained with the registration.

The five tables are shared across studies. Frozen observations use a `snapshot_id, row_json`
envelope plus an authenticated typed schema, so independent studies can use different column
names and types. Readback checks both the transport and the decoded typed content using the
existing bounded Parquet row hash. Arrays retain order; maps are canonicalized. Duplicates
affect the hash. Logical row hashes and physical artifact hashes remain distinct identities.
String-key maps and ordinary Spark scalar/array/struct types are supported; unsupported types
fail registration. Delta storage requires a UTC Spark session.

This is a new registration format and frozen-benchmark v4; historical IDs/formats are not
adapted or rewritten. The table snapshot **and** authenticated Parquet artifacts are retained.
Execution still reads Parquet; this is not zero-copy or table-native execution. Protect the
local registration baseline and artifact root with storage permissions. Hashes cannot protect
against an actor who can rewrite every trusted baseline. File roots must support POSIX atomic
publication, and scratch paths must be visible to Spark workers; object-store URLs are not
supported filesystem roots. No process-global temporary directory is changed.

Native Delta transaction IDs make identical writes retryable; different content for the same
identity is rejected. Definitions and both result-stage rows are published last, so partial
snapshot/forecast writes are not discoverable as completed studies/results. Multiple tables
and files are not one transaction. Do not delete/recreate destination tables or expire native
transaction records while writers are in flight. Sequential retries also compare existing rows.

### Tracking, publication and returned evidence

| Tracking | Publish | Effect |
|---|---|---|
| `none` | `False` | Retain local checked workflow evidence only; no MLflow API calls |
| `mlflow` | `False` | Also export checked evidence to an existing active experiment |
| `none` | `True` | Also publish tables with durable model paths and null MLflow links |
| `mlflow` | `True` | Export tracking evidence and publish tables |

Explicit arguments override environment controls, which override registry runtime settings.
The environment controls are `AUTOFORECAST_TRACKING`, `AUTOFORECAST_PUBLISH` (`true`/`false`),
`AUTOFORECAST_LOG_LEVEL`, `AUTOFORECAST_TRACKING_URI` and `AUTOFORECAST_MLFLOW_EXPERIMENT`.
Defaults are `tracking="none"`,
`publish=False`, `log_level="INFO"`. MLflow requires `tracking_uri=` and
`mlflow_experiment=` (an existing name or ID), or the corresponding runtime settings.
It validates the experiment before fitting and refuses an unrelated active run. It never
creates an experiment. Tracking exports completed evidence through the existing exporter.

`result.study_id`, `parent_study_id`, `run_id`, `validation_metrics`, `testing_metrics`,
`coverage`, `model_links`, `publication` and `summary()` expose the saved workflow. Full scores,
metric availability/reasons, per-series contributions and configured horizon results remain
in `result.validation` / `result.testing` and the sealed directory at `result.path`.
Unavailable values remain `None`. All returned paths refer to durable evidence.
The summary distinguishes the retained/excluded cohort from each stage's forecasted series count.

Stage events use standard Python logging and retained JSONL evidence. They include cohort counts,
exclusion and fill warnings, and the actual copy/scoring outcomes when reusing saved models.
The sealed completion record establishes successful workflow completion. Reduced console verbosity
does not remove audit records. Failed publication leaves the completed model workflow intact;
the exception's `workflow_path` and `audit_path` identify recovery evidence. Load it with
`load_forecast_workflow(workflow_path)` from `autoforecast.study_workflow`, then retry
`registry.publish_results(result.study_id, result.path)` using that loaded result. This uses
the effective ID even when preflight derived a cohort, without fitting or scoring again. When
tracking was used, recover verified links with `export_forecast_workflow()` and pass them as
`mlflow_links=`. Dashboard SQL is available from `registry.store.dashboard_sql()`; it joins
study ID **and** registered SHA, leaving run/stage identity on each result row. Creating a
dashboard view remains an explicit caller action.

## Inspect daily WRMSSE reference issues

Use the strict numeric report before registering caller-supplied or calculated references:

```python
from datetime import date
from autoforecast.study_references import check_wrmsse_references

issues = check_wrmsse_references(
    references,
    origins=[("tuning", date(2024, 1, 3)), ("tuning", date(2024, 1, 5))],
)
issues.show(truncate=False)
```

`references` is a Spark DataFrame with `keys` (the original typed key struct), `stage`,
`origin` (ISO date string), `weight`, and `scale`. Declare every expected boundary, even
when exclusions leave it empty. The result always has `keys`, `stage`, `origin`, `field`,
`issue_code`, `observed_value`, and `message`; valid input produces an empty report.

The codes are `INVALID_WEIGHT`, `INVALID_SCALE`, and `INVALID_TOTAL_WEIGHT`. Weights must
be finite and nonnegative; scales must be finite and positive. The total includes only
rows where both values are usable. Total issues have null keys; an empty usable total
has a null observed value. Null and non-finite offending values remain visible.

For calculated references, call `prepare_wrmsse_references(..., validate=False)` once,
then pass the result to this checker. Its default `validate=True` still raises `ValueError`
for invalid references. Use Spark `persist()` and materialize the references if repeated
actions should reuse a calculation; release them with `unpersist()` when finished.
The checker never changes values or removes series. Your application saves issues and
applies its exclusion policy across the shared cohort, then checks the retained references
again. Keep study registration and integrity checks: a clean numeric report does not
establish identity, complete cohort coverage, or artifact integrity. Existing frozen-study
handling of undefined scales is unchanged.

## Use your own Python model

Keep model training in your application. AutoForecast calls your function with the right
history and forecast dates, checks its predictions, calculates scores, and saves the results.

The Python Study API needs **one validation origin**: the last training date before
validation. For the daily quickstart data, change these fields in `recipe.json`:

```json
"validation_origins": ["2024-01-21"],
"tuning_windows": [
  {"origin": "2024-01-06", "horizon": 2},
  {"origin": "2024-01-08", "horizon": 2},
  {"origin": "2024-01-10", "horizon": 2}
]
```

These earlier windows support the tuning example below. Freeze again with the same command
and use the `study_id` from the new `benchmark.json`. If you also run the CLI campaign,
update its temporal settings to match the revised recipe.

The following is an integration outline. Supply your own `fit_model(request)` function,
`model.yaml`, that study ID, and a Spark session:

```python
import hashlib
from pathlib import Path
from autoforecast.study import load_study

model_implementation = {
    "name": "my-model",
    "version": "1.0.0",
    "source_sha256": hashlib.sha256(Path("my_model.py").read_bytes()).hexdigest(),
    "dependencies": {},  # Add the model libraries and their installed versions.
}

with load_study(study_id, registry_root="my-experiment/.autoforecast/benchmarks",
                spark=spark) as study:
    run = study.run_forecast_workflow(
        fit_predict=fit_model,
        implementation=model_implementation,
        config="model.yaml",
        fixed_parameters={},
        output_dir="outputs/demand-baseline",
    )
    print(run.path)
```

| Argument | Meaning in this example |
|---|---|
| `fit_predict=fit_model` | Your function that fits once and returns predictions and saved model files |
| `implementation=model_implementation` | A dictionary recording your code version, source hash, and dependencies |
| `config="model.yaml"` | A file defining the model ID, parameter names, defaults, and allowed values |
| `fixed_parameters={}` | Use the profile defaults without searching; an override could be `{"model.level": 4.0}` |
| `output_dir="outputs/demand-baseline"` | Parent folder for separate saved runs |

For example, a profile for a constant-value model can contain:

```yaml
schema_version: 2
model: {id: constant-example, adapter: external}
hyperparameters:
  model.level:
    default: 4.0
    valid: {kind: float, low: 0.0, high: 100.0, log: false}
    search: {kind: float, low: 1.0, high: 10.0, log: false}
```

`valid` sets the allowed range; `search` sets the tuning range. `log: false` uses a regular
numeric scale. Fixed mode ignores the search range.

The function receives `request.training_data`, `request.forecast_dates`, and parameter
dictionaries. Here, `request.model_parameters["level"]` is `4.0`.
Predict every supplied series/date pair and apply `request.seed`. Save forecast Parquet
(columns `series_id`, `date`, `yhat`) beneath `request.output_dir`, for example in its
`forecasts` subfolder. Return a `FitPredictResult` containing a `Forecast` that points to it.
Save model files under `request.output_dir / "model"` and optional diagnostics under
`request.output_dir / "artifacts"`. This workspace is temporary; `run.path` is the retained result.
All Spark workers must also be able to access temporary working paths.
Your application still owns data access, business rules, and publishing.

See the [small callback examples](autoforecast/test_study_campaign.py) and
[callback types](autoforecast/study_tuning.py). These are trusted Python functions:
AutoForecast supplies bounded history but cannot control other data your code reads.

![Application responsibilities and the checked registration and model lifecycle](assets/02-workflow-function.svg)

## Tune and evaluate over time

**Tuning** tries different model settings. **Cross-validation** repeats that comparison
on several earlier time windows. Each fit sees only history available before its forecast.
**Refitting** trains again on more recent history using the chosen settings.

Declare tuning windows in the study recipe before freezing. To search them, omit
`fixed_parameters` and add these sections to the model profile:

```yaml
hpo: {backend: optuna, max_trials: 5, seed: 42}
selection: {stage: tuning, origins: all, metric: mae}
```

The profile's `search` range is used only when tuning. In the example, it tries
`model.level` values between 1.0 and 10.0; fixed mode uses the default of 4.0.
With five trials and three tuning windows, a successful fresh workflow makes
**15 tuning calls + 1 validation call + 1 refit call = 17 calls**.
Fixed settings make two calls. Final test scores never choose the settings.
MAE means mean absolute error: the average size of the forecast errors.

![Earlier tuning windows, validation, and the separate final test](assets/01-cross-validation.svg)

To use saved settings on new data, pass `selected_config=run.selected_config` to a new
study's workflow, keep the same implementation identity, and omit `fixed_parameters`. It performs two fresh fits without tuning. To tune again, use
`config=run.search_config` from a profile containing `hpo` and the intended `selection` settings;
omit `selected_config` and `fixed_parameters`. For a fixed-only profile, add the tuning
sections to a separate editable copy first.
A study ID identifies frozen data and rules; a run ID identifies one execution.

## Run several models

Use `study.run_campaign(models=[...])` with one
[`StudyModel`](autoforecast/study_campaign.py) declaration per model.
For example, `StudyModel("my_models.baseline:fit", model_implementation, "model.yaml",
fixed_parameters={}, cpus=1, memory_gib=4)` names an importable function and declares
one CPU and a 4 GiB memory budget for its complete job.

| Execution setting | What happens |
|---|---|
| `execution="sequential"` | Default: models run one after another |
| `execution="local", max_parallel_models=2` | Up to two model jobs run in separate processes on one machine |
| `execution="cluster", ray_address="...", max_parallel_models=4` | Up to four model jobs run on an existing Ray cluster |

Available CPU, memory, and GPUs can reduce concurrency. Trials and windows within each
model job stay sequential. A model can manage its own parallel training, but AutoForecast
does not split an arbitrary training algorithm across processes or machines.

All cluster workers need the same code and accessible study/output paths on a shared filesystem.
Use local or mounted shared filesystem paths. Object-store URLs such as `s3://...` are
unsupported and are interpreted as local paths; do not pass them. Real two-machine testing remains pending.

Use `final_evaluation="validation_winner"` to refit only the validation winner, or
`"all_models"` to refit every validated candidate. Matching campaign requests resume saved work;
choose a new campaign name or destination for an independent repeat. The returned dictionary
contains `campaign_path`, the folder holding the campaign results.

![One machine, several processes, and several machines](assets/03-parallel-execution.svg)

## Find and analyse results

Each ordinary workflow call creates a new folder under `output_dir`, for example:

```text
outputs/demand-baseline/
  workflow-20261028t081530z-<full-random-uuid>/
    RUN.md
    metadata/
    validation/
    test/
```

The UTC timestamp makes runs easy to find; the full random ID prevents repeated or concurrent
runs from sharing a folder. Without `output_dir`, the default parent is
`.autoforecast/study-workflows/<study-id>/` beside the registry. Use `run.path` for the exact path.

Open the generated **`RUN.md`** first: it lists the available files and their purpose.

| Saved format | Contents |
|---|---|
| Parquet | Forecasts, scores, and analysis tables |
| JSON | Run status, effective parameters, data/code identities, and file checksums |
| YAML | Supplied or reconstructed model configuration |
| Native model files | Models saved by your fitting function |
| Markdown | The generated `RUN.md` file guide |

Tuning retains trial/window results, forecasts, and model files. Validation and refit retain
separate outputs. Optional training logs and feature importance are saved when your function
produces them. Failed runs retain available evidence; abrupt process loss may leave incomplete files.
Check evaluation status and coverage before reporting results.

```python
from autoforecast.results import read_workflow_results, export_results

tables = read_workflow_results(run.path, include_tuning=True)
export_results(tables, run.path.parent / (run.path.name + "-analysis"))
```

The reader verifies saved files. The export creates Parquet tables, JSON metadata, and a file
guide for analysis or dashboards, without retraining. For CLI campaigns use `read_results(path)`;
for separate searches use `read_tuning_results(path)` from the same module.

A Python `study.run_campaign` saves a workflow per model and phase. Tuning belongs to
its validation phase; the final workflow does not automatically include that earlier search.
For a completed campaign, export both phases:

```python
from pathlib import Path
from autoforecast.results import read_workflow_results, export_results

# campaign is the dictionary returned by study.run_campaign(...).
assert campaign["state"]["status"] == "completed"
analysis_root = Path(campaign["campaign_path"] + "-analysis")
for phase in ("validation", "final"):
    for candidate, saved in campaign["phases"][phase]["results"].items():
        tables = read_workflow_results(saved["workflow_path"], include_tuning=(phase == "validation"))
        export_results(tables, analysis_root / phase / candidate)
```

Final exports repeat the retained validation forecasts. Use one phase's validation rows
when combining dashboard data. This campaign folder is not an input to `read_results`.
The default campaign parent is `.autoforecast/study-campaigns/<study-id>/`; `output_dir` changes it.

Keep referenced study and search folders: the workflow folder alone may not contain every input.
Saved records must use the current format; older formats are rejected.

![Saved forecasts, fitted models, execution evidence, and verified reuse](assets/04-run-outputs.svg)

## More examples and development

- [Study API](autoforecast/study.py): load studies, inspect data, and score forecasts.
- [Runnable Python example](scripts/study_example.py): `python -m scripts.study_example /absolute/path/to/new-example` demonstrates registration and scoring without training.
- [Preparation helpers](autoforecast/preparation.py): explicit aggregation, duplicate, and missing-value rules.
- [MLflow export](autoforecast/run_mlflow.py): optional tracking of saved workflow results.
- [Installed-package checks](scripts/verify_timeseries_release.py): exercise the workflow outside the source checkout.

Tests live beside their modules. Install the `ci` dependency group and any model extras required
by your tests, then run `python scripts/run_tests.py`. Each module runs in a separate process
so Spark sessions and runtimes cannot leak between modules.

CI runs preflight checks first, then starts ten regression shards, native-model acceptance,
installed-wheel checks, and model-extra checks in parallel. Every group must pass. Superseded
pull-request runs are cancelled automatically; a failing matrix member does not cancel its
siblings, so one run can report failures across all supported variants. Installed-wheel jobs
group short frequency checks in the same environment, using a fresh project directory for each
case. This reduces runner queues while preserving every input, Python and frequency combination.

The regression runner balances modules using recent timings and estimates for new or split
modules in `configs/ci/test_durations.json`. Durations affect scheduling only: every discovered
module still runs exactly once, modules without an entry receive a 60-second estimate, and
stale entries do not add tests. Campaign recovery and availability checks run in a separate
module so one long module does not hold up the entire matrix. The real Ray placement check
also has its own process, without a live Spark session from earlier tests. Refresh estimates from completed
CI module timings when workloads change.
Run one of the same shards locally with:

```sh
python scripts/run_tests.py --shard-index 0 --shard-count 10 --durations configs/ci/test_durations.json
```

Local Linux containers can run these same entry points with the locked dependencies, Python
and Java versions used by CI. Match concurrency to the container's memory budget. Native ARM
containers on Apple Silicon provide Linux checks but do not replace the x86 Linux CI matrix.
