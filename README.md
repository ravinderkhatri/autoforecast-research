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
It must apply `request.seed`, return a `FitPredictResult` containing a complete `Forecast`,
and save its model under `request.output_dir / "model"`. Optional diagnostics belong in
`request.output_dir / "artifacts"`. That workspace is temporary; `run.path` is the retained result.
All Spark workers must also be able to access temporary working paths.
Your application still owns data access, business rules, and publishing.

See the [small callback examples](autoforecast/test_study_campaign.py) and
[callback types](autoforecast/study_tuning.py). These are trusted Python functions:
AutoForecast supplies bounded history but cannot control other data your code reads.

![Arguments, callback inputs, model fitting, and retained results](assets/02-workflow-function.svg)

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

![Time windows and a worked example of selecting settings](assets/01-cross-validation.svg)

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

![Run folders, file formats, verified reading, and reuse](assets/04-run-outputs.svg)

## More examples and development

- [Study API](autoforecast/study.py): load studies, inspect data, and score forecasts.
- [Runnable Python example](scripts/study_example.py): `python -m scripts.study_example /absolute/path/to/new-example` demonstrates registration and scoring without training.
- [Preparation helpers](autoforecast/preparation.py): explicit aggregation, duplicate, and missing-value rules.
- [MLflow export](autoforecast/run_mlflow.py): optional tracking of saved workflow results.
- [Installed-package checks](scripts/verify_timeseries_release.py): exercise the workflow outside the source checkout.

Tests live beside their modules. Install the `ci` dependency group and any model extras required
by your tests, then run `python scripts/run_tests.py`. CI runs fast checks first.
Eight regression groups and native-model acceptance then run in parallel. Installed-wheel
and model-extra checks wait for that stage to pass. Every group must pass.
