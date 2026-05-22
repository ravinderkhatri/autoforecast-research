# AutoForecast

Run reproducible forecasting experiments on your own daily time series. AutoForecast validates
input data, compares models on validation periods, refits the selected model, and scores a
separate test period. Each run retains its data identities, configuration, forecasts, and metrics.

Start with CSV or Parquet. Column names and series keys are configurable; M5 is an optional
benchmark, not a required data format. Data and outputs stay local unless you explicitly configure
an external tracking service.

## Requirements and installation

- Python 3.12 or 3.13 on Linux or macOS.
- Java 17 for the local Spark preparation and evaluation runtime.
- Enough local disk and memory for your dataset. The quickstart uses two small series.

```bash
git clone https://github.com/ravinderkhatri/autoforecast-research.git
cd autoforecast-research
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[daily,plots]'
```

Set `JAVA_HOME` to your Java 17 installation if it is not already configured. For a small local
run, limit Spark to two cores and a 1 GB JVM heap:

```bash
export FORECAST_SPARK_CORES=2
export FORECAST_SPARK_MEMORY=1g
```

The `daily` extra includes the local data/runtime dependencies. The two built-in baselines need
neither a GPU nor a model download. Additional model libraries are optional.

## Run the example

Create an experiment outside this checkout:

```bash
cd ..
autoforecast init-example my-experiment
autoforecast freeze --project-root my-experiment --source data
```

The example creates 84 observations: two `(shop, sku)` series across 42 days. Preparation retains
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

## Use your own data

Create the example project as a template, then replace its data and edit these files:

| File | What to change |
|---|---|
| `dataset.yaml` | Dataset ID; series, date, and target columns; season length; optional feature roles. |
| `recipe.json` | Source key columns, training start, validation origins, horizons, final cutoff, target unit. |
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

CSV keys remain strings, so `001` stays distinct from `1`. Dates must be `YYYY-MM-DD`.
The importer selects the mapped columns and records excluded columns, source checksum,
date range, row count, and series count in the Parquet file's metadata. Preparation records the
output checksum as part of its source identity. The importer refuses
conflicting output files. Use typed Parquet directly when you need additional features or
integral keys. The CSV convenience importer reads the selected observations into memory.

Map this example in `dataset.yaml`:

```yaml
schema_version: 1
dataset:
  id: my-daily-data
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
`final_test_horizon` to dates and horizons supported by your history. Copy those temporal values
into `campaign.yaml`. Validation windows must end by the final training cutoff. The test window
starts the following day. Update the dataset source labels in the campaign to describe your data.

```bash
autoforecast freeze --project-root my-experiment --source custom-data
```

Use the newly printed benchmark path with `autoforecast run`. The two-row-per-series excerpt
above only illustrates the format; training requires a complete daily history over your chosen
training, validation, and test windows.

### Data rules

- The release workflow supports finite, non-negative, observed daily values. Fractional values
  work with the baselines; count models can impose integer-target requirements.
- Parquet dates must use the date type. Each series/date must appear once.
- The cohort is fixed from series present by the first validation origin in the selection
  source. The manifest reconciles all source series, exclusions, and the retained cohort.
- Every retained series needs every required daily observation. Gaps and null targets are
  errors. A reported zero is an observation; an absent row is never filled with zero.
- New series default to an error. The explicit `exclude_after_first_origin` recipe policy
  excludes series absent from the initial cohort and records the count.
- `history_policy: fixed` requires overlapping history to agree across snapshots.
  `revised_snapshot` permits declared corrections and creates a new benchmark identity.
- Historical features use `past_only`. Prepared future-known features require declared
  availability evidence and an observation-state column. Unknown future availability is refused.

Separate snapshots can be supplied explicitly:

```bash
autoforecast freeze --project-root my-experiment \
  --selection data/selection --final-training data/training --final-score data/labels
```

Parquet directories must contain `.parquet` parts, optionally under Hive partition directories.
Hidden completion markers are ignored. Visible sidecars, summary metadata, streaming logs,
links, and special files are refused so every reader-visible observation is authenticated.

Hourly/monthly data, signed targets, missing-value imputation, raw warehouse feeds, and cloud
execution are outside this release workflow. Use `observed_nonnegative` for measured quantities,
including fractional values such as energy use. The two built-in baselines support this target.
Use `sales_given_availability` for observed retail sales. Optional retail models declare their
supported targets and reject incompatible inputs; none of these observed-target contracts
infers unobserved demand.

## Inspect results

`run` prints the campaign directory. It contains stage logs, run bundles, selection information,
forecasts, metrics, and source identities. Read and export verified results:

```python
from pathlib import Path
from autoforecast.results import read_results, export_results, comparison_plot

results = read_results('/path/printed/by/run')
print(results.rows.to_pandas())
print(results.metadata['final_score'])
export_results(results, Path('comparison-export'))
comparison_plot(results, series_id=results.rows['series_id'][0].as_py(),
                output=Path('comparison.png'))
```

Generic fixed-population scoring supports RMSE, MAE, and WAPE. All candidates at a scoring
boundary must return the same keys; missing predictions or actuals cannot reduce the denominator.
Result reading and resume reject changed retained artifacts. The synthetic example checks the
pipeline, not real-world predictive accuracy.

## Models and M5

The maintained model catalog is in `configs/models/`. Optional extras include `classical`,
`shallow`, `custom-torch`, `tft`, `autogluon-zoo`, and `timesfm`. Install only the runtime for your selected model.

The custom model families include MFT, SFT, HMFNet, PACT, TFT-Q, and TFT-HNB.
Each adapter declares its input, horizon, device, and parameter requirements. A model being
listed does not mean it supports every dataset or device. The installed release check exercises
the two built-in baselines; optional models retain their own tests and runtime contracts.

M5 dataset and model requests remain in `configs/datasets/` and `configs/autoforecast/` for
source-checkout workflows. Acquire M5 separately under its dataset terms. Use the prepared
Parquet workflow for ordinary daily comparisons. Competition WRMSSE needs the M5 hierarchy,
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
python -m scripts.verify_daily_release /absolute/path/to/new-parquet-check
python -m scripts.verify_daily_release /absolute/path/to/new-csv-check --csv
```

These checks cover preparation, isolated workers, validation selection, refit, final scoring,
export, plotting, resume, and tamper rejection. CI repeats the installed command-line workflows on Linux with Python 3.12 and 3.13. Source tests live
beside their modules; optional-model tests require their corresponding extras. Dataset-backed
tests report skips when the external dataset is absent.

With the test extras and `pytest` installed, run `python scripts/run_tests.py`. Each module runs
in a separate process because Spark sessions belong to individual test modules. The runner
also uses the same Python interpreter for Spark workers and the test driver.

No datasets, trained weights, experiment outputs, or credentials are distributed in this
repository. Third-party libraries and downloaded models retain their own licenses and terms.
