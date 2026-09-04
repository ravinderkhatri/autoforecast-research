# Learn AutoForecast

Start in the library checkout after following the main README installation steps.
In the same terminal, activate the environment and start Jupyter:

```bash
source .venv/bin/activate
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name autoforecast --display-name "AutoForecast"
python -m jupyterlab
```

In Jupyter's file browser, open `tutorials/01-first-forecast.ipynb` for the command-line
introduction, or `02-register-data.ipynb` for Python. Select the **AutoForecast** kernel.
Run cells from top to bottom. Use **Kernel > Restart Kernel** before the next Spark notebook.
Restarting the kernel clears Python memory; it does not delete saved workspace files.

Generated data and runs go outside this repository, in `~/autoforecast-tutorials` by default.
Set `AUTOFORECAST_TUTORIAL_DIR` in the terminal before starting Jupyter to choose another folder.
The published notebooks contain no saved outputs or credentials.

| Order | Notebook | What you learn | Requires |
|---|---|---|---|
| 1 | [First forecast](01-first-forecast.ipynb) | Run included models on two products | Package and Java 17 |
| 2 | [Register data](02-register-data.ipynb) | Save data and evaluation rules | Package and Java 17 |
| 3 | [Custom model](03-custom-model.ipynb) | Write forecasts and save model files | 2 |
| 4 | [Tune and backtest](04-tune-and-backtest.ipynb) | Compare settings on earlier dates | 2, 3 |
| 5 | [Read results](05-read-results.ipynb) | Understand predictions, errors and coverage | 4 |
| 6 | [Reload and reuse](06-reload-and-reuse.ipynb) | Open saved work and understand retries | 4 |
| 7 | [Delta](07-publish-delta.ipynb) | Publish through Spark | 4, Delta runtime, fresh kernel |
| 8 | [PostgreSQL](08-publish-postgresql.ipynb) | Publish with a database transaction | 4, postgres extra, database |
| 9 | [BigQuery](09-publish-bigquery.ipynb) | Publish with cloud jobs | 4, bigquery extra, cloud project |

Notebook 1 is a standalone command-line introduction. Notebooks 2 through 6 follow one Python
example. Choose any destination notebook after 4; there is no need to run all three.
Restart the kernel between notebooks that start Spark, especially before the Delta notebook.
All examples use synthetic data. Database examples create persistent destination tables.

## Before a database lesson

Results tables store predictions as 64-bit floating-point numbers. If a saved value would
need rounding in that format, publication stops before writing either table. This can happen
with exact decimal values or very large integers. The saved forecast files stay unchanged.

Use the same activated environment and library checkout:

```bash
# Install only the destination you want to use.
python -m pip install '.[postgres]'
# Or, for BigQuery:
python -m pip install '.[bigquery]'
```

PostgreSQL needs an existing database and write access to its `public` schema.
Set a connection string in the terminal that will launch Jupyter. Replace these placeholders:

```bash
export AUTOFORECAST_POSTGRES_DSN='postgresql://USERNAME:PASSWORD@HOST:5432/DATABASE'
```

BigQuery needs an existing dataset, its region, and an account allowed to create tables and run jobs.
Follow Google's [Application Default Credentials setup](https://cloud.google.com/docs/authentication/provide-credentials-adc).
Then set your own values before launching Jupyter:

```bash
export AUTOFORECAST_BIGQUERY_DATASET='your-project.your_dataset'
export AUTOFORECAST_BIGQUERY_LOCATION='EU'
export AUTOFORECAST_TABLE_PREFIX='tutorial'
```

Use the dataset's actual region; `EU` is only an example. Restart Jupyter from that terminal
after changing environment variables. Do not paste passwords or credentials into notebook cells.
The destination notebooks explain which tables remain after the lesson.

## How the pieces fit

Your application supplies the data and model. AutoForecast saves the study, runs the model
and checks its output.

![Application and library responsibilities](../assets/02-workflow-function.svg)

Earlier windows help choose settings. The final test remains separate.

![Tuning, validation and final testing](../assets/01-cross-validation.svg)

Start with one model at a time. Parallel workers need accessible code and shared paths.
Cluster mode needs an existing Ray cluster; it does not distribute arbitrary model internals.

![Sequential, local and cluster execution](../assets/03-parallel-execution.svg)

Saved outputs let you inspect and publish the same result without training again.

![Saved outputs](../assets/04-run-outputs.svg)

## Progress and logs

Call `configure_logging()` once before registering data or running a model. Repeating
it in another cell is safe. At the default INFO level, a notebook with display
support shows one panel per registration, model-run or publication attempt.
The panel updates in place without clearing other output. WARNING or higher hides
the panel. Stage outcomes remain as ordinary INFO log lines, so you can still
read them in an exported notebook. Opening saved evidence can also show ordinary
verification messages, without starting a numbered progress panel.

```python
from autoforecast import configure_logging, register_study, run_study

# Enable progress for this notebook.
configure_logging()

# Register data and show each step as it runs.
study_id = register_study(config_path, registry=registry)

# Hide INFO messages and the panel. Audit events are still saved.
result = run_study(study_id, "seasonal_naive", registry=registry, log_level="WARNING")
```

For example, `Stage 3/5` means stage three is running. It does not mean three
stages have finished. `2/5 stages completed` means two have passed. Downloading
all the bytes does not finish the integrity checks that follow. A skipped stage
is listed separately. Failed or cancelled work stays in the remaining list.

Progress uses counters the work already has. If a total is unknown, the display
says so. There is no estimated whole-run finish time. A panel that keeps showing
elapsed time means the reporting thread is alive; it does not prove the model
is making progress. Updates normally happen every ten seconds.

Both registration and model runs choose their logging level in this order:

1. The call's `log_level` argument.
2. The `AUTOFORECAST_LOG_LEVEL` environment variable.
3. The resolved configuration's `runtime.log_level`.
4. `INFO`.

Use `configure_logging(progress="plain")` for job logs, or `progress="off"` to
keep only permanent log messages. Applications that already configure Python
logging can use their own handlers instead. The helper does not change the root logger.

Registration saves its audit under the registry's `checks/<attempt>/audit.jsonl`.
A model run returns its audit path in `result.publication["audit_path"]`. These
files record lifecycle events and decisions. Display refreshes do not add audit
rows. Detailed SQL, exclusions, coverage and scores stay in their existing
evidence files; the log points to them. Issue reasons can overlap, so adding
their counts does not give the number of excluded series.

Registration can write frozen data and registry tables even when `runtime.publish`
is false. That setting controls model-result publishing. MLflow receives the
completed workflow afterward; it is not the source of live fit progress.

If a callback fails or a catchable cancellation occurs, the library records the
operation and re-raises the original exception. A hard process stop can leave an
incomplete attempt. Keep its evidence.

If fitting finished but result-table publication failed, the caught exception's
`workflow_path` gives the saved workflow location. Check that it is not `None`.
Load that workflow with `load_forecast_workflow(workflow_path)`, then retry
`registry.publish_results(saved.study_id, saved.path, destination=destination)`
using the loaded object's study ID and your intended destination. This retries
**table publication only**, without fitting again. It does not retry MLflow export.
Earlier failures may have no completed workflow to publish.

The INFO log shows parameter names. Their values stay in the saved configuration,
because custom model settings can contain private information, even as numbers.
Logs show the model ID. If several runs share that ID, give each a public name:

```python
from autoforecast import run_study

# This name helps readers recognise the run in the logs.
result = run_study(
    study_id, "seasonal_naive", registry=registry, model_label="Baseline check",
)
```

You will see `model_label=Baseline check` alongside the model ID in console and audit
logs. The label only names the run: it does not select a model, change its settings or
change the returned result type. Use a name you are happy to share, without private values.
The same optional `model_label` argument works with `run_forecast_workflow()`.
Leave it out to keep using the model ID. Within one comparison campaign, each candidate
still needs a distinct model ID.

## Keeping saved runs safe

Think of a saved run as a folder you can reopen later. It contains the forecasts,
scores and the information needed to check them.

For example, Monday's run saves its work in one folder. Tuesday's run gets a new
folder. Both may read the same saved study. They must not change that study while
either run is reading it. If you change the data, save a new version.

- **One writer per run folder.** Two notebooks can run at the same time when each
  writes to a separate folder. A new run or attempt needs a new ID. Reusing an
  existing run directory is refused.
- **Keep completed files unchanged.** The library creates files without replacing
  existing ones, then reads them back to check the bytes. A run bundle or attempt
  writes a manifest of filenames and SHA hashes before writing `completion.json`
  last. A hash is a fingerprint used to detect changed bytes.
- **Let the library check completion.** Readers check the completion record,
  manifest, file list and hashes. A stopped run without valid completion evidence
  is not a usable result. Keep it for troubleshooting and start a fresh attempt;
  do not create a completion file yourself.
- **Protect shared folders outside the library too.** On a remote volume, the
  library accesses files by path. It cannot lock a directory's identity or commit
  several files as one transaction. Its locks coordinate cooperating writers;
  they cannot stop a separate notebook or cleanup job from changing those paths.
  Use storage permissions and coordinate cleanup so nobody renames, replaces or
  deletes source snapshots or active output folders. Local filesystem protections
  do not provide this guarantee on a remote volume, even through a local mount.
- **Check against the saved fingerprint.** Copies and reloaded results are checked
  against their previously frozen hashes. Hashing a newly copied file alone does
  not prove the source stayed unchanged during the copy. Keep source snapshots
  unchanged throughout reads, including failed writes and retries.

For a cleanup example, do not delete Monday's study while Tuesday's run still
reads it. Wait until all readers and writers have stopped, then follow your
storage-retention rules. A stopped writer may also leave a lock: confirm it has
stopped before removing that lock. Never remove a lock belonging to an active run.
