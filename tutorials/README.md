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
| 10 | [Supply your own stages](10-supply-your-own-stages.ipynb) | Register DataFrames with optional validation | Package and Java 17; standalone |
| 11 | [Prepare and check data](11-prepare-and-check-data.ipynb) | Validate settings and explain exclusions | Package and Java 17; standalone |
| 12 | [Keep experiment inputs](12-retained-inputs.ipynb) | Choose files or protected table captures and reuse the original data | Package and Java 17; 10 recommended; postgres extra and protected archive for table settings |
| 13 | [Reuse raw source data](13-reuse-source-captures.ipynb) | Share one raw capture while each study makes its own decisions | Package and Java 17; standalone |

Notebook 1 is a standalone command-line introduction. If you already have split DataFrames,
start with notebook 10. Use notebook 11 when your application needs a preparation rule. Notebooks 2 through 6 follow one Python
example. Choose any destination notebook after 4; there is no need to run all three.
Restart the kernel between notebooks that start Spark, especially before the Delta notebook.
All examples use synthetic data. Database examples create persistent destination tables.

## Before a database lesson

### Saved inputs are separate from the live source

An experiment needs the data it originally used, even after the source changes.
Suppose you register a study on Monday when product A has a value of 100. On Tuesday,
someone corrects the source to 120. Reopening Monday's study must still use 100.
AutoForecast does this by saving the study's inputs when you register it.

The `snapshot_mode` setting chooses how those inputs are kept:

| Setting | Saved inputs | Check when opening them |
|---|---|---|
| `hash` (default) | Files | Compare their content with the recorded hashes |
| `table` | Separate database captures | Check the capture identity and the database's protection rules |
| `table+hash` | Separate database captures | Also compare the row content with its recorded hash |

With `hash`, registration also records an identity for each source read. By default
(`storage.input_identity: rows`) it takes one fingerprint of the rows each read
returns, after the query's own filters. The same rows give the same study ID however
they were read: from a newer Delta version, from rewritten files, by another query or
from another table. Supplied stages work the same way, whether each stage is a
DataFrame or a Delta table. Registering the same rows again returns the saved study.
How, where and when the rows were read (query text, tables, Delta versions, table IDs,
storage locations, capture times and database snapshots) stays with the study as the
record of its first registration, but does not change its ID. Other rows, or other
study settings, give another study.

For Delta sources you can choose `storage.input_identity: source_version` instead.
Each read is then named by its table IDs, its pinned Delta versions and its query
text. This skips row hashing for the raw source captures; the frozen model-stage data
is still row-hashed. A committed Delta version never changes, so repeating the same
read keeps its source identity, provided the query is deterministic. A new table
version gives a new study ID, even when its rows are unchanged. The saved input files
are still checked against their file hashes during registration.

```yaml
storage:
  root: /Volumes/catalog/schema/studies
  input_identity: source_version   # Delta sources only; the default is rows
```

The source table and the saved capture are different objects. Table mode does not
mean saving only a source table name and an upload timestamp. The saved capture
must remain available until the study is released. A timestamp cannot recover an
older upload that was never captured and whose database history no longer exists.

Protected table inputs use PostgreSQL, MySQL or SQL Server and an archive set up once
by a database administrator. Registration copies the
inputs there, then the database removes its write access. Training can read those
inputs but cannot change them. When fitting needs a smaller training window, the library
seals a temporary capture using the restricted registration connection. It cannot
change the original study. Deleting a retained capture needs a separate account.
Use an ordinary account for PostgreSQL source reads too, with only the access
needed to read the source. Administrator, database-owner and archive-cleanup
accounts are refused. This applies even when the source uses a different server
address: an address alone does not prove that it reaches a separate database cluster.

For example, correcting A from 100 to 120 in the source does not change Monday's
capture. Trying the same update on the capture with the registration or training
account is refused by the database. If the capture is missing or its protection has
changed, opening the study fails. It never substitutes today's source data.

Database administrators and storage operators remain trusted. They can bypass
normal database protections. Table mode does not claim to detect every such
administrator change; `table+hash` adds a comparison with the recorded row content.
Neither mode can repair data that an administrator has deleted.

BigQuery source reads and result publishing are separate from retained inputs.
The BigQuery snapshot transport has service tests, but ordinary-account deletion,
expiry and permission controls still need their own live gate. Registration and
replay refuse BigQuery table inputs until that gate passes. Delta retained
destinations are also unfinished; a connection alone does not provide the
required protection.

These settings control experiment inputs. The destination lessons below publish
results, which is a separate choice.

Use `run_study`, `Study.tune` and `Study.run_campaign` with retained table inputs.
The older command-line runner, HPO bridge and campaign compiler require file
inputs. They refuse table inputs before reading or hashing their rows.

Registry methods use the connection settings given to that registry. Standalone
readers such as `load_forecast_workflow(path)`, and separate workers, need the same
connection aliases in a configuration file named by `AUTOFORECAST_CONFIG`.
Keep passwords out of that file: its `connections` section names environment
variables, as in the archive examples below. Reloading needs reader credentials.
Fitting can also need registration credentials to create temporary training captures.

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

## Set up a protected PostgreSQL archive

This setup is for **saved experiment inputs**, as used in notebook 12. It is separate
from notebook 8, which writes result tables.

From the checkout, install the PostgreSQL driver in the activated notebook environment:

```bash
# Install the driver before using either table setting.
python -m pip install '.[postgres]'
```

The administrator's setup environment needs this driver too. The default file lesson
does not need it.

A database administrator does this once, in a separate setup session:

1. Create an isolated `NOLOGIN` owner role with no members or administrative rights.
2. Create three distinct, ordinary login roles for registration, reading and release.
   None may inherit the owner or an administrative role. The reader cannot inherit
   registration, and normal application roles cannot inherit release.
3. Choose a new archive schema and provision it with the helper below.

Use your own role names. The helper refuses an existing schema so it cannot replace
another archive's settings. Configure authentication through your normal database
administration process. Do not put passwords in source files or notebook cells.

```python
import os
import psycopg
from autoforecast.integrations.postgres import provision_snapshot_schema

# Run only in the administrator's setup session, using an existing credential.
with psycopg.connect(os.environ["AF_ADMIN_DSN"], autocommit=True) as admin:
    provision_snapshot_schema(
        admin,
        schema="forecast_archive",
        owner="forecast_archive_owner",
        registration_role="forecast_register",
        reader_role="forecast_reader",
        release_role="forecast_release",
    )
```

The owner keeps control of the archive. Registration gets permission to insert into a
new capture only while creating it. Sealing removes that permission before commit.
Once the capture is retained, registration cannot delete it either. The reader only reads.
Fitting also needs the restricted registration connection to seal temporary training
windows. This does not give it permission to change existing retained inputs.
A separate release account handles deliberate deletion after readers and dependent studies
have finished. Keep administrator and release credentials out of ordinary training jobs.

For notebook 12, configure `AF_TUTORIAL_WRITER_DSN` and `AF_TUTORIAL_READER_DSN` for
the registration and reader roles. Set `AF_TUTORIAL_SNAPSHOT_SCHEMA=forecast_archive`.
Set `AF_TUTORIAL_RELEASE_DSN` only in a separate cleanup session, for deliberate
`registry.release` or unpublished-capture recovery with `registry.recover_capture`.
Leave it unset in ordinary training sessions.
An ordinary source account must not have archive ownership or administrative rights.
This includes rights inherited through other roles. For example, giving the
registration account membership in an administrator role makes it unsuitable,
even when its own account settings look restricted.
Source connections must use `standard_conforming_strings=on`. AutoForecast refuses
a connection or query that changes how PostgreSQL interprets quoted SQL text.

The library checks these protections when capturing and reopening data. It stops if
ownership, permissions or its installed sealing routine have changed. Administrators
remain trusted; these checks do not prove an administrator never changed a row.

## Set up a protected MySQL archive

MySQL 8.4 uses the same study APIs and input modes. Install the optional driver
with `pip install '.[mysql]'`. An administrator first creates four distinct
`user@%` accounts: a locked owner, registration, reader and release. The owner
must have no role members. Ordinary application accounts must not inherit the
owner or release account. Use an archive server whose ordinary accounts have
no global mutation or role-administration privileges.

Run this once in the administrator's setup session:

```python
import os
from autoforecast.integrations.mysql import connect, provision_snapshot_database

# Create a new archive with the accounts prepared by your database administrator.
with connect(os.environ["AF_ADMIN_DSN"]) as admin:
    provision_snapshot_database(
        admin,
        database="forecast_archive",
        owner="forecast_archive_owner",
        registration_role="forecast_register",
        reader_role="forecast_reader",
        release_role="forecast_release",
    )
```

The helper refuses an existing database. Registration can insert into a new
capture while building it. Sealing removes that permission and moves the table
to its final name. The rename waits for open transactions using the table to end,
under [MySQL's metadata-locking rules](https://dev.mysql.com/doc/refman/8.4/en/metadata-locking.html).
After sealing, none of the ordinary application accounts can change its
rows. A separate release account can delete a capture after it is no longer used.

Name the reader and registration credentials in the study configuration:

```yaml
connections:
  archive:
    kind: mysql
    dsn_env: AF_READER_DSN
    writer_dsn_env: AF_WRITER_DSN
    release_dsn_env: AF_RELEASE_DSN
storage:
  root: /your/study-registry
  snapshots:
    connection: archive
    namespace: forecast_archive
snapshot_mode: table
```

Each variable contains a `mysql://USERNAME:PASSWORD@HOST:3306/DATABASE`
connection string. Release credentials are opened only for an explicit cleanup
operation. Keep them out of training jobs. A source account on this server must
not also be an administrator or archive-release account.
Read-only source roles are allowed, but their entire chain of inherited permissions
must remain read-only for the archive. Server-wide mandatory roles are unsupported.
The administrator must leave `mandatory_roles` empty for this archive setup.

Source connections must leave `ANSI_QUOTES` and `NO_BACKSLASH_ESCAPES` off.
These settings change how MySQL reads quoted SQL. Executable comments such as
`/*! ... */` are also refused, because MySQL can run their contents as SQL.
Ordinary comments and quoted text remain usable.

MySQL connections must use `utf8mb4` in both the driver and the server's client,
connection and result character sets. The library's `connect()` helper sets this
up. The adapter refuses incompatible settings before transferring data, so a
changed session cannot silently damage text.

The adapter requires strict SQL mode, checks the installed sealing routine and
actual permissions, and stops if those protections change. It preserves typed
numbers, text, binary values, timestamps and nested structs. Float32 values use
DOUBLE storage to avoid MySQL text-protocol rounding, while their original
float32 schema is kept for replay. Unsupported values fail instead of being
silently rounded. Transfers use bounded batches through the Python driver.

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

## Create or reuse a study in one call

Use `run_study()` when you want to save new inputs and run a model together.
Inside notebook 2, you can use this after creating the registry and before stopping Spark.

```python
from autoforecast import run_study

# Save the configured data, then run a model. Keep the results as files.
config_path = work / "study.yaml"
study_info = {"study_id": None, "create_study": True, "publish_results": False}
result = run_study(study_info=study_info, config=config_path,
                   registry=registry, model="seasonal_naive")
print(result.study_id)
```

Keep the printed ID. To run on the same saved data, pass that ID and set
`create_study=False`. Leave out `config` and `prepare`: reuse reads the frozen
study, not the original source. The dictionary must contain all three keys, and it
may also set `file_checks` (see "Keeping saved runs safe").

```python
# Use the saved study without reading its source again.
study_info = {"study_id": result.study_id, "create_study": False, "publish_results": False}
result = run_study(study_info=study_info, registry=registry, model="seasonal_naive")
print(result.path)
```

The path contains the saved forecasts and scores. Set `publish_results=True`
to also use the registry's configured result tables. This does not control study
storage or MLflow tracking. A new registration always saves its checked data.
For direct DataFrame inputs, see notebook 10. An optional `prepare` hook works
with configured sources, as shown in notebook 11.

If training fails after registration, the exception's `registered_study_id` identifies the
saved study. Retry with that ID and `create_study=False`. If result-table publication
fails and the exception's `workflow_path` is not `None`, use its `study_id` and
`workflow_path` with `registry.publish_results(study_id, workflow_path)` to retry
the table write without training again. This does not retry MLflow; use a separate
`export_forecast_workflow` call with the saved workflow for that export.
Calling creation again can reread and check the source; it is not a shortcut for
reusing a saved study.

The existing `run_study(study_id, model, ...)` call still works. Choose either that
form or `study_info`; do not supply both. Use `register_study()` alone when you
want to save data without training.

## Progress and logs

Registration and model runs show progress automatically. At the default INFO level,
a notebook with display
support shows one panel per registration, model-run or publication attempt.
The panel updates in place without clearing other output. WARNING or higher hides
the panel. Stage outcomes also use Python logging. An application's logging
settings decide which of those messages appear. Opening saved evidence can show ordinary
verification messages, without starting a numbered progress panel.

```python
from autoforecast import register_study, run_study

# Register data and show each step as it runs.
study_id = register_study(config_path, registry=registry)

# Hide INFO messages and the panel. Audit events are still saved.
result = run_study(study_id, "seasonal_naive", registry=registry, log_level="WARNING")
```

Choose what the console shows with `display`. Saved audit events are the same in
every choice.

```python
from autoforecast import configure_logging

configure_logging(display="progress")  # The panel, plus warnings and errors.
configure_logging(display="logs")      # Event lines, without a panel.
configure_logging(display="both")      # The panel and event lines (the default).
```

A study configuration can set the same choice as `runtime.display`; an explicit
`configure_logging(display=...)` takes precedence. A run keeps the choice it started
with, and a registration started inside `run_study` uses the run's choice.
Without `configure_logging`, Python logging still decides which event lines appear
beside the panel. `display="logs"` has no panel, so it shows its event lines itself.
`display="progress"` with `log_level="WARNING"` shows only warnings and errors.

For example, `Stage 3/5` means stage three is running. It does not mean three
stages have finished. `2/5 stages completed` means two have passed. Downloading
all the bytes does not finish the integrity checks that follow. A skipped stage
is listed separately. Failed or cancelled work stays in the remaining list.

Progress uses counters the work already has. If a total is unknown, the display
says so.

During training the panel counts model fits. One fit trains the model for every
series on one window, then forecasts and scores that window. A search of 5 trials
over 3 windows, plus a validation fit and a test fit, makes 17 fits. Each fit names
its trial, its window and its periods in the study's own frequency:

```text
Model fits: 7/17 [========>...........] 41% | about 1 h left at 6 min per fit, measured
Now fitting: Parameter search · trial 3 of 5 · window 2 of 3 · Training period 2023-01-02 to 2026-06-28 · Validation period 2026-06-29 to 2026-07-12 (14 days)
```

The parameter search uses only validation periods. The test period appears once,
on the last fit, with a note when its actual values are missing or incomplete. The
dates are period labels from the data's time column; with weekly data the last date
labels the last week.

The time left appears after the first fit and is updated after each one: the fits
left, multiplied by the measured time per fit so far. It is approximate, because
settings, the larger final fit and the cluster's speed all change the time per fit,
and it leaves out work after the last fit, such as the MLflow export. Counters for
rows and bytes show no estimate. A panel that keeps showing elapsed time means the
reporting thread is alive; it does not prove the model is making progress. Updates
normally happen every ten seconds.

Both registration and model runs choose their logging level in this order:

1. The call's `log_level` argument.
2. The `AUTOFORECAST_LOG_LEVEL` environment variable.
3. The resolved configuration's `runtime.log_level`.
4. `INFO`.

For optional customization, import `configure_logging` from `autoforecast`.
Use `configure_logging(progress="plain")` for job logs, or `progress="off"` to
keep only permanent log messages. Set `AUTOFORECAST_LOG_LEVEL=WARNING` to hide normal
progress throughout a session.

By default, progress adds no logging handlers and leaves the root logger unchanged.
Application handlers or filters on the library logger take control of this default
display. A root level of ERROR or CRITICAL also hides it; the usual WARNING level
allows it.

Calling `configure_logging()` installs AutoForecast's console handler. Its progress
panels do not follow the root logger's level or the library logger's filters.
Use `log_level="WARNING"` or `AUTOFORECAST_LOG_LEVEL=WARNING` to hide normal progress.
Use `configure_logging(progress="off")` to hide panels while keeping log messages.
Unless you choose an explicit stream, text output follows the current notebook
cell's standard output. Creation uses runtime
settings in its configuration over registry defaults; explicit call options and
environment settings keep the precedence listed above.

Registration saves its audit under the registry's `checks/<attempt>/audit.jsonl`.
`run_study()` returns its audit path in `result.publication["audit_path"]`. These
audits record lifecycle events and decisions. On volumes, the path points to a small
record describing an adjacent `.events` folder. Events are saved once; a closed audit
also records its final event count and identity. Use the same reader for both formats:

```python
from autoforecast import read_execution_events

# Read the audit saved by the run_study call above.
audit_path = result.publication["audit_path"]
events = read_execution_events(audit_path)
# Show each saved step and its outcome.
for event in events:
    print(event["stage"], event["status"])
```

Each line shows a step and its recorded state, such as START or PASS. The reader checks
remote event order and contents. Reading events alone does not prove a run completed;
load the saved run through the normal API to check its completion record too.
Display refreshes do not add audit rows. Detailed SQL, exclusions, coverage and scores stay in their existing
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
Use the same configured registry to retry
`registry.publish_results(error.study_id, error.workflow_path, destination=destination)`,
where `error` is the caught exception and `destination` is your chosen result destination.
The registry supplies the connections needed to read table inputs. This retries
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

## After a notebook stops unexpectedly

First check that its process has stopped. Do not delete its tables or edit a
completion file by hand.

In a fresh session, reopen the **existing** registry folder. Do not rerun the
lesson cell that creates a new UUID-based folder. For notebook 12's PostgreSQL
example, restore the reader, registration and schema environment variables and
set `AF_TUTORIAL_RELEASE_DSN` in this cleanup session:

```python
import os
from pathlib import Path
from autoforecast import StudyRegistry
from pyspark.sql import SparkSession

# Replace this with the lesson folder printed by the interrupted notebook.
lesson = Path("/path/to/existing/saved-inputs-LESSON_ID")
# A fresh session needs Spark before opening the registry.
spark = SparkSession.builder.master("local[2]").getOrCreate()
registry = StudyRegistry(
    lesson / "registry", spark=spark, snapshot_mode="table",
    connection_settings={"archive": {
        "kind": "postgresql", "dsn_env": "AF_TUTORIAL_READER_DSN",
        "writer_dsn_env": "AF_TUTORIAL_WRITER_DSN",
        "release_dsn_env": "AF_TUTORIAL_RELEASE_DSN",
    }},
    snapshots={"connection": "archive", "namespace": os.environ["AF_TUTORIAL_SNAPSHOT_SCHEMA"]},
)
```

Use the mode and connections from the original registration. For a file-only
registry, `StudyRegistry(lesson / "registry", spark=spark)` is enough. These IDs have different uses:

| ID | Where to find it | Cleanup operation |
|---|---|---|
| Study ID, 64 hex characters | Returned by `register_study`, or saved under `benchmarks/STUDY_ID/` | `release(study_id)` for table studies |
| Source-capture ID, 64 hex characters | Returned by `capture_sources` | `release(source_capture_id)` |
| Failed table-capture ID, 32 hex characters | `capture-journal/` directory name | `recover_capture(failed_table_capture_id)` |
| Reader lease ID, 32 hex characters | File name under `readers/STUDY_OR_CAPTURE_ID/` | `recover_reader(study_or_capture_id, lease_id)` |

The paths in this table are relative to the existing registry folder. Copy the
exact saved IDs into the calls below; a source-capture ID is not a failed table-capture ID.

A stopped reader may leave an entry at `readers/STUDY_ID/LEASE_ID.json`.
That entry blocks cleanup while the reader might still need the data.
Recover that exact reader with:

```python
# Close the saved reader entry after its original process has stopped.
registry.recover_reader(study_id, lease_id)
```

A failed registration may leave a protected table before it saves a benchmark.
Its capture ID and original table identity are saved under
`capture-journal/CAPTURE_ID/`. Use a registry configured with separate release
credentials to remove that unpublished capture:

```python
# Remove only this failed capture, using its saved identity.
registry.recover_capture(failed_table_capture_id)
```

Registration can also stop after saving `benchmarks/STUDY_ID/benchmark.json`
but before returning the study ID. For example, the data may be saved successfully
just before the notebook loses its connection.

For `table` and `table+hash` studies, release the whole saved study as shown below.
Replace `STUDY_ID` with its existing folder name. File-only studies do not support
this study-release call. Keep an incomplete file attempt for investigation and
start a fresh lesson folder. Remove its files only after all readers and dependent
runs have finished, following your retention rules. Releasing a reusable file
source capture is a separate, supported operation.

```python
from autoforecast.benchmark import read_benchmark_manifest

# Choose the study folder left by this failed registration.
failed_study_id = "STUDY_ID"
saved_manifest = registry.root / "benchmarks" / failed_study_id / "benchmark.json"
_, saved = read_benchmark_manifest(saved_manifest)
# Check that the saved record belongs to the study we intend to remove.
if saved["study_id"] != failed_study_id:
    raise ValueError("The saved record belongs to a different study.")
# Remove the tables owned by this study, including its saved checking reports.
registry.release(failed_study_id)
```

This also works when the registration record was never written. It keeps the
saved records and marks the study as released. If cleanup itself stops, call
`release` again with the same ID to finish it.

Recovery refuses active owners and captures used by published studies. Release
also refuses studies that still have readers or dependent studies.
If the sealed identity is missing or changed, recovery stops for investigation.
It never deletes a table just because its name looks familiar. On remote volumes,
a stopped process can leave a lock that an operator must inspect before recovery.

After finishing cleanup, close the registry and Spark session:

```python
# Finish this cleanup session without deleting any additional inputs.
registry.close()
spark.stop()
```

## Databricks and shared storage

Managed Delta source queries use Spark's public DataFrame and SQL APIs.
Live validation on Spark Connect and Databricks Serverless is still pending. Queries read
only their declared aliases, with one pinned Delta version per physical source table.
Script transforms and Spark `${...}` substitution are not accepted in managed source SQL.

A storage root under `/Volumes/catalog/schema/volume/...` uses the Databricks Files API.
The adapter creates files without replacing existing paths and uses lock files to coordinate
AutoForecast writers. Storage permissions must keep other writers and cleanup jobs from
changing active paths. Live volume validation is still pending. The SDK must authenticate
to the same workspace as Spark and have read/write access to the volume.
Other paths keep the local POSIX behavior. Hash checks download fresh remote bytes, so
allow local disk space for the checked artifact. When saving a file through this adapter, uploads larger than 5 GiB are refused.

Execution logs on volumes save each event separately, so progress does not rewrite the growing log.
Use `read_execution_events(path)` to read these logs or ordinary local logs.

A stopped process can leave a volume lock behind. A later run stops instead of stealing it.
Confirm that the owning run has stopped before removing that lock; do not remove an active
writer's lock. Existing study files and SHA mismatches are never silently replaced.

### Keeping saved runs safe

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
- **Each file is checked once per run.** Inside a run, a saved file that does not
  change (the frozen study, window inputs, saved models, forecasts and sealed
  results) is downloaded and its SHA-256 checked once. Later steps reuse those
  checked bytes, so nothing is downloaded or hashed again between fits. The run
  trusts that nothing changes its files while it runs. A change to the saved study
  is found by the next run that loads it.
- **Every-stage checks are optional.** Add `"file_checks": "every_stage"` to
  `study_info` (or pass `file_checks="every_stage"` to `tune_study`) when other
  people or model code may change files during a run. Each stage then also ends
  with a fresh download and check: after the coverage check, at the end of the
  parameter search, at the end of the workflow and before results are published.
  A file changed during a stage, for example by model code between fits, stops
  the run there. The run records which checks it made in its configuration stage.
- **Registration keeps its own checks** whatever the run chooses. It checks again,
  from fresh bytes, everything its `prepare` hook could have changed. Outside a run,
  every read downloads and checks fresh bytes.

For a cleanup example, do not delete Monday's study while Tuesday's run still
reads it. Wait until all readers and writers have stopped, then follow your
storage-retention rules. A stopped writer may also leave a lock: confirm it has
stopped before removing that lock. Never remove a lock belonging to an active run.

Registration records now use formats v3 and v4. Older registration records cannot be reused.
To register the same inputs with this version, choose a new `storage.root`. If you use
Delta registry tables, choose a new `storage.namespace` or new table names too.
For example, keep an old registry at `studies-v1` and put new registrations in `studies-v3`.
Existing registries are left intact; this version does not migrate their records.

A reusable file capture can be released after registration finishes. Each child
study has its own verified files, so it remains usable. A reusable table capture
stays shared: release its dependent studies and close active readers first.
For either mode, `registry.release(capture_id)` prevents new studies from using
that capture. It never substitutes the current source data.

## Read SQL Server source tables

Install the optional driver with `pip install '.[sqlserver]'` from the library checkout.
This installs `mssql-python`, which SQL Server input reads require. Use the same
driver when supplying a Python connection yourself. Put your connection string
in an environment variable, then name that variable in the configuration.
Do not paste a password into the configuration file.

```yaml
connections:
  source:
    kind: sqlserver
    dsn_env: FORECAST_SOURCE_DSN
source:
  connection: source
  sql: queries/observations.sql
  tables:
    observations: dbo.daily_observations
```

The SQL file can contain `SELECT item, day, value FROM observations`. The alias
`observations` refers to the declared table. Primary and audit queries share one
SNAPSHOT transaction, so an update arriving between those queries cannot mix two
versions of the rows. A database administrator must enable snapshot isolation
first; AutoForecast does not change that database setting.

This reads a source into bounded batches through the Python driver. Choose the
retained-input destination separately. Columns requiring sub-microsecond timestamps
or unsupported native types need an explicit cast in the SQL. No automatic
precision loss is accepted.


## Set up a protected SQL Server archive

Use SQL Server 2022 and install `pip install '.[sqlserver]'`. The optional driver
is pinned to a version checked by the database tests. A database administrator
creates a dedicated database, a user `WITHOUT LOGIN` to own the archive, and three
distinct SQL login/user pairs for registration, reading and release. Each pair
uses the same name for its login and database user.

Run the setup helper once with the administrator's connection to that database:

```python
import os
from autoforecast.integrations.sqlserver import connect, provision_snapshot_schema

# Create a protected archive using the accounts prepared by the administrator.
with connect(os.environ["AF_ADMIN_DSN"]) as admin:
    provision_snapshot_schema(
        admin,
        schema="forecast_archive",
        owner="forecast_archive_owner",
        registration_role="forecast_register",
        reader_role="forecast_reader",
        release_role="forecast_release",
    )
```

The helper creates a new schema. Registration builds and seals each capture in
one transaction. Its temporary insert permission is removed before that
transaction finishes. For example, the registration account can save Monday's
input, but cannot later change its sales values or delete it. Cleanup uses the
separate release account.

```yaml
connections:
  archive:
    kind: sqlserver
    dsn_env: AF_READER_DSN
    writer_dsn_env: AF_WRITER_DSN
    release_dsn_env: AF_RELEASE_DSN
storage:
  root: /your/study-registry
  snapshots:
    connection: archive
    namespace: forecast_archive
snapshot_mode: table
```

The variables hold standard SQL Server connection strings, each using its own
account and the archive database. Training jobs do not need release credentials.
Keep ordinary source accounts separate from administrators and the release
account too.
Use dedicated accounts with direct permissions, without database or server role
memberships. Ordinary accounts must not be allowed to act as another user or login,
including through a role or `PUBLIC`. For example, an account that can act as `dbo`
can bypass table protection and is refused.

When the source and archive both use SQL Server, table modes require them to be
on the same SQL Server instance. The source can be in a different database on
that instance. An administrator must also map the source login to an ordinary
user in the archive database:

```sql
-- Run in the archive database, using your own database and account names.
USE [forecast_archive_db];
CREATE USER [forecast_source] FOR LOGIN [forecast_source];
-- Let this account connect so its permissions can be checked.
GRANT CONNECT TO [forecast_source];
```

The source account needs no access to archive tables. Give it the required
`SELECT` permissions in its source database. Keep its archive user and login
mapping unchanged until its captures have been released. This lets reopening
check that the original source account still cannot change saved inputs.
Different-server or unverifiable source connections are refused in table modes.
Hash-only inputs and standalone source reads do not need this archive mapping.
If supplying connection objects yourself, use separate, idle autocommit
connections for the source and archive reader.

The helper grants the reader access to security metadata so the library can
check the actual permissions. It does not enable `TRUSTWORTHY`, cross-database
ownership chaining or source snapshot isolation. The first two must remain off.
Reopening a capture fails if its identity, schema, installed sealing routine or
protection rules change. Database administrators and storage operators remain
trusted, as explained in the input-mode example above.
