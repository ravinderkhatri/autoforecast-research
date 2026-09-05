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
