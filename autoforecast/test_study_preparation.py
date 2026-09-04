"""Public preparation handoff checks use synthetic sensor populations only."""

import dataclasses
import datetime as dt
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import local
from urllib.parse import unquote, urlparse

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pyspark.sql import functions as F

from autoforecast import register_study
from autoforecast.study_registration import (
    EXCLUSION_REASON_COLUMN,
    START_COLUMN,
    PreparationResult,
    StudyRegistry,
)
from autoforecast.test_study_registration import configuration
from autoforecast.test_study_tuning import spark as _spark


@pytest.fixture(scope="module")
def spark():
    yield from _spark.__wrapped__()


class SensorPreparation:
    def validate(self, settings):
        if type(settings.parameters.get("exclude_count", 0)) is not int:
            raise ValueError("exclude_count must be an integer")

    def prepare(self, context):
        count = context.settings.parameters.get("exclude_count", 0)
        excluded = context.starting_keys.filter(F.col("item") >= f"sensor-{100 - count:03d}")
        retained = context.starting_keys.join(excluded, ["item"], "left_anti")
        return PreparationResult(
            context.observations.join(retained, ["item"], "inner"), context.starting_keys,
            retained, excluded.withColumn(EXCLUSION_REASON_COLUMN, F.lit("APPROVED_SENSOR_EXCLUSION")),
            "sensor-eligibility", "1", {"exclude_count": count},
            ("apply declared sensor eligibility",),
        )


def test_formats_retry_evidence_and_offline_inputs(spark, tmp_path):
    """Population: two synthetic sensors, both retained, with no eligibility exclusions."""
    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    value["records"] = {"format": "yaml"}
    path.write_text(yaml.safe_dump(value))
    registry = StudyRegistry.from_config(path, spark=spark)
    hook = SensorPreparation()
    study_id = register_study(path, registry=registry, prepare=hook)
    successful_audit = next((registry.root / "checks").glob("*/audit.jsonl"))
    definition, _ = registry.definition(study_id)
    record_path = registry.root / definition["admission_evidence"]["path"]
    original = record_path.read_bytes()
    baseline = registry.root / "registrations" / (study_id + ".yaml")
    baseline_bytes = baseline.read_bytes()
    admission = yaml.safe_load(original)
    assert admission["outcome"]["starting"] == admission["outcome"]["final_retained"] == 2
    assert not (record_path.parent / "prepared").exists()
    assert not (record_path.parent / "reference_input").exists()
    assert spark.read.parquet(str(record_path.parent / "exclusions")).count() == 0
    # Same semantic input and repartitioned rows, different config and output encodings.
    value["records"] = {"format": "json"}
    json_path = tmp_path / "study.json"
    json_path.write_text(json.dumps(value, ensure_ascii=False))
    class Repartitioned(SensorPreparation):
        def prepare(self, context):
            result = super().prepare(context)
            return dataclasses.replace(result, observations=result.observations.repartition(2))

    assert register_study(json_path, registry=registry, prepare=Repartitioned()) == study_id
    assert record_path.read_bytes() == original and baseline.read_bytes() == baseline_bytes
    assert not baseline.with_suffix(".json").exists()
    other = StudyRegistry(tmp_path / "other-registry", spark=spark)
    assert register_study(json_path, registry=other, prepare=hook) == study_id
    assert (other.root / "registrations" / (study_id + ".json")).exists()
    other_definition, _ = other.definition(study_id)
    other_record = other.root / other_definition["admission_evidence"]["path"]
    other_bytes = other_record.read_bytes()
    changed = json.loads(other_bytes)
    changed["identity"]["preparation"]["policy"] = {"exclude_count": 99}
    other_record.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="baseline"):
        other.load_study(study_id)
    other_record.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        other.load_study(study_id)
    other_record.write_bytes(other_bytes)
    # Material policy changes must not silently reuse the immutable study.
    class DifferentPolicy(SensorPreparation):
        def prepare(self, context):
            return dataclasses.replace(super().prepare(context), policy={"rule": "reviewed"})

    changed_id = register_study(json_path, registry=registry, prepare=DifferentPolicy())
    assert changed_id != study_id
    from autoforecast.benchmark import digest, parquet_content_hash

    record_path.write_bytes(b"\n" + original)
    with pytest.raises(ValueError, match="bytes differ"):
        registry.load_study(study_id)
    record_path.write_bytes(original)
    for wrong in (None, registry.definition(changed_id)[0]["admission_evidence"]):
        envelope = yaml.safe_load(baseline_bytes)
        if wrong is None:
            envelope["definition"].pop("admission_evidence")
        else:
            envelope["definition"]["admission_evidence"] = wrong
        envelope["sha256"] = digest(envelope["definition"])
        baseline.write_text(yaml.safe_dump(envelope))
        try:
            with pytest.raises(ValueError, match="admission evidence"):
                registry.load_study(study_id)
        finally:
            baseline.write_bytes(baseline_bytes)
    artifact = next((record_path.parent / "starting_keys").glob("*.parquet"))
    before = artifact.read_bytes()
    pq.write_table(pq.read_table(artifact), artifact, compression="gzip")
    try:
        assert parquet_content_hash(artifact.parent) == admission["artifacts"]["starting_keys"]["logical_sha256"]
        assert artifact.read_bytes() != before
        with pytest.raises(ValueError, match="artifact differs"):
            registry.load_study(study_id)
    finally:
        artifact.write_bytes(before)
    for field in ("policy", "version"):
        changed = yaml.safe_load(original)
        changed["identity"]["preparation"][field] = "changed"
        record_path.write_text(yaml.safe_dump(changed))
        try:
            with pytest.raises(ValueError, match="baseline"):
                registry.load_study(study_id)
            with pytest.raises(ValueError, match="baseline"):
                register_study(json_path, registry=registry, prepare=hook)
        finally:
            record_path.write_bytes(original)
    dual = record_path.with_suffix(".json")
    dual.write_text(json.dumps(admission))
    try:
        with pytest.raises(ValueError, match="conflicting"):
            registry.definition(study_id)
    finally:
        dual.unlink()
    for role in ("starting_keys", "exclusions", "source_findings"):
        artifact = next((record_path.parent / role).glob("*.parquet"))
        artifact_bytes = artifact.read_bytes()
        artifact.write_bytes(artifact_bytes + b"changed")
        try:
            with pytest.raises(ValueError):
                registry.load_study(study_id)
        finally:
            artifact.write_bytes(artifact_bytes)
    path.unlink()
    json_path.unlink()
    (tmp_path / "source.parquet").unlink()
    with registry.load_study(study_id) as study:
        assert study.training_data.count() == 9
        assert study.validation_data.filter("value is null").count() == 1
    audit = [json.loads(line) for line in successful_audit.read_text().splitlines()]
    assert any(e["stage"] == "Registration inputs" and e["status"] == "INFO" for e in audit)
    assert any(e["stage"] == "Business preparation and input verification"
               and e["status"] == "PASS" for e in audit)
    assert [(e["stage"], e["status"]) for e in audit if e.get("plan_stage")] == [
        (stage, status) for stage in ("configuration", "sources", "admission", "freeze", "publication")
        for status in ("START", "PASS")
    ]


@pytest.mark.parametrize("late_series", [False, True])
def test_exact_hundred_ninety_five_ninety_three_population(spark, tmp_path, late_series):
    """100 sensor IDs; five application exclusions, then two native exclusions.

    Native exclusions are two invalid references, or one reference and one late series.
    """
    path = configuration(tmp_path)
    pq.write_table(pa.Table.from_pylist([
        {"item": f"sensor-{sid:03d}", "day": dt.date(2024, 1, day), "value": float(day)}
        for sid in range(100) for day in range(1, 10)
        if not late_series or sid != 94 or day == 9
    ]), tmp_path / "source.parquet")
    value = yaml.safe_load(path.read_text())
    value["dataset"]["schema_version"] = 2
    value["dataset"]["dataset"]["required_series_count"] = 99 if late_series else 100
    if late_series:
        value["recipe"]["new_series_policy"] = "exclude_after_first_origin"
    value["preparation"] = {"exclude_count": 5}
    value["references"] = {"prepared": True}
    value["invalid_references"] = "exclude"
    path.write_text(yaml.safe_dump(value))

    class ReferencePreparation(SensorPreparation):
        def prepare(self, context):
            result = super().prepare(context)
            windows = context.observations.sparkSession.createDataFrame(
                [(stage, origin) for stage, origin, _ in context.settings.windows],
                "stage string, origin string",
            )
            refs = result.retained_keys.select(F.struct("item").alias("keys")).crossJoin(windows)
            refs = refs.withColumn("weight", F.lit(1.0)).withColumn(
                "scale", F.when(F.col("keys.item").isin("sensor-093", "sensor-094"), 0.0)
                .otherwise(1.0),
            )
            return dataclasses.replace(result, scoring_references=refs)

    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry, prepare=ReferencePreparation())
    definition, _ = registry.definition(study_id)
    evidence = registry.root / definition["admission_evidence"]["path"]
    record = json.loads(evidence.read_bytes())
    assert {name: record["outcome"][name] for name in (
        "starting", "application_retained", "final_retained",
    )} == {"starting": 100, "application_retained": 95, "final_retained": 93}
    final = {r.item for r in spark.read.parquet(str(evidence.parent / "final_keys")).collect()}
    assert final == {f"sensor-{sid:03d}" for sid in range(93)}
    native = spark.read.parquet(str(evidence.parent / "native_exclusions")).collect()
    assert {(r.item, r[EXCLUSION_REASON_COLUMN]) for r in native} == {
        ("sensor-093", "INVALID_REFERENCE"),
        ("sensor-094", "NEW_SERIES_POLICY" if late_series else "INVALID_REFERENCE"),
    }
    assert spark.read.parquet(str(evidence.parent / "exclusions")).count() == 5
    with registry.load_study(study_id) as study:
        assert study.training_data.select("item").distinct().count() == 93


@pytest.mark.parametrize("case", ["wrong_start", "invented", "lost", "overlap", "no_reason"])
def test_exact_membership_rejects_equal_counts_and_unexplained_changes(spark, tmp_path, case):
    path = configuration(tmp_path)

    class Invalid(SensorPreparation):
        def prepare(self, context):
            result = super().prepare(context)
            if case == "wrong_start":
                return dataclasses.replace(result, starting_keys=result.starting_keys.withColumn(
                    "item", F.when(F.col("item") == "A", "C").otherwise(F.col("item"))))
            if case == "invented":
                return dataclasses.replace(result, observations=result.observations.withColumn(
                    "item", F.when(F.col("item") == "A", "C").otherwise(F.col("item"))))
            if case == "lost":
                return dataclasses.replace(result, retained_keys=result.retained_keys.filter("item='A'"))
            excluded = result.starting_keys.filter("item='A'").withColumn(
                EXCLUSION_REASON_COLUMN, F.lit("" if case == "no_reason" else "approved"))
            return dataclasses.replace(result, exclusions=excluded)

    registry = StudyRegistry.from_config(path, spark=spark)
    with pytest.raises(ValueError, match="membership|nonblank"):
        register_study(path, registry=registry, prepare=Invalid())
    assert not list((registry.root / "registrations").glob("*.*"))
    attempt = json.loads(next((registry.root / "checks").glob("*/attempt.json")).read_text())
    assert attempt == {"status": "failed", "stage": "admission", "exception": "ValueError"}
    assert list((registry.root / "checks").glob("*/source/*.parquet"))


@pytest.mark.parametrize("case", ["native", "domain", "nested_duplicate", "audit_query"])
def test_all_configuration_validates_before_source_access(spark, tmp_path, monkeypatch, case):
    from autoforecast.integrations import delta

    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    if case == "native":
        value["unrecognized"] = True
    elif case == "domain":
        value["preparation"] = {"exclude_count": "five"}
    elif case == "audit_query":
        (tmp_path / "primary.sql").write_text("SELECT * FROM observations")
        (tmp_path / "audit.sql").write_text("SELECT * FROM undeclared")
        value["source"] = {"kind": "delta", "sql": "primary.sql",
                           "tables": {"observations": "neutral.data"},
                           "audit_queries": {"original": "audit.sql"}}
    path.write_text(yaml.safe_dump(value))
    if case == "nested_duplicate":
        path = tmp_path / "study.json"
        path.write_text(json.dumps(value).replace('"metric": "mae"',
                                                 '"metric": "mae", "metric": "mae"'))
    monkeypatch.setattr(delta, "latest_delta_version", lambda *_: pytest.fail("source lookup"))

    class NoRead(SensorPreparation):
        def prepare(self, context):
            pytest.fail("preparation must not execute")

    with pytest.raises(ValueError):
        register_study(path, spark=spark, prepare=NoRead())
    assert not (tmp_path / "registry").exists()


@pytest.mark.parametrize("case", ["application_failure", "native_failure", "source_mutation"])
def test_failed_or_mutating_preparation_retains_evidence_without_publication(
    spark, tmp_path, case,
):
    path = configuration(tmp_path)

    class Failing(SensorPreparation):
        def prepare(self, context):
            result = super().prepare(context)
            if case == "application_failure":
                issues = context.findings.withColumn("status", F.lit("failed")).withColumn(
                    "issue_code", F.lit("SENSOR_REJECTED"))
                return dataclasses.replace(result, findings=issues)
            if case == "native_failure":
                return dataclasses.replace(result, observations=result.observations.unionByName(
                    result.observations.limit(1)), findings=context.findings.withColumn(
                        "status", F.lit("passed")))
            original = context.observations
            independent = spark.createDataFrame(original.collect(), original.schema)
            part = unquote(urlparse(original.inputFiles()[0]).path)
            table = pq.read_table(part)
            index = table.schema.get_field_index("value")
            pq.write_table(table.set_column(index, table.schema.field(index),
                                            pa.array([99.0] * len(table))), part)
            return dataclasses.replace(result, observations=independent)

    registry = StudyRegistry.from_config(path, spark=spark)
    with pytest.raises(ValueError, match="block registration|changed during preparation"):
        register_study(path, registry=registry, prepare=Failing())
    assert not list((registry.root / "registrations").glob("*.*"))
    assert list((registry.root / "checks").glob("*/attempt.json"))
    if case == "application_failure":
        report = spark.read.parquet(str(next((registry.root / "checks").glob("*/application"))))
        assert report.filter("issue_code = 'SENSOR_REJECTED' AND status = 'failed'").count() > 0
    if case == "native_failure":
        report = spark.read.parquet(str(next((registry.root / "checks").glob("*/prepared"))))
        assert report.filter("issue_code = 'DUPLICATE_DATE' AND status = 'failed'").count() > 0


def test_evidence_write_failure_and_interrupted_retry_keep_completed_baseline(
    spark, tmp_path, monkeypatch,
):
    from autoforecast import study_registration as registration

    path = configuration(tmp_path)
    registry = StudyRegistry.from_config(path, spark=spark)
    old_id = register_study(path, registry=registry, prepare=SensorPreparation())
    old = registry.definition(old_id)

    class Revised(SensorPreparation):
        def prepare(self, context):
            return dataclasses.replace(super().prepare(context), version="2")

    encode = registration.record_bytes

    def fail_evidence(value, *, format):
        if isinstance(value, dict) and value.get("schema") == "autoforecast.admission.v1":
            raise OSError("injected admission write failure")
        return encode(value, format=format)

    with monkeypatch.context() as patch:
        patch.setattr(registration, "record_bytes", fail_evidence)
        with pytest.raises(OSError, match="admission write failure"):
            register_study(path, registry=registry, prepare=Revised())
    assert registry.definition(old_id) == old
    assert len(list((registry.root / "registrations").glob("*.json"))) == 1
    new_id = register_study(path, registry=registry, prepare=Revised())
    assert new_id != old_id and registry.definition(old_id) == old
    assert registry.definition(new_id)[0]["study_id"] == new_id


def test_concurrent_studies_and_late_policy_mutation_keep_private_baselines(
    spark, tmp_path, monkeypatch,
):
    from autoforecast import study_registration as registration

    path = configuration(tmp_path)
    registry = StudyRegistry.from_config(path, spark=spark)
    state = local()

    class Policy(SensorPreparation):
        def __init__(self, name):
            self.name = name

        def prepare(self, context):
            policy = {"choice": self.name}
            state.policy = policy
            # A detached settings view cannot change the native recipe.
            context.settings.recipe["actual_missing"] = "zero"
            return dataclasses.replace(super().prepare(context), policy=policy,
                                       observations=context.observations.repartition(2))

    check = registration._check_key_frame

    def mutate_after_capture(*args, **kwargs):
        if kwargs["label"] == "declared_starting":
            state.policy["choice"] = "late mutation"
        return check(*args, **kwargs)

    monkeypatch.setattr(registration, "_check_key_frame", mutate_after_capture)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(register_study, path, registry=registry, prepare=Policy(name))
                   for name in ("first", "second")]
        ids = [future.result() for future in futures]
    assert ids[0] != ids[1]
    for study_id, name in zip(ids, ("first", "second"), strict=True):
        definition, _ = registry.definition(study_id)
        record = json.loads((registry.root / definition["admission_evidence"]["path"]).read_text())
        assert record["identity"]["preparation"]["policy"] == {"choice": name}
        assert definition["recipe"]["actual_missing"] == "mask"


def test_one_register_call_two_models_and_fresh_process_offline_replay(spark, tmp_path):
    from autoforecast import run_study
    from autoforecast.test_study_tuning import config, implementation, predict

    path = configuration(tmp_path)
    registry = StudyRegistry.from_config(path, spark=spark)
    calls = []

    class Once(SensorPreparation):
        def validate(self, settings):
            calls.append("validate")
            super().validate(settings)

        def prepare(self, context):
            calls.append("prepare")
            return super().prepare(context)

    study_id = register_study(path, registry=registry, prepare=Once())
    profile = config(tmp_path)
    results = []
    for level in (4.0, 10.0):
        def model(request, expected=level):
            assert request.model_parameters["level"] == expected
            return predict(request)

        results.append(run_study(study_id, model, registry=registry, model_config=profile,
                                 implementation=implementation(),
                                 fixed_parameters={"model.level": level}, log_level="ERROR"))
    assert calls == ["validate", "prepare"]
    assert results[0].study_id == results[1].study_id == study_id
    assert results[0].testing_metrics["mae"] != results[1].testing_metrics["mae"]
    path.unlink()
    (tmp_path / "source.parquet").unlink()
    code = """
import sys
from pyspark.sql import SparkSession
from autoforecast.study_registration import StudyRegistry
from autoforecast.integrations import delta
def forbidden(*args, **kwargs):
    raise AssertionError('live source access is forbidden')
delta.read_source = delta.read_sources = delta.latest_delta_version = forbidden
spark = SparkSession.builder.master('local[1]').config('spark.ui.enabled','false').getOrCreate()
try:
    registry = StudyRegistry(sys.argv[1], spark=spark)
    with registry.load_study(sys.argv[2]) as study:
        assert study.training_data.count() == 9
    print('OFFLINE_REPLAY_PASSED')
finally:
    spark.stop()
"""
    result = subprocess.run([sys.executable, "-c", code, str(registry.root), study_id],
                            capture_output=True, text=True, env={**os.environ,
                                                               "PYSPARK_PYTHON": sys.executable})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OFFLINE_REPLAY_PASSED" in result.stdout


def test_declared_starts_and_reference_input_reuse_native_calculator(spark, tmp_path):
    from autoforecast.benchmark import artifact_path

    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    value['references'] = {'weight_col': 'importance', 'missing_history': 'na',
                           'duplicate_policy': 'error', 'scale_start': 'history_start'}
    path.write_text(yaml.safe_dump(value))

    class Calculated(SensorPreparation):
        def prepare(self, context):
            result = super().prepare(context)
            starts = context.observations.groupBy('item').agg(
                F.min('day').cast('string').alias(START_COLUMN))
            reference = context.observations.withColumn('value', F.dayofmonth('day').cast('double'))
            reference = reference.withColumn('importance', F.lit(2.0))
            return dataclasses.replace(result, starts=starts, reference_input=reference,
                                       planned_actions=('inspect future sensor arrivals',))

    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry, prepare=Calculated())
    definition, _ = registry.definition(study_id)
    record_path = registry.root / definition['admission_evidence']['path']
    admission = json.loads(record_path.read_bytes())
    assert admission['identity']['preparation']['planned_actions'] == ['inspect future sensor arrivals']
    assert not (record_path.parent / 'reference_input').exists()
    assert spark.read.parquet(str(record_path.parent / 'starts')).count() == 2
    with registry.load_study(study_id) as study:
        refs = pq.read_table(artifact_path(study._path, study._record(), 'scoring_references')).to_pylist()
        tuning = [row for row in refs if row['boundary'].startswith('tuning/')]
        assert {json.loads(row['series_id'])['item']: row['weight'] for row in tuning} == {'A': 6.0, 'B': 4.0}
        assert {row['scale'] for row in tuning} == {1.0}
        # Reference-only repair must not impute held-back observations.
        assert study.validation_data.filter('value is null').count() == 1
        assert study.training_data.count() == 9


def test_changed_returned_materialization_fails_before_freezing(spark, tmp_path, monkeypatch):
    from autoforecast import study_registration as registration

    path = configuration(tmp_path)
    registry = StudyRegistry.from_config(path, spark=spark)
    check = registration._same_keys
    changed = []

    def mutate_after_membership(*args, **kwargs):
        result = check(*args, **kwargs)
        if args[-1] == 'prepared observations':
            part = next((registry.root / 'checks').glob('*/admission-staging/prepared/*.parquet'))
            table = pq.read_table(part)
            index = table.schema.get_field_index('value')
            pq.write_table(table.set_column(index, table.schema.field(index),
                                            pa.array([77.0] * len(table))), part)
            # Keep replacement rows readable; admission integrity must detect
            # the change independently of Hadoop's optional checksum sidecar.
            part.with_name('.' + part.name + '.crc').unlink(missing_ok=True)
            changed.append(part)
        return result

    monkeypatch.setattr(registration, '_same_keys', mutate_after_membership)
    with pytest.raises(ValueError, match='materialization changed'):
        register_study(path, registry=registry, prepare=SensorPreparation())
    assert changed and not list((registry.root / 'registrations').glob('*.*'))


def test_concurrent_identical_admissions_and_derived_study_keep_parent_evidence(spark, tmp_path):
    from autoforecast import run_study
    from autoforecast.test_study_tuning import config, implementation, predict

    path = configuration(tmp_path)
    settings = yaml.safe_load(path.read_text())
    settings["records"] = {"format": "yaml"}
    path.write_text(yaml.safe_dump(settings))
    registry = StudyRegistry.from_config(path, spark=spark)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(register_study, path, registry=registry, prepare=SensorPreparation())
                   for _ in range(2)]
        ids = [future.result() for future in futures]
    assert ids[0] == ids[1]
    parent, _ = registry.definition(ids[0])
    assert len(list((registry.root / 'registrations').glob('*.*'))) == 1
    profile = config(tmp_path)
    value = yaml.safe_load(profile.read_text())
    value['series_policy'] = 'exclude_unavailable'
    value['minimum_history'] = 3
    profile.write_text(yaml.safe_dump(value))
    result = run_study(ids[0], predict, registry=registry, model_config=profile,
                       implementation=implementation(), fixed_parameters={'model.level': 4.0},
                       log_level='ERROR')
    assert result.study_id != ids[0] and result.parent_study_id == ids[0]
    derived, _ = registry.definition(result.study_id)
    assert derived['admission_evidence'] == parent['admission_evidence']
    assert (registry.root / 'registrations' / (result.study_id + '.yaml')).exists()
    assert not (registry.root / 'registrations' / (result.study_id + '.json')).exists()
    child_path = registry.root / 'registrations' / (result.study_id + '.yaml')
    original = child_path.read_bytes()
    manifest = registry.root / 'benchmarks' / result.study_id / 'benchmark.json'
    assert registry.retain_derived(ids[0], manifest) == result.study_id
    assert child_path.read_bytes() == original
    with registry.load_study(result.study_id) as study:
        assert study.n_series == 1


@pytest.mark.parametrize('case', [
    'result_type', 'identifier', 'policy', 'actions', 'frame_type', 'key_type',
    'duplicate_keys', 'invalid_start', 'both_references', 'finding_status', 'finding_count',
    'missing_columns', 'extra_columns', 'null_keys', 'null_reason', 'reason_type',
    'version', 'planned_actions', 'finding_schema', 'null_status', 'null_count',
    'starts_missing', 'starts_added', 'settings_payload',
])
def test_invalid_preparation_contract_fails_without_publication(spark, tmp_path, case):
    path = configuration(tmp_path)
    registry = StudyRegistry.from_config(path, spark=spark)

    class Invalid(SensorPreparation):
        def prepare(self, context):
            result = super().prepare(context)
            if case == 'result_type':
                return {'observations': result.observations}
            if case == 'settings_payload':
                object.__setattr__(context.settings, '_payload', context.settings._payload + b' ')
                return result
            changes = {
                'identifier': {'identifier': ' '},
                'policy': {'policy': {'nested': {'rule': 'unsupported'}}},
                'actions': {'actions': ['untyped action']},
                'frame_type': {'observations': None},
                'key_type': {'retained_keys': result.retained_keys.withColumn('item', F.lit(1))},
                'duplicate_keys': {'retained_keys': result.retained_keys.unionByName(result.retained_keys)},
                'invalid_start': {'starts': result.retained_keys.withColumn(START_COLUMN, F.lit('not-a-date'))},
                'both_references': {'reference_input': result.observations,
                                    'scoring_references': result.observations},
                'finding_status': {'findings': context.findings.withColumn('status', F.lit('unknown'))},
                'finding_count': {'findings': context.findings.withColumn('affected_rows', F.lit(-1).cast('long'))},
                'missing_columns': {'exclusions': result.exclusions.drop('item')},
                'extra_columns': {'retained_keys': result.retained_keys.withColumn('extra', F.lit(1))},
                'null_keys': {'retained_keys': result.retained_keys.withColumn('item', F.lit(None).cast('string'))},
                'null_reason': {'exclusions': result.starting_keys.withColumn(EXCLUSION_REASON_COLUMN, F.lit(None).cast('string'))},
                'reason_type': {'exclusions': result.exclusions.withColumn(EXCLUSION_REASON_COLUMN, F.lit(1))},
                'version': {'version': ' '},
                'planned_actions': {'planned_actions': (' ',)},
                'finding_schema': {'findings': context.findings.drop('status')},
                'null_status': {'findings': context.findings.withColumn('status', F.lit(None).cast('string'))},
                'null_count': {'findings': context.findings.withColumn('affected_rows', F.lit(None).cast('long'))},
                'starts_missing': {'starts': result.retained_keys.filter("item='A'").withColumn(START_COLUMN, F.lit('2024-01-01'))},
                'starts_added': {'starts': result.retained_keys.withColumn('item', F.lit('C')).withColumn(START_COLUMN, F.lit('2024-01-01'))},
            }
            return dataclasses.replace(result, **changes[case])

    with pytest.raises(ValueError, match=("provide reference input or precomputed references"
                                         if case == 'both_references' else None)):
        register_study(path, registry=registry, prepare=Invalid())
    assert not list((registry.root / 'registrations').glob('*.*'))
    assert list((registry.root / 'checks').glob('*/attempt.json'))


@pytest.mark.parametrize('key', ['reason', 'Reason', 'start', 'Start'])
def test_preparation_metadata_does_not_collide_with_valid_source_keys(spark, tmp_path, key):
    # Three sensors: application excludes the second, references exclude the third.
    path = configuration(tmp_path)
    source = tmp_path / 'source.parquet'
    rows = pq.read_table(source).to_pylist()
    rows += [{**row, 'item': 'C'} for row in rows if row['item'] == 'A']
    integral = key.islower()
    ids = {sid: i if integral else f'sensor-{i}' for i, sid in enumerate(('A', 'B', 'C'), 1)}
    pq.write_table(pa.Table.from_pylist([
        {key: ids[row['item']], 'day': row['day'], 'value': row['value']} for row in rows
    ]), source)
    value = yaml.safe_load(path.read_text())
    value['recipe']['source_keys'] = [key]
    value['dataset']['dataset']['panel']['series_col'] = key
    value['references'] = {'prepared': True}
    value['invalid_references'] = 'exclude'
    path.write_text(yaml.safe_dump(value))

    class Keep:
        def validate(self, settings):
            pass

        def prepare(self, context):
            excluded = context.starting_keys.filter(F.col(key) == ids['B'])
            retained = context.starting_keys.join(excluded, [key], 'left_anti')
            windows = spark.createDataFrame(
                [(stage, origin) for stage, origin, _ in context.settings.windows],
                'stage string, origin string',
            )
            references = retained.select(F.struct(key).alias('keys')).crossJoin(windows)
            references = references.withColumn('weight', F.lit(1.0)).withColumn(
                'scale', F.when(F.col('keys')[key] == ids['C'], 0.0).otherwise(1.0))
            return PreparationResult(
                context.observations.join(retained, [key], 'inner'), context.starting_keys, retained,
                excluded.withColumn(EXCLUSION_REASON_COLUMN, F.lit('APPROVED_SENSOR_EXCLUSION')),
                'sensor-membership', '1', {}, scoring_references=references,
                starts=context.observations.join(retained, [key], 'inner').groupBy(key).agg(
                    F.min('day').cast('string').alias(START_COLUMN)),
            )

    registry = StudyRegistry.from_config(path, spark=spark)
    study_id = register_study(path, registry=registry, prepare=Keep())
    definition, _ = registry.definition(study_id)
    evidence = registry.root / definition['admission_evidence']['path']
    native = spark.read.parquet(str(evidence.parent / 'native_exclusions'))
    assert native.schema[key].dataType.typeName() == ('long' if integral else 'string')
    assert {(r[key], r[EXCLUSION_REASON_COLUMN]) for r in native.collect()} == {
        (ids['C'], 'INVALID_REFERENCE'),
    }
    application = spark.read.parquet(str(evidence.parent / 'exclusions'))
    assert {(r[key], r[EXCLUSION_REASON_COLUMN]) for r in application.collect()} == {
        (ids['B'], 'APPROVED_SENSOR_EXCLUSION'),
    }
    with registry.load_study(study_id) as study:
        assert study.n_series == 1
        assert {r[key] for r in study.training_data.select(key).distinct().collect()} == {ids['A']}


@pytest.mark.parametrize('case', ['valid', 'missing_retained', 'unexplained'])
def test_application_exclusions_prune_only_declared_excluded_starts(spark, tmp_path, case):
    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    values = [{'keys': {'item': 'A'}, 'start': '2024-01-01'},
              {'keys': {'item': 'B'}, 'start': '2024-01-02'}]
    if case == 'missing_retained':
        values = values[1:]
    elif case == 'unexplained':
        values.append({'keys': {'item': 'C'}, 'start': '2024-01-01'})
    value['recipe']['series_starts'] = {'rule': 'declared', 'values': values}
    path.write_text(yaml.safe_dump(value))

    class ExcludeB(SensorPreparation):
        def prepare(self, context):
            result = super().prepare(context)
            retained = context.starting_keys.filter("item='A'")
            excluded = context.starting_keys.filter("item='B'")
            return dataclasses.replace(
                result, observations=context.observations.join(retained, ['item'], 'inner'),
                retained_keys=retained,
                exclusions=excluded.withColumn(EXCLUSION_REASON_COLUMN, F.lit('approved')),
            )

    registry = StudyRegistry.from_config(path, spark=spark)
    if case != 'valid':
        with pytest.raises(ValueError, match='declared starts.*membership'):
            register_study(path, registry=registry, prepare=ExcludeB())
        assert not list((registry.root / 'registrations').glob('*.*'))
    else:
        study_id = register_study(path, registry=registry, prepare=ExcludeB())
        assert registry.definition(study_id)[0]['recipe']['series_starts']['values'] == values[:1]
        with registry.load_study(study_id) as study:
            assert study.n_series == 1 and study.training_data.count() == 5


@pytest.mark.parametrize('case', [
    'record_format', 'record_field', 'nested_parameters', 'parameters_without_hook',
    'audits_without_hook', 'references_without_hook', 'audit_name', 'reserved_audit',
    'population', 'matching_type', 'prepared_false', 'validation_payload',
])
def test_preparation_configuration_controls_fail_before_source_access(
    spark, tmp_path, monkeypatch, case,
):
    from autoforecast import study_registration as registration

    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    if case.startswith('audit') or case in {'reserved_audit', 'population', 'matching_type'}:
        (tmp_path / 'primary.sql').write_text('SELECT * FROM observations')
        value['source'] = {'kind': 'delta', 'sql': 'primary.sql',
                           'tables': {'observations': 'neutral.sensors'},
                           'audit_queries': {'population': 'primary.sql'}}
    if case == 'record_format':
        value['records'] = {'format': 'xml'}
    elif case == 'record_field':
        value['records'] = {'unsupported': True}
    elif case == 'nested_parameters':
        value['preparation'] = {'nested': {}}
    elif case == 'parameters_without_hook':
        value['preparation'] = {'exclude_count': 0}
    elif case in {'references_without_hook', 'prepared_false'}:
        value['references'] = {'prepared': case != 'prepared_false'}
    elif case in {'audit_name', 'reserved_audit'}:
        value['source']['audit_queries'] = {
            'invalid-name' if case == 'audit_name' else 'primary': 'primary.sql'}
    elif case == 'population':
        value['source']['population'] = 'undeclared'
    elif case == 'matching_type':
        value['source']['require_matching_population'] = 1
    path.write_text(yaml.safe_dump(value))

    def forbidden(*args, **kwargs):
        pytest.fail('data lifecycle must not start')

    class NoRead(SensorPreparation):
        def validate(self, settings):
            super().validate(settings)
            if case == 'validation_payload':
                object.__setattr__(settings, '_payload', settings._payload + b' ')

    monkeypatch.setattr(registration, '_register_study', forbidden)
    hook = None if case.endswith('without_hook') else NoRead()
    with pytest.raises(ValueError):
        register_study(path, spark=spark, prepare=hook)
    assert not (tmp_path / 'registry').exists()


@pytest.mark.parametrize('case', [
    'precomputed_without_flag', 'raw_without_settings', 'raw_with_path', 'raw_with_prepared',
    'missing_precomputed', 'invalid_precomputed',
])
def test_preparation_reference_routes_require_matching_configuration(spark, tmp_path, case):
    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    if case in {'raw_with_prepared', 'missing_precomputed', 'invalid_precomputed'}:
        value['references'] = {'prepared': True}
    elif case == 'raw_with_path':
        value['references'] = {'path': 'unused.parquet'}
    path.write_text(yaml.safe_dump(value))

    class References(SensorPreparation):
        def prepare(self, context):
            result = super().prepare(context)
            if case.startswith('raw_'):
                return dataclasses.replace(result, reference_input=context.observations)
            if case == 'missing_precomputed':
                return result
            windows = spark.createDataFrame(
                [(stage, origin) for stage, origin, _ in context.settings.windows],
                'stage string, origin string',
            )
            refs = result.retained_keys.select(F.struct('item').alias('keys')).crossJoin(windows)
            return dataclasses.replace(result, scoring_references=refs.withColumn(
                'weight', F.lit(1.0)).withColumn('scale', F.lit(0.0)))

    registry = StudyRegistry.from_config(path, spark=spark)
    with pytest.raises(ValueError):
        register_study(path, registry=registry, prepare=References())
    assert not list((registry.root / 'registrations').glob('*.*'))
    if case == 'invalid_precomputed':
        reports = list((registry.root / 'checks').glob('*/references'))
        assert len(reports) == 1 and spark.read.parquet(str(reports[0])).count() > 0


@pytest.mark.parametrize('case', ['application_findings', 'secondary_failure'])
def test_failure_evidence_writes_preserve_original_exception_and_block_publication(
    spark, tmp_path, monkeypatch, caplog, case,
):
    from autoforecast import study_registration as registration

    path = configuration(tmp_path)
    registry = StudyRegistry.from_config(path, spark=spark)

    class Fail(SensorPreparation):
        def prepare(self, context):
            if case == 'secondary_failure':
                raise RuntimeError('original preparation failure')
            return dataclasses.replace(super().prepare(context), findings=context.findings)

    save = registration._save_findings
    encode = registration.record_bytes

    def fail_findings(registry, frame, issues_root, name, check_id):
        if name == 'application':
            raise OSError('findings transport unavailable')
        return save(registry, frame, issues_root, name, check_id)

    def fail_attempt(value, *, format):
        if value.get('status') == 'failed':
            raise OSError('attempt record unavailable')
        return encode(value, format=format)

    if case == 'application_findings':
        monkeypatch.setattr(registration, '_save_findings', fail_findings)
        with pytest.raises(OSError, match='findings transport unavailable'):
            register_study(path, registry=registry, prepare=Fail())
        assert list((registry.root / 'checks').glob('*/attempt.json'))
    else:
        monkeypatch.setattr(registration, 'record_bytes', fail_attempt)
        with pytest.raises(RuntimeError, match='original preparation failure') as caught:
            register_study(path, registry=registry, prepare=Fail())
        assert caught.value.__notes__ == ['Registration failure evidence could not be verified.']
        assert 'Registration failure evidence could not be verified.' in caplog.text
    events = [json.loads(line) for line in next(
        (registry.root / 'checks').glob('*/audit.jsonl')).read_text().splitlines()]
    assert any(e['stage'] == 'Registration inputs' and e['status'] == 'INFO' for e in events)
    assert any(e['stage'] == 'Business preparation and input verification'
               and e['status'] == 'FAIL' for e in events)
    assert [(e['stage'], e['status']) for e in events if e.get('plan_stage')] == [
        ('configuration', 'START'), ('configuration', 'PASS'), ('sources', 'START'),
        ('sources', 'PASS'), ('admission', 'START'), ('admission', 'FAIL'),
    ]
    assert not list((registry.root / 'registrations').glob('*.*'))


@pytest.mark.parametrize("route", [
    "configured", "returned", "default", "invalid_returned", "out_of_bounds", "wrong_initial_count",
])
@pytest.mark.parametrize("app_exclude", [False, True])
def test_preparation_preserves_native_initial_cohort(spark, tmp_path, route, app_exclude):
    """Three source IDs; two initial-cohort IDs; optionally exclude B, then exclude late C."""
    path = configuration(tmp_path)
    source = tmp_path / "source.parquet"
    original = pq.read_table(source)
    late = pa.Table.from_pylist([
        {"item": "C", "day": dt.date(2024, 1, 9), "value": 1.0},
    ], schema=original.schema)
    pq.write_table(pa.concat_tables([original, late]), source)
    value = yaml.safe_load(path.read_text())
    value["dataset"]["schema_version"] = 2
    value["dataset"]["dataset"]["required_series_count"] = 1 if route == "wrong_initial_count" else 2
    value["recipe"]["new_series_policy"] = "exclude_after_first_origin"
    if route == "configured":
        value["recipe"]["series_starts"] = {"rule": "declared", "values": [
            {"keys": {"item": "A"}, "start": "2024-01-01"},
            {"keys": {"item": "B"}, "start": "2024-01-02"},
        ]}
    path.write_text(yaml.safe_dump(value))
    if route == "default":
        native_registry = StudyRegistry(tmp_path / "native", spark=spark)
        native = register_study(path, registry=native_registry)
        with native_registry.load_study(native) as study:
            assert study.n_series == 2

    class Prepare(SensorPreparation):
        def prepare(self, context):
            result = super().prepare(context)
            if app_exclude:
                excluded = context.starting_keys.filter("item='B'")
                retained = context.starting_keys.join(excluded, ["item"], "left_anti")
                result = dataclasses.replace(
                    result, retained_keys=retained,
                    exclusions=excluded.withColumn(EXCLUSION_REASON_COLUMN, F.lit("APPROVED")),
                    observations=context.observations.join(retained, ["item"], "inner"),
                )
            if route in {"returned", "invalid_returned", "out_of_bounds"}:
                starts = result.observations.groupBy("item").agg(
                    F.min("day").cast("string").alias(START_COLUMN),
                )
                if route == "invalid_returned":
                    starts = starts.withColumn(START_COLUMN, F.when(
                        F.col("item") == "C", F.lit("not-a-date"),
                    ).otherwise(F.col(START_COLUMN)))
                if route == "out_of_bounds":
                    starts = starts.withColumn(START_COLUMN, F.when(
                        F.col("item") == "A", F.lit("2024-01-06"),
                    ).otherwise(F.col(START_COLUMN)))
                result = dataclasses.replace(result, starts=starts)
            return result

    registry = StudyRegistry(tmp_path / "prepared", spark=spark)
    if route in {"invalid_returned", "out_of_bounds", "wrong_initial_count"}:
        message = "required_series_count" if route == "wrong_initial_count" else None
        with pytest.raises(ValueError, match=message):
            register_study(path, registry=registry, prepare=Prepare())
        assert not list((registry.root / "registrations").glob("*.*"))
        return
    study_id = register_study(path, registry=registry, prepare=Prepare())
    definition, _ = registry.definition(study_id)
    record_path = registry.root / definition["admission_evidence"]["path"]
    record = json.loads(record_path.read_bytes())
    assert {name: record["outcome"][name] for name in (
        "starting", "application_retained", "final_retained",
    )} == {"starting": 3, "application_retained": 2 if app_exclude else 3,
          "final_retained": 1 if app_exclude else 2}
    assert {(row.item, row[EXCLUSION_REASON_COLUMN]) for row in spark.read.parquet(
        str(record_path.parent / "native_exclusions"),
    ).collect()} == {("C", "NEW_SERIES_POLICY")}
    assert {entry["keys"]["item"] for entry in definition["recipe"]["series_starts"]["values"]} == (
        {"A"} if app_exclude else {"A", "B"})
    with registry.load_study(study_id) as study:
        assert {row.item for row in study.training_data.select("item").distinct().collect()} == (
            {"A"} if app_exclude else {"A", "B"})
