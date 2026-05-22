"""Tests for the panel and forecast-artifact contracts.

Real M5 Parquet is used wherever the question is about source data.  Small synthetic forecast
artifacts are necessary for corruptions a real adapter must never produce: duplicate keys, NaNs,
wrong dates, and manifest/row-count disagreement.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autoforecast.panel import (
    ELIGIBLE_STATES,
    Forecast,
    ForecastError,
    Panel,
    PanelError,
    PlanSource,
    _any_fragment,
)

M5 = pathlib.Path(__file__).resolve().parent.parent / "data" / "m5_parquet"


def m5_panel(**over: object) -> Panel:
    """The real M5 panel, with fields overridden per test."""
    kw: dict[str, object] = dict(path=M5, series_col="id", time_col="date", target_col="qty",
                                 exposure_col="observation_state", season_length=7)
    kw.update(over)
    return Panel(**kw)  # type: ignore[arg-type]


def _require_m5() -> None:
    if not M5.is_dir():
        pytest.skip("public M5 Parquet is not installed")


def _structural_panel(**over: object) -> Panel:
    """A tiny valid schema for panel-contract cases that do not measure M5."""
    path = pathlib.Path(tempfile.mkdtemp()) / "panel.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["item-store"],
                "date": [dt.date(2016, 1, 1)],
                "qty": [1.0],
                "observation_state": ["exact"],
                "wm_yr_wk": [1],
                "dept_id": ["foods"],
                "cat_id": ["food"],
                "sell_price": [1.5],
                "item_id": ["item"],
                "store_id": ["store"],
                "pe": [1.0],
                "pl": [0.0],
                "pu": [0.0],
            }
        ),
        path,
    )
    return m5_panel(path=path, **over)


def test_validate_accepts_the_real_m5_panel() -> None:
    _require_m5()
    p = m5_panel().validate()
    assert p.has_exposure
    assert p.season_length == 7


def test_panel_record_round_trip_restores_json_types_and_old_defaults() -> None:
    panel = Panel(
        "/panel",
        "id",
        "date",
        "qty",
        known_future=("planned_price",),
        static=("store",),
        static_reals=("weight",),
        plan=PlanSource(
            "/plan", "valid_date", "known_at", snapshot_col="snapshot",
            price_col="planned_price",
        ),
    )
    record = json.loads(json.dumps(dataclasses.asdict(panel)))
    restored = Panel.from_record(record)
    assert restored == panel
    assert isinstance(restored.known_future, tuple)
    assert isinstance(restored.static, tuple)
    assert isinstance(restored.static_reals, tuple)
    assert isinstance(restored.plan, PlanSource)

    historical = dict(record)
    historical.pop("static_reals")
    historical.pop("frequency")
    old = Panel.from_record(historical)
    assert old.static_reals == () and old.frequency == "D"


def test_panel_record_refuses_ambiguous_container_shapes() -> None:
    base = dataclasses.asdict(Panel("/panel", "id", "date", "qty"))
    cases = (
        ([], "one JSON object"),
        ({}, "malformed fields"),
        ({**base, "known_future": "price"}, "JSON array"),
        ({**base, "plan": "not-an-object"}, "object or null"),
        ({**base, "unknown_field": "silent drift"}, "unknown field"),
        ({**base, "plan": {"path": "/plan"}}, "malformed PlanSource"),
        ({**base, "plan": {"path": "/plan", "valid_date_col": "date",
                           "known_at_col": "known_at"}}, "malformed PlanSource"),
    )
    for record, message in cases:
        with pytest.raises(PanelError, match=message):
            Panel.from_record(record)  # type: ignore[arg-type]


def test_validate_refuses_a_non_plan_source_before_adapter_execution() -> None:
    malformed = _structural_panel(plan={"path": "/plan"})  # type: ignore[arg-type]
    with pytest.raises(PanelError, match="PlanSource"):
        malformed.validate()


def test_validate_reads_the_real_parquet_schema_not_the_dataclass() -> None:
    # `qty_typo` is a plausible name that is NOT in M5. If validate trusted the strings it was
    # handed, this would pass and every adapter downstream would fail later and less legibly.
    _require_m5()
    try:
        m5_panel(target_col="qty_typo").validate()
    except PanelError as e:
        assert "absent from the Parquet schema" in str(e)
        assert "qty" in str(e)  # the message lists what IS there
        return
    raise AssertionError("accepted a column absent from the real M5 schema")


def test_validate_refuses_a_missing_path() -> None:
    try:
        m5_panel(path="/nonexistent/panel").validate()
    except PanelError as e:
        assert "does not exist" in str(e); return
    raise AssertionError("accepted a path that does not exist")


@pytest.mark.parametrize("role,value", [
    ("series_col", "meter.id"), ("time_col", "day date"), ("target_col", "energy`kwh"),
    ("exposure_col", "state.value"), ("oos_col", "stock.out"),
    ("known_future", ("future.value",)), ("past_only", ("past.value",)),
    ("static", ("store.name",)), ("static_reals", ("store.area",)),
    ("price_col", "price.value"), ("promo_col", "promo.value"),
    ("factor_group", "group.id"), ("product_col", "product.id"),
    ("store_col", "store.id"), ("category_col", "category.id"),
    ("status_prob_cols", {"exact": "exact.prob", "lower_bound": "pl", "unobserved": "pu"}),
])
def test_mapped_column_identifiers_fail_before_parquet_read(tmp_path, role, value):
    path = tmp_path / "invalid.parquet"
    path.touch()  # Opening this as Parquet would fail before the intended diagnostic.
    defaults = dict(path=path, series_col="id", time_col="date", target_col="qty")
    panel = Panel(**{**defaults, role: value})
    with pytest.raises(PanelError, match="ASCII identifiers"):
        panel.validate()


def test_internal_identifiers_and_semantic_aliases_remain_valid():
    panel = _structural_panel(known_future=("sell_price",), price_col="sell_price")
    panel.validate()
    dataclasses.replace(panel, series_col="__af_series_id").validate_column_names()


def test_plan_identifiers_are_validated_at_construction():
    with pytest.raises(PanelError, match="ASCII identifiers"):
        PlanSource("/plan", "valid.date", "known_at", price_col="price")


@pytest.mark.parametrize("extra", ["ID", "unmapped.column"])
def test_all_parquet_columns_follow_the_identifier_contract(tmp_path, extra):
    path = tmp_path / "panel.parquet"
    pq.write_table(pa.table({"id": ["one"], "date": [dt.date(2024, 1, 1)],
                             "qty": [1.0], extra: ["unmapped"]}), path)
    with pytest.raises(PanelError, match="rename headers"):
        Panel(path, "id", "date", "qty").validate()


def test_panel_roles_refuse_case_ambiguous_aliases_before_io():
    with pytest.raises(PanelError, match="distinct ignoring case"):
        Panel("unused", "id", "date", "qty", static=("ID",)).validate_column_names()


def test_parquet_refuses_duplicate_physical_names(tmp_path):
    path = tmp_path / "panel.parquet"
    table = pa.table({"id": ["one"], "date": [dt.date(2024, 1, 1)], "qty": [1.0]})
    table = table.append_column("id", pa.array(["other"]))
    pq.write_table(table, path)
    with pytest.raises(PanelError, match="rename duplicate headers"):
        Panel(path, "id", "date", "qty").validate()


def test_validate_refuses_one_column_in_two_roles() -> None:
    try:
        _structural_panel(exposure_col="qty").validate()
    except PanelError as e:
        assert "more than one role" in str(e) and "qty" in str(e); return
    raise AssertionError("accepted qty as both target and exposure")


def test_validate_refuses_the_target_declared_known_future() -> None:
    try:
        _structural_panel(known_future=("qty",)).validate()
    except PanelError as e:
        assert "leaking" in str(e); return
    raise AssertionError("accepted the target as a known-future covariate -- that is the leak")


def test_validate_refuses_a_non_positive_season_length() -> None:
    for bad in (0, -7):
        try:
            _structural_panel(season_length=bad).validate()
        except PanelError as e:
            assert "season_length" in str(e); continue
        raise AssertionError(f"accepted season_length={bad}")


def test_validate_refuses_an_unsupported_panel_frequency() -> None:
    for frequency in ("0D", "1h", "YE-FEB", True):
        with pytest.raises(PanelError, match="frequency|supported"):
            _structural_panel(frequency=frequency).validate()


def test_panel_default_season_length_follows_the_cadence_family() -> None:
    assert {
        frequency: Panel("/not-read", "id", "date", "qty", frequency=frequency).season_length
        for frequency in ("H", "D", "W", "M", "ME", "MS", "Y", "YE", "YS", "2D")
    } == {
        "H": 24, "D": 7, "W": 52, "M": 12, "ME": 12, "MS": 12,
        "Y": 1, "YE": 1, "YS": 1, "2D": 7,
    }
    assert Panel(
        "/not-read", "id", "date", "qty", frequency="W", season_length=7,
    ).season_length == 7


def test_hourly_panel_requires_a_timezone_naive_timestamp_column(tmp_path) -> None:
    with pytest.raises(PanelError, match="time_col.*timestamp"):
        _structural_panel(frequency="H").validate()

    path = tmp_path / "hourly.parquet"
    pq.write_table(pa.table({
        "id": ["a"],
        "date": pa.array([dt.datetime(2024, 1, 1, 3)], type=pa.timestamp("us")),
        "qty": [1.0],
    }), path)
    Panel(path, "id", "date", "qty", frequency="H").validate()

    aware = tmp_path / "aware.parquet"
    pq.write_table(pa.table({
        "id": ["a"],
        "date": pa.array(
            [dt.datetime(2024, 1, 1, 3, tzinfo=dt.UTC)],
            type=pa.timestamp("us", tz="UTC"),
        ),
        "qty": [1.0],
    }), aware)
    with pytest.raises(PanelError, match="timezone-naive"):
        Panel(aware, "id", "date", "qty", frequency="H").validate()


def test_has_exposure_is_false_when_no_column_is_declared() -> None:
    p = _structural_panel(exposure_col=None).validate()
    assert not p.has_exposure
    assert "observation_state" not in p.role_columns


def test_unconstrained_demand_requires_censoring_evidence() -> None:
    try:
        _structural_panel(
            exposure_col=None, target_estimand="unconstrained_demand"
        ).validate()
    except PanelError as exc:
        assert "requires an observation contract" in str(exc)
    else:
        raise AssertionError("declared latent demand with no censoring evidence")
    # Soft probabilities ARE censoring evidence: the same estimand with status_prob_cols builds.
    soft = _structural_panel(exposure_col=None, target_estimand="unconstrained_demand")
    soft = dataclasses.replace(
        soft, status_prob_cols={"exact": "pe", "lower_bound": "pl", "unobserved": "pu"})
    assert soft.validate() is soft


def test_generic_observed_target_preserves_panel_and_forecast_identity() -> None:
    panel = _structural_panel(target_estimand="observed_nonnegative").validate()
    assert Panel.from_record(dataclasses.asdict(panel)).target_estimand == "observed_nonnegative"
    forecast = Forecast(
        "last_value", "/nonexistent/forecast", "2024-01-01", 1, 1, {"emitted": 1},
        target_estimand=panel.target_estimand,
    ).validate()
    assert Forecast.from_record(forecast.as_record()).target_estimand == "observed_nonnegative"


def test_unknown_target_estimand_is_refused() -> None:
    try:
        _structural_panel(target_estimand="revenue").validate()
    except PanelError as exc:
        assert "unsupported target_estimand" in str(exc)
        return
    raise AssertionError("accepted an undefined target quantity")


def test_role_columns_is_ordered_and_includes_every_declared_role() -> None:
    p = m5_panel(known_future=("wm_yr_wk",), static=("dept_id", "cat_id"))
    assert p.role_columns == ("id", "date", "qty", "observation_state",
                              "wm_yr_wk", "dept_id", "cat_id")


def test_semantic_roles_may_deliberately_alias_feature_roles() -> None:
    p = _structural_panel(
        known_future=("sell_price",),
        price_col="sell_price",
        static=("item_id", "cat_id"),
        product_col="item_id",
        category_col="cat_id",
        factor_group="dept_id",
        store_col="store_id",
    ).validate()
    assert p.role_columns.count("sell_price") == 1
    assert p.role_columns.count("item_id") == 1
    assert p.role_columns[-2:] == ("dept_id", "store_id")


def test_every_optional_semantic_role_is_checked_against_the_schema() -> None:
    for role in ("oos_col", "price_col", "promo_col", "factor_group", "product_col", "store_col",
                 "category_col"):
        try:
            _structural_panel(**{role: "not_a_real_column"}).validate()
        except PanelError as e:
            assert "not_a_real_column" in str(e), (role, e)
            continue
        raise AssertionError(f"accepted an absent column in {role}")


def test_invalid_or_ambiguous_column_role_names_are_refused() -> None:
    invalid = [
        {"price_col": ""},
        {"static": (None,)},
        {"known_future": ("sell_price", "")},
        {"price_col": "sell_price", "promo_col": "sell_price"},
        {"product_col": "id"},
    ]
    for override in invalid:
        try:
            _structural_panel(**override).validate()
        except PanelError:
            continue
        raise AssertionError(f"accepted invalid/ambiguous role declaration {override!r}")


def test_static_real_and_soft_state_roles_cannot_bypass_feature_role_validation() -> None:
    invalid = (
        {"static_reals": ["pe"]},
        {"static_reals": ("pe", "pe")},
        {"static": ("pe",), "static_reals": ("pe",)},
        {"exposure_col": None, "status_prob_cols": ["exact", "lower_bound", "unobserved"]},
        {
            "exposure_col": None,
            "status_prob_cols": {"exact": "", "lower_bound": "pl", "unobserved": "pu"},
        },
        {
            "exposure_col": None,
            "status_prob_cols": {"exact": 1, "lower_bound": "pl", "unobserved": "pu"},
        },
        {
            "exposure_col": None,
            "status_prob_cols": {"exact": "qty", "lower_bound": "pl", "unobserved": "pu"},
        },
        {
            "exposure_col": None,
            "status_prob_cols": {"exact": "pe", "lower_bound": "pe", "unobserved": "pu"},
        },
    )
    for override in invalid:
        with pytest.raises(PanelError):
            _structural_panel(**override).validate()


@pytest.mark.parametrize(
    "role,column,replacement",
    [
        ("static_reals", "pe", pa.array(["one"])),
        ("price_col", "sell_price", pa.array([True])),
        ("promo_col", "pu", pa.array(["yes"])),
    ],
)
def test_numeric_semantic_roles_refuse_incompatible_arrow_types(
    tmp_path, role, column, replacement
) -> None:
    panel = _structural_panel(**{role: (column,) if role == "static_reals" else column})
    table = pq.read_table(panel.path)
    changed = table.set_column(table.column_names.index(column), column, replacement)
    path = tmp_path / f"bad-{role}.parquet"
    pq.write_table(changed, path)
    with pytest.raises(PanelError, match=rf"{role}.*incompatible Arrow type"):
        dataclasses.replace(panel, path=str(path)).validate()


def test_path_is_normalised_so_a_path_and_a_str_caller_hash_alike() -> None:
    assert Panel(pathlib.Path(M5), "id", "date", "qty") == Panel(str(M5), "id", "date", "qty")


def test_validate_accepts_a_hive_partition_column() -> None:
    """`store_id` lives in M5's DIRECTORY NAMES, not in any fragment's schema. Spark reads it
    fine, so a panel that declares it must validate -- and every WRMSSE level above the item uses
    it, so rejecting it would make the metric unusable on its own benchmark."""
    _require_m5()
    p = m5_panel(path=M5, static=("store_id",)).validate()
    assert "store_id" in p.role_columns


def test_a_column_in_neither_the_schema_nor_the_partitions_is_still_refused() -> None:
    _require_m5()
    try:
        m5_panel(path=M5, static=("region_id",)).validate()
    except PanelError as e:
        assert "region_id" in str(e); return
    raise AssertionError("partition-column support let an unknown column through")


def test_any_fragment_finds_a_file_inside_a_partitioned_directory() -> None:
    _require_m5()
    frag = _any_fragment(M5)  # M5 is written partitioned by store_id
    assert frag.suffix == ".parquet" and frag.is_file()


def test_any_fragment_refuses_a_directory_with_no_parquet() -> None:
    # SYNTHETIC (structure): M5 always contains fragments, so it cannot produce this case.
    with tempfile.TemporaryDirectory() as d:
        try:
            _any_fragment(pathlib.Path(d))
        except PanelError as e:
            assert "no .parquet fragment" in str(e); return
    raise AssertionError("accepted a directory containing no parquet")


def test_validate_reports_a_missing_column_against_a_hand_built_schema() -> None:
    # SYNTHETIC (structure): a Parquet whose schema is WRONG. M5's is right.
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "part.parquet"
        pq.write_table(pa.table({"id": ["a"], "date": ["2016-01-01"]}), f)
        try:
            Panel(f, "id", "date", "qty").validate()
        except PanelError as e:
            assert "qty" in str(e); return
    raise AssertionError("accepted a panel whose target column is not in the file")


def test_eligible_states_keeps_censored_cells_and_drops_unobserved() -> None:
    assert ELIGIBLE_STATES == {"exact", "lower_bound"}
    assert "unobserved" not in ELIGIBLE_STATES


def test_forecast_validate_accepts_a_complete_account() -> None:
    f = Forecast("naive", "/tmp/f", "2016-04-24", 28, 3,
                 {"emitted": 70, "declined_no_source": 14}).validate()
    assert f.n_emitted == 70 and f.n_declined == 14


def test_forecast_validate_refuses_an_incomplete_account() -> None:
    # 3 x 28 = 84 requested; 80 accounted for. A silent shortfall is a good score on what is left.
    try:
        Forecast("naive", "/tmp/f", "2016-04-24", 28, 3, {"emitted": 80}).validate()
    except ForecastError as e:
        assert "80 cells but 84 were requested" in str(e); return
    raise AssertionError("accepted a forecast that lost four cells")


def test_forecast_validate_refuses_a_negative_bucket() -> None:
    try:
        Forecast("naive", "/tmp/f", "2016-04-24", 2, 1, {"emitted": 4, "declined": -2}).validate()
    except ForecastError as e:
        assert "negative" in str(e); return
    raise AssertionError("accepted a negative disposition count")


def test_forecast_validate_refuses_a_non_positive_shape() -> None:
    for horizon, n_series in ((0, 1), (1, 0)):
        try:
            Forecast("naive", "/tmp/f", "2016-04-24", horizon, n_series, {}).validate()
        except ForecastError as e:
            assert ">= 1" in str(e); continue
        raise AssertionError(f"accepted horizon={horizon}, n_series={n_series}")


def test_forecast_validate_refuses_invalid_metadata_before_artifact_exists() -> None:
    """A sidecar/remote Forecast is still a contract; absence of Parquet cannot defer validation."""
    bad = (
        Forecast("naive", "/tmp/not-created", "not-a-date", 1, 1, {"emitted": 1}),
        Forecast("naive", "/tmp/not-created", "2016-04-24", True, 1, {"emitted": 1}),
        Forecast("naive", "/tmp/not-created", "2016-04-24", 1, 1, {"emitted": True}),
        Forecast("naive", "/tmp/not-created", "2016-04-24", 1, 1, {"emitted": 1.0}),
    )
    for candidate in bad:
        try:
            candidate.validate()
        except ForecastError:
            continue
        raise AssertionError(f"accepted malformed Forecast metadata: {candidate!r}")


def test_panel_and_forecast_refuse_blank_paths() -> None:
    """Path('') means the current directory and can accidentally validate an unrelated artifact."""
    for make in (
        lambda: Panel("", "id", "date", "qty"),
        lambda: Forecast("naive", "", "2016-04-24", 1, 1, {"emitted": 1}),
    ):
        try:
            make()
        except (PanelError, ForecastError) as exc:
            assert "path" in str(exc) and "blank" in str(exc)
            continue
        raise AssertionError("accepted a blank artifact path")


def test_forecast_validate_refuses_an_unknown_estimand() -> None:
    try:
        Forecast("naive", "/tmp/f", "2016-04-24", 1, 1, {"emitted": 1},
                 target_estimand="revenue").validate()
    except ForecastError as exc:
        assert "unsupported target_estimand" in str(exc)
        return
    raise AssertionError("accepted a forecast of an undefined quantity")


def test_forecast_validate_refuses_unsupported_or_unrepresentable_cadence() -> None:
    """Even an unmaterialised sidecar must describe a representable forecast calendar."""
    for frequency in ("Q", "100000000000000000000D"):
        with pytest.raises(ForecastError, match="frequency|calendar"):
            Forecast(
                "naive",
                "/tmp/not-created",
                "2016-04-24",
                2,
                1,
                {"emitted": 2},
                frequency=frequency,
            ).validate()


@pytest.mark.parametrize(
    "origin,frequency",
    [
        ("2024-03-01T22:00:00", "H"),
        ("2024-01-03", "W"),
        ("2024-01-31", "M"),
        ("2024-01-01", "MS"),
        ("2023-12-31", "Y"),
        ("2024-01-01", "YS"),
    ],
)
def test_forecast_metadata_accepts_every_supported_cadence(origin, frequency) -> None:
    Forecast(
        "model", "/tmp/not-created", origin, 2, 1, {"emitted": 2}, frequency=frequency,
    ).validate()


def test_n_emitted_counts_every_emitted_prefixed_bucket() -> None:
    f = Forecast("naive", "/tmp/f", "2016-04-24", 4, 1,
                 {"emitted": 1, "emitted_from_lower_bound": 2, "declined_no_source": 1})
    f.validate()
    assert f.n_emitted == 3 and f.n_declined == 1


def test_from_record_round_trips_a_forecast() -> None:
    import json
    original = Forecast("naive", "/tmp/f", "2016-04-24", 4, 1,
                        {"emitted": 3, "declined_no_source": 1}).validate()
    rebuilt = Forecast.from_record(json.loads(json.dumps(original.as_record()))).validate()
    assert rebuilt == original


def test_from_record_recomputes_the_totals_rather_than_trusting_them() -> None:
    """A resumed run rebuilds a Forecast from a file on disk. If it trusted the file's totals, an
    edited record would resume as a forecast that never accounted for its cells."""
    rec = Forecast("naive", "/tmp/f", "2016-04-24", 4, 1, {"emitted": 4}).validate().as_record()
    rec["n_emitted"] = 999            # a lie, in the derived field
    rebuilt = Forecast.from_record(rec).validate()
    assert rebuilt.n_emitted == 4, "trusted the record's derived total instead of recomputing"


def test_from_record_still_refuses_an_unaccounted_forecast() -> None:
    rec = Forecast("naive", "/tmp/f", "2016-04-24", 4, 1, {"emitted": 4}).validate().as_record()
    rec["disposition"] = {"emitted": 2}
    try:
        Forecast.from_record(rec).validate()
    except ForecastError as e:
        assert "2 cells but 4 were requested" in str(e); return
    raise AssertionError("resumed a forecast that lost two cells")


def test_as_record_is_json_safe_and_states_the_derived_totals() -> None:
    import json
    rec = Forecast("naive", "/tmp/f", "2016-04-24", 2, 1, {"emitted": 2}).validate().as_record()
    assert json.loads(json.dumps(rec))["n_emitted"] == 2
    assert rec["n_declined"] == 0


def _write_forecast(
    root: pathlib.Path,
    rows: list[dict[str, object]],
    *,
    time_type: pa.DataType | None = None,
) -> pathlib.Path:
    """Write a Spark-shaped directory (not a single file) for artifact-boundary tests."""
    root.mkdir()
    schema = pa.schema([
        ("series_id", pa.string()),
        ("date", time_type or pa.date32()),
        ("yhat", pa.float64()),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), root / "part-00000.parquet")
    return root


def test_hourly_forecast_keeps_two_hours_on_one_day_as_distinct_keys() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = _write_forecast(
            pathlib.Path(d) / "forecast",
            [
                {"series_id": "a", "date": dt.datetime(2024, 3, 1, 23), "yhat": 1.0},
                {"series_id": "a", "date": dt.datetime(2024, 3, 2, 0), "yhat": 2.0},
            ],
            time_type=pa.timestamp("us"),
        )
        Forecast(
            "model", root, "2024-03-01T22:00:00", 2, 1, {"emitted": 2}, frequency="H",
        ).validate()


def test_hourly_forecast_refuses_date_physical_type_and_off_hour_keys() -> None:
    with tempfile.TemporaryDirectory() as d:
        date_root = _write_forecast(pathlib.Path(d) / "date", [
            {"series_id": "a", "date": dt.date(2024, 3, 2), "yhat": 1.0},
        ])
        with pytest.raises(ForecastError, match="Arrow timestamp"):
            Forecast(
                "model", date_root, "2024-03-01T22:00:00", 1, 1, {"emitted": 1},
                frequency="H",
            ).validate()

        minute_root = _write_forecast(
            pathlib.Path(d) / "minute",
            [{"series_id": "a", "date": dt.datetime(2024, 3, 1, 23, 30), "yhat": 1.0}],
            time_type=pa.timestamp("us"),
        )
        with pytest.raises(ForecastError, match="exact hour"):
            Forecast(
                "model", minute_root, "2024-03-01T22:00:00", 1, 1, {"emitted": 1},
                frequency="H",
            ).validate()


def test_calendar_forecast_uses_anchored_keys_and_rejects_lossy_timestamps() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = _write_forecast(pathlib.Path(d) / "monthly", [
            {"series_id": "a", "date": dt.date(2024, 2, 29), "yhat": 1.0},
            {"series_id": "a", "date": dt.date(2024, 3, 31), "yhat": 2.0},
        ])
        Forecast(
            "model", root, "2024-01-31", 2, 1, {"emitted": 2}, frequency="M",
        ).validate()

        nonmidnight = _write_forecast(
            pathlib.Path(d) / "nonmidnight",
            [{"series_id": "a", "date": dt.datetime(2024, 2, 29, 1), "yhat": 1.0}],
            time_type=pa.timestamp("us"),
        )
        with pytest.raises(ForecastError, match="cannot discard"):
            Forecast(
                "model", nonmidnight, "2024-01-31", 1, 1, {"emitted": 1}, frequency="M",
            ).validate()


def test_forecast_validate_inspects_a_valid_existing_partition_directory() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = _write_forecast(pathlib.Path(d) / "forecast", [
            {"series_id": "a", "date": dt.date(2016, 4, 25), "yhat": 0.0},
            {"series_id": "a", "date": dt.date(2016, 4, 26), "yhat": 2.5},
        ])
        got = Forecast("model", root, "2016-04-24", 2, 1, {"emitted": 2}).validate()
        assert got.n_emitted == 2


def test_forecast_validate_allows_declined_cells_but_no_dates_outside_the_window() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = _write_forecast(pathlib.Path(d) / "forecast", [
            {"series_id": "a", "date": dt.date(2016, 4, 26), "yhat": 1.0},
        ])
        Forecast("model", root, "2016-04-24", 2, 1,
                 {"emitted": 1, "declined_no_source": 1}).validate()
        bad = Forecast("model", root, "2016-04-26", 1, 1, {"emitted": 1})
        try:
            bad.validate()
        except ForecastError as e:
            assert "outside the requested window" in str(e)
            return
    raise AssertionError("accepted a forecast date outside origin + 1 .. origin + horizon")


def test_forecast_validate_refuses_duplicate_keys_and_manifest_row_disagreement() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = _write_forecast(pathlib.Path(d) / "duplicates", [
            {"series_id": "a", "date": dt.date(2016, 4, 25), "yhat": 1.0},
            {"series_id": "a", "date": dt.date(2016, 4, 25), "yhat": 2.0},
        ])
        try:
            Forecast("model", root, "2016-04-24", 1, 2, {"emitted": 2}).validate()
        except ForecastError as e:
            assert "duplicate forecast key" in str(e)
        else:
            raise AssertionError("accepted duplicate (series_id, date) keys")

        one = _write_forecast(pathlib.Path(d) / "one", [
            {"series_id": "a", "date": dt.date(2016, 4, 25), "yhat": 1.0},
        ])
        try:
            Forecast("model", one, "2016-04-24", 1, 2, {"emitted": 2}).validate()
        except ForecastError as e:
            assert "claims 2 emitted cells" in str(e)
        else:
            raise AssertionError("accepted a Parquet row count that disagrees with n_emitted")


def test_forecast_validate_refuses_nonfinite_or_negative_predictions() -> None:
    for value in (float("nan"), float("inf"), -0.01):
        with tempfile.TemporaryDirectory() as d:
            root = _write_forecast(pathlib.Path(d) / "forecast", [
                {"series_id": "a", "date": dt.date(2016, 4, 25), "yhat": value},
            ])
            try:
                Forecast("model", root, "2016-04-24", 1, 1,
                         {"emitted": 1}).validate()
            except ForecastError as e:
                assert "finite and non-negative" in str(e)
                continue
            raise AssertionError(f"accepted invalid yhat={value!r}")


def test_forecast_validate_refuses_an_existing_artifact_with_the_wrong_schema() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d) / "forecast"
        root.mkdir()
        pq.write_table(pa.table({"series_id": ["a"], "date": [dt.date(2016, 4, 25)]}),
                       root / "part.parquet")
        try:
            Forecast("model", root, "2016-04-24", 1, 1, {"emitted": 1}).validate()
        except ForecastError as e:
            assert "missing required columns" in str(e) and "yhat" in str(e)
            return
    raise AssertionError("accepted a forecast artifact without yhat")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
