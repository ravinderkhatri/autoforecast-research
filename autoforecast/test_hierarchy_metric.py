"""Independent arithmetic and exact-full-group gradient tests for the cached metric."""

from typing import Any

import numpy as np
import pytest
import torch

from autoforecast.hierarchy_metric import HierarchyMetric, read_forecast_grid
from autoforecast.rolling_training import staged_hierarchy_backward


def example_metric(**kwargs):
    """Hand-checkable arrays, used only for arithmetic (not accuracy evidence)."""
    history = np.array([[0, 1, 3, 2], [1, 2, 2, 4]], dtype=float)
    return HierarchyMetric.build({"id": ("a", "b")}, history, np.ones_like(history),
                                  ((), ("id",)), weight_days=2, **kwargs)


def test_hierarchy_metric_hand_calculation():
    metric = example_metric()
    # Total [1,3,5,6]: scale=(4+4+1)/3. Bottom scales (4+1)/2 and (1+0+4)/3.
    np.testing.assert_allclose(metric.scales[0], [3.0])
    np.testing.assert_allclose(metric.scales[1], [2.5, 5 / 3])
    np.testing.assert_allclose(metric.weights[1], [5 / 11, 6 / 11])
    truth = np.array([[1, 2], [2, 3]], float)
    prediction = truth + np.array([[1], [2]])
    expected = (np.sqrt(3) + (5 / 11) / np.sqrt(2.5) + (6 / 11) * 2 / np.sqrt(5 / 3)) / 2
    assert metric.score(prediction, truth)["wrmsse"] == pytest.approx(expected, abs=1e-12)
    assert metric.loss(torch.tensor(prediction), torch.tensor(truth), epsilon=1e-12).item() \
        == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("fault", ["scale", "weights", "nonfinite", "metadata", "levels",
                                  "names", "duplicate_level", "shape", "window"])
def test_hierarchy_metric_build_refusals(fault):
    history = np.array([[0, 1, 3, 2], [1, 2, 2, 4]], dtype=float)
    price = np.ones_like(history)
    metadata: dict[str, tuple[str, ...]] = {"id": ("a", "b")}
    levels, weight_days = ((), ("id",)), 2
    if fault == "scale":
        history[0] = 0
    elif fault == "weights":
        price[:] = np.nan
    elif fault == "nonfinite":
        history[0, 0] = np.nan
    elif fault == "metadata":
        metadata["id"] = ("a",)
    elif fault == "levels":
        levels = (("missing",),)
    elif fault == "names":
        levels = ((), ("total",))
    elif fault == "duplicate_level":
        levels = ((), ())
    elif fault == "shape":
        price = price[:, :3]
    else:
        weight_days = True
    message = {"scale": "undefined RMSSE", "weights": "positive sales require",
               "nonfinite": "metric history", "metadata": "metadata must cover",
               "levels": "absent identity", "names": "unique and unambiguous",
               "duplicate_level": "unique and unambiguous", "shape": "invalid history/price",
               "window": "invalid history/price"}[fault]
    with pytest.raises(ValueError, match=message):
        HierarchyMetric.build(metadata, history, price, levels, weight_days=weight_days)


def test_metric_refuses_zero_dollars_and_bad_differentiable_inputs():
    with pytest.raises(ValueError, match="undefined dollar weights"):
        HierarchyMetric.build({"id": ("a",)}, np.array([[1., 2., 1.]]), np.zeros((1, 3)),
                              (("id",),), weight_days=2)
    metric = example_metric()
    for prediction, target, epsilon in ((torch.ones(1, 2), torch.ones(1, 2), 1e-8),
                                        (torch.ones(2, 2), torch.ones(2, 2), 0.),
                                        (-torch.ones(2, 2), torch.ones(2, 2), 1e-8),
                                        (torch.ones(2, 2), torch.full((2, 2), float("nan")), 1e-8)):
        with pytest.raises(ValueError):
            metric.loss(prediction, target, epsilon=epsilon)


@pytest.mark.parametrize("weighting", ["revenue", "uniform"])
@pytest.mark.parametrize("weight_days", [0, -1])
def test_hierarchy_metric_refuses_nonpositive_weight_windows(weighting, weight_days):
    history = np.array([[1., 2., 1.]])
    with pytest.raises(ValueError, match="invalid history/price shape, levels or weight window"):
        HierarchyMetric.build({"id": ("a",)}, history, np.ones_like(history), ((),),
                              weighting=weighting, weight_days=weight_days)


def test_hierarchy_metric_refuses_unknown_weighting():
    with pytest.raises(ValueError, match="weighting must be revenue or uniform"):
        example_metric(weighting="unknown")


def test_grid_refuses_invalid_population(tmp_path):
    with pytest.raises(ValueError, match="unique nonempty"):
        read_forecast_grid(str(tmp_path), ("a", "a"), "2020-01-01", 1,
                           series_col="id", time_col="date", value_col="value")


@pytest.mark.parametrize("bad_id", ["", " ", None, 1])
def test_grid_refuses_non_string_or_empty_series_identifiers(tmp_path, bad_id):
    """The keyed-grid boundary validates every population member before Arrow lookup."""
    import datetime as dt

    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "invalid-series.parquet"
    id_type = pa.string() if bad_id is None else None
    pq.write_table(
        pa.table(
            {
                "id": pa.array([bad_id], type=id_type),
                "date": [dt.date(2020, 1, 2)],
                "value": [1.0],
            }
        ),
        path,
    )
    with pytest.raises(ValueError, match="unique nonempty"):
        read_forecast_grid(
            str(path),
            (bad_id,),
            "2020-01-01",
            1,
            series_col="id",
            time_col="date",
            value_col="value",
        )


def test_training_surrogate_cannot_publish_score():
    metric = example_metric(training_surrogate=True)
    with pytest.raises(ValueError, match="surrogate"):
        metric.score(np.ones((2, 2)), np.ones((2, 2)))


def test_finite_inputs_cannot_publish_overflowed_metric_values():
    metadata = {"id": ("a", "b")}
    with pytest.raises(ValueError, match="dollar weights"):
        HierarchyMetric.build(metadata, np.array([[2., 1.], [2., 1.]]),
                              np.full((2, 2), 1e308), ((), ("id",)), weight_days=1)
    with pytest.raises(ValueError, match="aggregate history"):
        HierarchyMetric.build(metadata, np.full((2, 2), 1e308), np.full((2, 2), 1e-308),
                              ((),), weight_days=1, training_surrogate=True)
    with pytest.raises(ValueError, match="scale differences"):
        HierarchyMetric.build({"id": ("a",)}, np.array([[1., 1e200, 1.]]),
                              np.ones((1, 3)), ((),), weight_days=1, training_surrogate=True)
    with pytest.raises(ValueError, match="RMSSE scales"):
        HierarchyMetric.build({"id": ("a",)}, np.array([[1., 1e154, 1.]]),
                              np.ones((1, 3)), ((),), weight_days=1, training_surrogate=True)
    metric = example_metric()
    for value in (1e200, 1e308):
        with pytest.raises(ValueError, match="numerical overflow"):
            metric.score(np.full((2, 2), value), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="hierarchy objective"):
        metric.loss(torch.full((2, 2), 1e200, dtype=torch.double),
                     torch.zeros(2, 2, dtype=torch.double), epsilon=1e-8)
    wide_scale = HierarchyMetric.build({"id": ("a",)}, np.array([[1., 1e20, 1.]]),
                                       np.ones((1, 3)), ((),), weight_days=1)
    with pytest.raises(ValueError, match="constants overflow"):
        wide_scale.loss(torch.ones(1, 2), torch.zeros(1, 2), epsilon=1e-8)


@pytest.mark.parametrize("prediction", [np.ones((1, 2)), np.full((2, 2), np.nan),
                                       -np.ones((2, 2)), np.ones((2, 0))])
def test_hierarchy_score_refuses_bad_grid(prediction):
    with pytest.raises(ValueError, match="grid"):
        example_metric().score(prediction, np.ones_like(prediction))


def test_staged_gradient_matches_direct():
    torch.manual_seed(91)
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(0.5),
                                 torch.nn.Softplus()).double()
    x = torch.randn(2, 3, dtype=torch.double)
    target = torch.ones(2, 2, dtype=torch.double)
    metric = example_metric()
    model.eval()
    loss = 0.2 * metric.loss(model(x), target, epsilon=1e-8)
    loss.backward()
    expected = []
    for parameter in model.parameters():
        assert parameter.grad is not None
        expected.append(parameter.grad.clone())
    model.zero_grad(set_to_none=True)
    model.train()
    value = staged_hierarchy_backward(model, (np.array([1]), np.array([0])),
                                      lambda rows: model(x[rows]), metric, target,
                                      weight=0.2, epsilon=1e-8)
    assert value == pytest.approx(loss.item(), abs=1e-12)
    for parameter, wanted in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, wanted, atol=1e-12, rtol=1e-12)
    assert model.training and model[1].training


def test_staged_mixed_gradient_uses_float32_hierarchy_surrogate():
    model = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Softplus()).half()
    metric = example_metric()
    target = torch.ones(2, 2)
    seen = []

    class Float32Metric:
        n_bottom = metric.n_bottom

        def loss(self, prediction, actual, *, epsilon):
            seen.append((prediction.dtype, actual.dtype))
            assert prediction.dtype == actual.dtype == torch.float32
            return metric.loss(prediction, actual, epsilon=epsilon)

    value = staged_hierarchy_backward(
        model,
        (np.array([1]), np.array([0])),
        lambda rows: model(torch.ones(len(rows), 1, dtype=torch.float16)),
        Float32Metric(),
        target,
        weight=0.2,
        epsilon=1e-8,
        backward_scale=lambda loss: loss * 2,
    )
    assert np.isfinite(value)
    assert seen == [(torch.float32, torch.float32)]
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_staged_gradient_refuses_missing_rows_and_stochastic_forward():
    model = torch.nn.Linear(1, 2)
    metric = example_metric()
    with pytest.raises(ValueError, match="exactly once"):
        staged_hierarchy_backward(model, (np.array([0]),), lambda r: model(torch.ones(len(r), 1)),
                                  metric, torch.ones(2, 2), weight=1, epsilon=1e-8)
    with pytest.raises(ValueError, match="passes differ"):
        staged_hierarchy_backward(model, (np.arange(2),),
                                  lambda r: model(torch.ones(len(r), 1)).exp() + torch.rand(2, 2),
                                  metric, torch.ones(2, 2), weight=1, epsilon=1e-8)
    assert model.training


@pytest.mark.parametrize("fault", [None, "duplicate", "outside", "series", "value", "missing"])
def test_read_forecast_grid(tmp_path, fault):
    import datetime as dt

    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [dict(id="b", date=dt.date(2020, 1, 2), value=2.0),
            dict(id="a", date=dt.date(2020, 1, 2), value=1.0)]
    if fault == "duplicate":
        rows[1] = dict(rows[0])
    elif fault == "outside":
        rows[0]["date"] = dt.date(2020, 1, 3)
    elif fault == "series":
        rows[0]["id"] = "c"
    elif fault == "value":
        rows[0]["value"] = np.nan
    elif fault == "missing":
        rows.pop()
    path = tmp_path / "forecast.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    arguments: dict[str, Any] = dict(path=str(path), series_ids=("a", "b"), origin="2020-01-01", horizon=1,
                     series_col="id", time_col="date", value_col="value")
    if fault:
        with pytest.raises(ValueError):
            read_forecast_grid(**arguments)
    else:
        np.testing.assert_array_equal(read_forecast_grid(**arguments), [[1], [2]])
        rows.append(dict(id="a", date=dt.date(2019, 12, 31), value=4.0))
        pq.write_table(pa.Table.from_pylist(rows), path)
        np.testing.assert_array_equal(read_forecast_grid(**arguments, filter_actuals=True), [[1], [2]])


def test_read_forecast_grid_refuses_boolean_values_before_numeric_coercion(tmp_path):
    """Boolean actuals are a corrupt count schema, not the numeric values zero and one."""
    import datetime as dt

    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "boolean-values.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["a", "b"],
                "date": [dt.date(2020, 1, 2), dt.date(2020, 1, 2)],
                "value": pa.array([False, True], type=pa.bool_()),
            }
        ),
        path,
    )
    with pytest.raises(ValueError, match="boolean"):
        read_forecast_grid(
            str(path),
            ("a", "b"),
            "2020-01-01",
            1,
            series_col="id",
            time_col="date",
            value_col="value",
            filter_actuals=True,
        )


def test_read_forecast_grid_refuses_noncanonical_origin_identity(tmp_path):
    import datetime as dt

    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "forecast.parquet"
    pq.write_table(
        pa.Table.from_pylist([dict(id="a", date=dt.date(2020, 1, 2), value=1.0)]),
        path,
    )
    with pytest.raises(ValueError, match="canonical"):
        read_forecast_grid(
            str(path),
            ("a",),
            "20200101",
            1,
            series_col="id",
            time_col="date",
            value_col="value",
        )
