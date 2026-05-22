"""Code-owned adapter routes; YAML selects IDs and cannot import executable objects."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterDescriptor:
    module: str
    callable: str
    runtime: str
    history: str
    parameters: str = "none"
    tracking: tuple[str, str] = ("none", "none")
    frequency: str = "D"
    target_estimands: tuple[str, ...] = ("sales_given_availability",)
    operations: tuple[str, ...] = ("fit_predict",)
    consumes_exposure: bool = True
    handles_right_censoring: bool = False
    requires_complete_window: bool = False


DESCRIPTORS = {
    "seasonal_naive": AdapterDescriptor(
        "naive", "seasonal_naive", "builtin", "season",
        target_estimands=("sales_given_availability", "observed_nonnegative"),
    ),
    "last_value": AdapterDescriptor(
        "last_value", "fit_predict", "builtin", "one", requires_complete_window=True,
        target_estimands=("sales_given_availability", "observed_nonnegative"),
    ),
    "croston": AdapterDescriptor("classical", "forecast_with", "classical", "two"),
    "auto_ets": AdapterDescriptor("classical", "forecast_with", "classical", "auto_ets"),
    "auto_theta": AdapterDescriptor("classical", "forecast_with", "classical", "auto_ets"),
    "shallow_global": AdapterDescriptor("shallow", "fit_predict", "shallow", "one", "typed"),
    "tft": AdapterDescriptor("tft", "fit_predict", "tft", "tft", "typed", ("autolog", "lightning")),
    "tft_q": AdapterDescriptor(
        "tft_q", "fit_predict_quantile", "tft", "tft_q", "typed", ("autolog", "lightning")
    ),
    "tft_hnb": AdapterDescriptor(
        "tft_q", "fit_predict_hnb", "tft", "tft_q", "typed", ("autolog", "lightning")
    ),
    "autogluon_zoo": AdapterDescriptor(
        "zoo", "fit_predict", "autogluon_zoo", "one", consumes_exposure=False
    ),
    "timesfm_standalone": AdapterDescriptor("foundation", "fit_predict", "timesfm", "timesfm"),
    "mft": AdapterDescriptor(
        "mft",
        "fit_predict",
        "custom_torch",
        "mft",
        "typed",
        ("reporter", "none"),
        target_estimands=("sales_given_availability", "unconstrained_demand"),
        handles_right_censoring=True,
    ),
    "sft": AdapterDescriptor(
        "mft",
        "fit_predict_revision_adapter",
        "custom_torch",
        "mft",
        "typed",
        ("reporter", "none"),
        target_estimands=("sales_given_availability", "unconstrained_demand"),
        handles_right_censoring=True,
    ),
    "hmfnet": AdapterDescriptor(
        "mft",
        "fit_predict_revision_adapter",
        "custom_torch",
        "mft",
        "typed",
        ("reporter", "none"),
        target_estimands=("sales_given_availability", "unconstrained_demand"),
        handles_right_censoring=True,
    ),
    "pact": AdapterDescriptor(
        "mft",
        "fit_predict_revision_adapter",
        "custom_torch",
        "mft",
        "typed",
        ("reporter", "none"),
    ),
}


def descriptor(name: str) -> AdapterDescriptor:
    try:
        result = DESCRIPTORS[name]
    except KeyError:
        raise ValueError(f"no maintained adapter descriptor for {name!r}") from None
    if (
        not result.module.isidentifier()
        or not result.callable.isidentifier()
        or result.parameters not in {"none", "typed"}
        or result.frequency != "D"
        or result.operations != ("fit_predict",)
    ):
        raise ValueError(f"incomplete or unsupported adapter descriptor: {name}")
    return result
