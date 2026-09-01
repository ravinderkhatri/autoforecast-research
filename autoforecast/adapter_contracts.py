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
    supported_frequency_families: tuple[str, ...] = ("D",)
    supports_strided_daily: bool = False
    supports_shared_studies: bool = False


DESCRIPTORS = {
    "seasonal_naive": AdapterDescriptor(
        "naive", "seasonal_naive", "builtin", "season",
        target_estimands=("sales_given_availability", "observed_nonnegative"),
        supported_frequency_families=("H", "D", "W", "M", "Y"),
        supports_strided_daily=True,
        supports_shared_studies=True,
    ),
    "last_value": AdapterDescriptor(
        "last_value", "fit_predict", "builtin", "one", requires_complete_window=True,
        target_estimands=("sales_given_availability", "observed_nonnegative"),
        supported_frequency_families=("H", "D", "W", "M", "Y"),
        supports_strided_daily=True,
        supports_shared_studies=True,
    ),
    "croston": AdapterDescriptor(
        "classical", "forecast_with", "classical", "two", supports_shared_studies=True,
    ),
    "auto_ets": AdapterDescriptor(
        "classical", "forecast_with", "classical", "auto_ets", supports_shared_studies=True,
    ),
    "auto_theta": AdapterDescriptor(
        "classical", "forecast_with", "classical", "auto_ets", supports_shared_studies=True,
    ),
    "shallow_global": AdapterDescriptor(
        "shallow", "fit_predict", "shallow", "one", "typed", supports_shared_studies=True,
    ),
    "tft": AdapterDescriptor(
        "tft", "fit_predict", "tft", "tft", "typed", ("autolog", "lightning"),
        supports_shared_studies=True,
    ),
    "tft_q": AdapterDescriptor(
        "tft_q", "fit_predict_quantile", "tft", "tft_q", "typed", ("autolog", "lightning"),
        supports_shared_studies=True,
    ),
    "tft_hnb": AdapterDescriptor(
        "tft_q", "fit_predict_hnb", "tft", "tft_q", "typed", ("autolog", "lightning"),
        supports_shared_studies=True,
    ),
    "timesfm_hnb": AdapterDescriptor(
        "timesfm_hnb", "fit_predict", "timesfm3", "one", "typed", ("reporter", "none"),
        supports_shared_studies=True,
    ),
    "timesfm_v8": AdapterDescriptor(
        "timesfm_v8", "fit_predict", "timesfm3", "one", "typed", ("reporter", "none"),
        supports_shared_studies=True,
    ),
    "autogluon_zoo": AdapterDescriptor(
        "zoo", "fit_predict", "autogluon_zoo", "one", consumes_exposure=False,
        supports_shared_studies=True,
    ),
    "timesfm_standalone": AdapterDescriptor(
        "foundation", "fit_predict", "timesfm", "timesfm", supports_shared_studies=True,
    ),
    "mft": AdapterDescriptor(
        "mft",
        "fit_predict",
        "custom_torch",
        "mft",
        "typed",
        ("reporter", "none"),
        target_estimands=("sales_given_availability", "unconstrained_demand"),
        handles_right_censoring=True,
        supports_shared_studies=True,
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
        supports_shared_studies=True,
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
        supports_shared_studies=True,
    ),
    "pact": AdapterDescriptor(
        "mft",
        "fit_predict_revision_adapter",
        "custom_torch",
        "mft",
        "typed",
        ("reporter", "none"),
        supports_shared_studies=True,
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
        or not result.supported_frequency_families
        or len(set(result.supported_frequency_families))
        != len(result.supported_frequency_families)
        or not set(result.supported_frequency_families) <= {"H", "D", "W", "M", "Y"}
        or type(result.supports_strided_daily) is not bool
        or type(result.supports_shared_studies) is not bool
        or result.operations != ("fit_predict",)
    ):
        raise ValueError(f"incomplete or unsupported adapter descriptor: {name}")
    return result
