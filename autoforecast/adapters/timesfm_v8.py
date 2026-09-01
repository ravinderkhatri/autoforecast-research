"""V8 uses the shared count workflow with a trainable TimesFM3 LoRA adapter."""

from autoforecast.adapters import timesfm_hnb

NAME = "timesfm_v8"
SCHEMA = "autoforecast.timesfm-v8-inference.v1"


def fit_predict(panel, origin, horizon, **kwargs):
    return timesfm_hnb.fit_predict(panel, origin, horizon, _adapter=NAME, **kwargs)


def predict_from_checkpoint(panel, origin, horizon, **kwargs):
    return timesfm_hnb.predict_from_checkpoint(panel, origin, horizon, _adapter=NAME, **kwargs)
