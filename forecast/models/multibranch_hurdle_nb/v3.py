"""Compatibility import for saved models; use ``forecast.models.mft.hmfnet``."""

import importlib
import sys

_module = importlib.import_module("forecast.models.mft.hmfnet")
_module.V3Prediction = _module.HMFNetPrediction
_module.V3Model = _module.HMFNetModel
sys.modules[__name__] = _module
