"""Compatibility import for saved models; use ``autoforecast.hmfnet_features``."""

import importlib
import sys

_module = importlib.import_module("autoforecast.hmfnet_features")
_module.V3Config = _module.HMFNetConfig
sys.modules[__name__] = _module
