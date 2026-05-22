"""Compatibility import for saved models; use ``autoforecast.pact_features``."""

import importlib
import sys

_module = importlib.import_module("autoforecast.pact_features")
_module.V5Config = _module.PACTConfig
sys.modules[__name__] = _module
