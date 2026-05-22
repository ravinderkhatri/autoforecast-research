"""Compatibility import for saved models; use ``forecast.models.mft.seasonality``."""

import importlib
import sys

_module = importlib.import_module("forecast.models.mft.seasonality")
sys.modules[__name__] = _module
