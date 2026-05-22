"""Compatibility import for saved models; use ``forecast.models.mft.sample``."""

import importlib
import sys

_module = importlib.import_module("forecast.models.mft.sample")
sys.modules[__name__] = _module
