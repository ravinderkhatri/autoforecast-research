"""Compatibility import for saved models; use ``forecast.models.mft.factor``."""

import importlib
import sys

_module = importlib.import_module("forecast.models.mft.factor")
sys.modules[__name__] = _module
