"""Compatibility import for saved models; use ``forecast.models.mft.fusion``."""

import importlib
import sys

_module = importlib.import_module("forecast.models.mft.fusion")
sys.modules[__name__] = _module
