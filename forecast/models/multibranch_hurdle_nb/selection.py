"""Compatibility import for saved models; use ``forecast.models.mft.selection``."""

import importlib
import sys

_module = importlib.import_module("forecast.models.mft.selection")
sys.modules[__name__] = _module
