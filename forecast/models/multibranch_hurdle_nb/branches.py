"""Compatibility import for saved models; use ``forecast.models.mft.branches``."""

import importlib
import sys

_module = importlib.import_module("forecast.models.mft.branches")
sys.modules[__name__] = _module
