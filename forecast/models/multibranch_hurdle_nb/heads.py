"""Compatibility import for saved models; use ``forecast.models.mft.heads``."""

import importlib
import sys

_module = importlib.import_module("forecast.models.mft.heads")
sys.modules[__name__] = _module
