"""Compatibility import for saved models; use ``forecast.models.mft.trunk``."""

import importlib
import sys

_module = importlib.import_module("forecast.models.mft.trunk")
sys.modules[__name__] = _module
