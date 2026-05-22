"""Compatibility import for saved models; use ``forecast.models.mft.regularisation``."""

import importlib
import sys

_module = importlib.import_module("forecast.models.mft.regularisation")
sys.modules[__name__] = _module
