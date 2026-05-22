"""Compatibility import for saved models; use ``forecast.models.mft.decoder``."""

import importlib
import sys

_module = importlib.import_module("forecast.models.mft.decoder")
sys.modules[__name__] = _module
