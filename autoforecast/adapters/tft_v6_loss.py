"""Compatibility import for saved models; use ``autoforecast.adapters.tft_q_loss``."""

import importlib
import sys

_module = importlib.import_module("autoforecast.adapters.tft_q_loss")
_module.V3HurdleLoss = _module.HMFNetHurdleLoss
sys.modules[__name__] = _module
