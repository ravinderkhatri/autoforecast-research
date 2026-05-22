"""Compatibility import for saved models; use ``forecast.models.mft.sft``."""

import importlib
import sys

_module = importlib.import_module("forecast.models.mft.sft")
_module.V2Model = _module.SFTModel
sys.modules[__name__] = _module
