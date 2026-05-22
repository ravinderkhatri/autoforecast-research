"""Compatibility import for saved models; use ``autoforecast.adapters.mft``."""

import importlib
import sys

_module = importlib.import_module("autoforecast.adapters.mft")
_module.V3RevisionFeatures = _module.HMFNetRevisionFeatures
_module.V5RevisionFeatures = _module.PACTRevisionFeatures
sys.modules[__name__] = _module
