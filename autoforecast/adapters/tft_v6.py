"""Compatibility import for saved models; use ``autoforecast.adapters.tft_q``."""

import importlib
import sys

_module = importlib.import_module("autoforecast.adapters.tft_q")
_module._publish_v6_artifacts = _module._publish_tft_artifacts
_module._v6_frame = _module._tft_frame
sys.modules[__name__] = _module
