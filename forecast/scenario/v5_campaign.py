"""Compatibility import for saved models; use ``forecast.scenario.pact_campaign``."""

import importlib
import sys

_module = importlib.import_module("forecast.scenario.pact_campaign")
_module.evaluate_v5_campaign_pair = _module.evaluate_pact_campaign_pair
sys.modules[__name__] = _module
