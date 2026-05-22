"""Compatibility import for saved models; use ``forecast.models.mft.pact``."""

import importlib
import sys

_module = importlib.import_module("forecast.models.mft.pact")
_module.V5ModelInputs = _module.PACTModelInputs
_module.V5Labels = _module.PACTLabels
_module.V5Batch = _module.PACTBatch
_module.V5DecoderBlock = _module.PACTDecoderBlock
_module.V5Prediction = _module.PACTPrediction
_module.V5Model = _module.PACTModel
sys.modules[__name__] = _module
