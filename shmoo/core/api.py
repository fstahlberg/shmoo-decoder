import copy

from typing import Any, Dict, Optional

from shmoo.core import registry


class Shmoo:

    def __init__(self):
        self._preprocessors = None
        self._postprocessors = None
        self._decoder = None
        self._predictors = None

    def set_up(self, config) -> None:
        self._preprocessors = [registry.make_preprocessor("", "")]
        self._postprocessors = [registry.make_postprocessor("", "")]
        self._decoder = registry.make_decoder("", "")
        self._decoder.add_predictor(registry.make_predictor("", ""))

    def decode_raw(self, raw: Any) -> Dict[str, Any]:
        return self.decode_features({"input_raw": raw})

    def decode_features(self, input_features: Dict[str, Any]) -> Dict[str, Any]:
        input_features = copy.deepcopy(input_features)
        for preprocessor in self._preprocessors:
            preprocessor.process(input_features)
        all_output_features = self._decoder.process(input_features)
        for output_features in all_output_features:
            for postprocessor in self._postprocessors:
                postprocessor.process(output_features)
        return all_output_features
