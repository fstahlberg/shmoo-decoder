from typing import Any, Dict, Sequence, Tuple, Union

from shmoo import prepostprocessing, decoders, predictors


def _split_spec(spec: str) -> Tuple[str, str]:
    try:
        name, config = spec.split(":", 1)
    except ValueError:
        name = spec
        config = ""
    return name, config

class Shmoo:

    def __init__(self):
        self._preprocessors = []
        self._postprocessors = []
        self._decoder = None

    def set_up(self, config: Dict[str, Any]) -> None:

        # Assumption: There can only be one decoder
        self._decoder = decoders.setup_decoder(list(config["decoder"].keys())[0], config)

        self._decoder.add_predictor(
                predictors.setup_predictor(config["framework"], config)
        )

        for preprocessor in config["preprocessors"]:
            # preprocessor is a str if no parameters are specified
            self._preprocessors.append(
                prepostprocessing.setup_processor(preprocessor if type(preprocessor) == str else list(preprocessor.keys())[0],
                                                  config)
            )

        for postprocessor in config["postprocessors"]:
            # postprocessor is a str if no parameters are specified
            self._postprocessors.append(
                prepostprocessing.setup_processor(postprocessor if type(postprocessor) == str else list(postprocessor.keys())[0],
                                                  config)
            )

    def decode_raw(self, raw: Any) -> Sequence[Dict[str, Any]]:
        return self.decode_features({"input_raw": raw})

    def decode_features(self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        for preprocessor in self._preprocessors:
            preprocessor.process(input_features)
        all_output_features = self._decoder.process(input_features)
        for output_features in all_output_features:
            for postprocessor in self._postprocessors:
                postprocessor.process(output_features)
        return all_output_features
