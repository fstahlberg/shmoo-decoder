from typing import Any, Dict


class Predictor:
    pass


class Processor:
    def process(self, features: Dict[str, Any]) -> None:
        pass


class Preprocessor(Processor):
    pass


class Postprocessor(Processor):
    pass


class Decoder(Processor):

    def __init__(self):
        self._predictors = []

    def add_predictor(self, predictor: Predictor):
        self._predictors.append(predictor)
