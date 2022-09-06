from typing import Any, Dict, Sequence

import numpy as np


class Prediction:
    def __init__(self, token_id):
        self.token_id = token_id


class Predictor:

    @classmethod
    def setup_predictor(cls, config):
        return cls()

    def initialize_state(self, input_features: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def update_states(self, states: Sequence[Dict[str, Any]],
                      predictions: Sequence[Prediction]) -> None:
        for state, prediction in zip(states, predictions):
            self.update_single_state(state, prediction)

    def update_single_state(self, state: Dict[str, Any],
                            prediction: Prediction) -> None:
        pass

    def predict_next(self, states: Sequence[Dict[str, Any]]):
        all_scores = []
        for state in enumerate(states):
            all_scores.append(self.predict_next_single(state))
        return np.stack(all_scores)

    def predict_next_single(self, state: Dict[str, Any]):
        pass


class Processor:

    @classmethod
    def setup_processor(cls, config):
        return cls()

    def process(self, features: Dict[str, Any]) -> None:
        pass


class Preprocessor(Processor):
    pass


class Postprocessor(Processor):
    pass


class Decoder:

    @classmethod
    def setup_decoder(cls, config):
        return cls()

    def __init__(self):
        self._predictors = []

    def add_predictor(self, predictor: Predictor):
        self._predictors.append(predictor)

    def decode(self, input_features: Dict[str, Any]) -> Sequence[
        Dict[str, Any]]:
        pass
