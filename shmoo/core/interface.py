from typing import Any, Dict, Sequence

import numpy as np


class Prediction:
    pass


class Predictor:

    def initialize_state(self, input_features: Dict[str, Any]) -> Dict[
        str, Any]:
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
    def process(self, features: Dict[str, Any]) -> None:
        pass


class Preprocessor(Processor):
    pass


class Postprocessor(Processor):
    pass


class Hypothesis:

    def __init__(self, states: Sequence[Dict[str, Any]], score: float,
                 output_features: Dict[str, Any]):
        self.states = states
        self.score = score
        self.output_features = output_features

    def is_final(self) -> bool:
        return False

class Decoder:

    def __init__(self):
        self._predictors = []

    def make_initial_hypothesis(
            self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        states = [predictor.initialize_state(input_features)
                  for predictor in self._predictors]
        return Hypothesis(states=states, score=0.0, output_features={
            "output_ids": []
        })

    def get_predictions(self, hypos: Sequence[Hypothesis], nbest: int) -> Sequence[Prediction]:
        all_predictor_scores = []
        for index, predictor in enumerate(self._predictors):
            predictor_states = [hypo.states[index] for hypo in hypos]
            all_predictor_scores.append(predictor.predict_next(predictor_states))
        


    def add_predictor(self, predictor: Predictor) -> None:
        self._predictors.append(predictor)

    def decode(
            self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        pass
