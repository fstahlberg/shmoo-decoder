import copy
from typing import Any, Dict, Sequence
import numpy as np


class Hypothesis:

    def __init__(self, states: Sequence[Dict[str, Any]], score: float,
                 output_features: Dict[str, Any]):
        self.states = states
        self.score = score
        self.output_features = output_features
        self.stale = False

    def is_final(self) -> bool:
        try:
            return self.output_features["output_ids"][-1] == 2
        except IndexError:
            return False

    def __str__(self):
        s = "score:%f feat:%s" % (self.score, self.output_features)
        if self.stale:
            s += " (STALE)"
        return s


class Prediction:
    def __init__(self, token_id: int, score: float,
                 parent_hypothesis: Hypothesis):
        self.token_id = token_id
        self.score = score
        self.parent_hypothesis = parent_hypothesis


class Predictor:

    @classmethod
    def setup_predictor(cls, config):
        return cls()

    def initialize_state(
            self, input_features: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def update_single_state(self, state: Dict[str, Any],
                            prediction: Prediction,
                            lazy: bool = False) -> Dict[str, Any]:
        return state

    def predict_next(self, states: Sequence[Dict[str, Any]]):
        all_scores = []
        for state in states:
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

    def make_initial_hypothesis(
            self, input_features: Dict[str, Any]) -> Hypothesis:
        states = [predictor.initialize_state(input_features)
                  for predictor in self._predictors]
        return Hypothesis(
            states=states, score=0.0, output_features={
                "output_ids": []
            })

    def make_hypothesis(self, prediction: Prediction,
                        lazy: bool = False) -> Hypothesis:
        new_states = [predictor.update_single_state(
            state=old_state, prediction=prediction, lazy=lazy) for
            predictor, old_state in zip(
                self._predictors, prediction.parent_hypothesis.states)]
        if lazy:
            prediction.parent_hypothesis.stale = True
            output_features = prediction.parent_hypothesis.output_features
        else:
            output_features = copy.deepcopy(
                prediction.parent_hypothesis.output_features)
        output_features["output_ids"].append(prediction.token_id)
        return Hypothesis(
            states=new_states, score=prediction.score,
            output_features=output_features)

    def make_final_output_features(
            self,
            input_features: Dict[str, Any],
            hypos: Sequence[Hypothesis]) -> Sequence[Dict[str, Any]]:
        all_output_features = []
        for hypo in sorted(hypos, key=lambda h: -h.score):
            hypo.output_features.update(input_features)
            hypo.output_features["score"] = hypo.score
            all_output_features.append(hypo.output_features)
        return all_output_features

    def get_predictions(
            self,
            hypos: Sequence[Hypothesis],
            nbest: int) -> Sequence[Prediction]:
        accumulated_scores = 0.0
        for index, predictor in enumerate(self._predictors):
            predictor_states = [hypo.states[index] for hypo in hypos]
            accumulated_scores += predictor.predict_next(predictor_states)
        accumulated_scores += [hypo.score for hypo in hypos]
        flat_indices = np.argpartition(-accumulated_scores, nbest, axis=None)
        indices = np.unravel_index(
            flat_indices[:nbest], accumulated_scores.shape)
        predictions = []
        for hypo_index, token_id in zip(indices[0], indices[1]):
            predictions.append(
                Prediction(
                    token_id=token_id,
                    score=accumulated_scores[hypo_index, token_id],
                    parent_hypothesis=hypos[hypo_index]))
        return predictions

    def add_predictor(self, predictor: Predictor) -> None:
        self._predictors.append(predictor)

    def decode(
            self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        pass
