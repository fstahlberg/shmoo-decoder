import copy
from typing import Any, Dict, Sequence
import numpy as np
from scipy.special import softmax

from shmoo.core import utils


class Hypothesis:

    def __init__(self, states: Sequence[Dict[str, Any]], score: float,
                 output_features: Dict[str, Any]):
        self.states = states
        self.score = score
        self.output_features = output_features
        self.stale = False

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

    def __str__(self):
        return "token_id:%d score:%f parent:%s" % (
            self.token_id, self.score, str(self.parent_hypothesis))


class Predictor:

    @classmethod
    def setup_predictor(cls, config):
        return cls(config)

    def __init__(self, config):
        pass

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
        return cls(config)

    def __init__(self, config):
        pass

    def process(self, features: Dict[str, Any]) -> None:
        pass


class Preprocessor(Processor):
    pass


class Postprocessor(Processor):
    pass


class Decoder:

    @classmethod
    def setup_decoder(cls, config):
        return cls(config)

    def __init__(self, config):
        try:
            self.eos_id = config["eos_id"]
        except KeyError:
            self.eos_id = utils.DEFAULT_EOS_ID
        self._predictors = []

    def is_finished(self, hypo: Hypothesis) -> bool:
        try:
            return hypo.output_features["output_ids"][-1] == self.eos_id
        except IndexError:
            return False

    def best_hypo_finished(self, hypos: Sequence[Hypothesis]) -> bool:
        if hypos:
            return self.is_finished(hypos[0])
        return True

    def all_hypos_finished(self, hypos: Sequence[Hypothesis]) -> bool:
        return all(self.is_finished(hypo) for hypo in hypos)

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
        if prediction.token_id is None:
            return prediction.parent_hypothesis

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
            if self.is_finished(hypo):
                hypo.output_features.update(input_features)
                hypo.output_features["score"] = hypo.score
                all_output_features.append(hypo.output_features)
        return all_output_features

    def get_position_scores(
            self,
            hypos: Sequence[Hypothesis]) -> np.ndarray:
        accumulated_scores = 0.0
        for index, predictor in enumerate(self._predictors):
            predictor_states = [hypo.states[index] for hypo in hypos]
            accumulated_scores += predictor.predict_next(predictor_states)
        return accumulated_scores

    def complete_finished_hypothesis(
            self,
            hypos: Sequence[Hypothesis],
            predictions: Sequence[Prediction]) -> Sequence[Prediction]:
        for hypo in hypos:
            if self.is_finished(hypo):
                predictions.append(
                    Prediction(
                        token_id=None, score=hypo.score,
                        parent_hypothesis=hypo))
        return predictions

    def get_predictions(
            self,
            hypos: Sequence[Hypothesis],
            nbest: int) -> Sequence[Prediction]:
        unfinished_hypos = [hypo for hypo in hypos if not self.is_finished(hypo)]
        accumulated_scores = self.get_position_scores(hypos=unfinished_hypos)
        base_scores = [hypo.score for hypo in unfinished_hypos]
        accumulated_scores += np.expand_dims(base_scores, 1)
        flat_indices = np.argpartition(-accumulated_scores, nbest, axis=None)
        indices = np.unravel_index(
            flat_indices[:nbest], accumulated_scores.shape)
        predictions = []
        for hypo_index, token_id in zip(indices[0], indices[1]):
            predictions.append(
                Prediction(
                    token_id=token_id,
                    score=accumulated_scores[hypo_index, token_id],
                    parent_hypothesis=unfinished_hypos[hypo_index]))
        predictions = self.complete_finished_hypothesis(hypos=hypos, predictions=predictions)
        predictions.sort(key=lambda p: -p.score)
        return predictions[:nbest]

    def sample_predictions(
            self,
            hypos: Sequence[Hypothesis],
            seed: int) -> Sequence[Prediction]:

        unfinished_hypos = [hypo for hypo in hypos if not self.is_finished(hypo)]
        accumulated_scores = self.get_position_scores(hypos=unfinished_hypos)
        base_scores = [hypo.score for hypo in unfinished_hypos]

        num_unfinished_hypos, vocab_size = accumulated_scores.shape
        predictions = []
        # Not compatible with batches yet
        for sample_id in range(num_unfinished_hypos):
            np.random.seed(seed + sample_id)
            # Sample token
            sampled_token_id = np.random.choice(
                a=[i for i in range(vocab_size)],
                size=1,
                p=softmax(accumulated_scores[sample_id,:]),
            )
            predictions.append(
                Prediction(
                    token_id=int(sampled_token_id),
                    score=accumulated_scores[sample_id, int(sampled_token_id)] + base_scores[sample_id],
                    parent_hypothesis=unfinished_hypos[sample_id])
            )
        predictions = self.complete_finished_hypothesis(hypos=hypos, predictions=predictions)
        return predictions

    def add_predictor(self, predictor: Predictor) -> None:
        self._predictors.append(predictor)

    def decode(
            self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        pass
