"""Main interfaces and base classes.

This module contains the main building blocks of the Shmoo framework. Details
about how these classes work together can be found here:
    https://github.com/fstahlberg/shmoo-decoder#software-design
"""

from collections import OrderedDict
from typing import Any, Dict, Sequence, Callable
import numpy as np

from shmoo.core import utils


class Hypothesis:
    """A (partial) hypothesis."""

    def __init__(self, states: Sequence[Dict[str, Any]], score: float,
                 output_ids: ...):
        self.states = states
        self.score = score
        self.output_ids = output_ids
        self.stale = False

    def __str__(self):
        s = "score:%f feat:%s" % (self.score, self.output_ids)
        if self.stale:
            s += " (STALE)"
        return s


class Prediction:
    """An expansion of a partial hypothesis by a single token."""

    def __init__(self, token_id: int, score: float,
                 parent_hypothesis: Hypothesis):
        self.token_id = token_id
        self.score = score
        self.parent_hypothesis = parent_hypothesis

    def __str__(self):
        return "token_id:%d score:%f parent:%s" % (
            self.token_id, self.score, str(self.parent_hypothesis))


class Predictor:
    """A scoring module that defines token-level scores based on its state.

    Predictors are left-to-right scoring modules that assign scores to each
    entry in the vocabulary given the internal predictor state. The predictor
    state is a Python dictionary that is specific to a hypothesis, and may
    store information such as the target side translation prefix for partial
    hypotheses.
    """

    @classmethod
    def setup_predictor(cls, config):
        """Factory function for predictors."""
        return cls(config)

    def __init__(self, config):
        pass

    def initialize_state(
            self, input_features: Dict[str, Any]) -> Dict[str, Any]:
        """Builds the initial predictor state.

        Args:
            input_features: Input feature dictionary.

        Returns:
            The initial predictor state `initial_state`. The token-level scores
            at time step 0 are given by `predict_next(initial_state)`.
        """
        pass

    def update_single_state(self, state: Dict[str, Any],
                            prediction: Prediction,
                            lazy: bool = False) -> Dict[str, Any]:
        """Updates a predictor state given a Prediction.

        This method is called when the Decoder expands a hypothesis. Predictors
        may implement this function in two different modes. If lazy is False,
        the `state` input directory is left unmodified, and a new state is
        returned. If lazy is true, `state` may be modified in-place and then
        returned. Decoders may use `lazy=true` if the parent hypothesis is not
        used with any other `Prediction`s to expand to other hypotheses.

        Args:
            state: Current predictor state
            prediction: A single prediction for the next timestep.
            lazy: Whether `state` may be modified in-place.

        Returns:
            The updated predictor state.
        """
        return state

    def predict_next(self, states: Sequence[Dict[str, Any]],
                     scores: ...) -> ...:
        """Predicts a batch of scores for the next time step.

        Args:
            states: `batch_size` predictor states.
            scores: A `[batch_size, vocab_size]` tensor containing the scores
                passed through from the previous predictor.

        Returns:
            A `[batch_size, vocab_size]` tensor of scores.
        """
        all_scores = []
        for state_index, state in enumerate(states):
            all_scores.append(
                self.predict_next_single(state, scores[state_index]))
        return np.stack(all_scores)

    def predict_next_single(self, state: Dict[str, Any], scores: ...) -> ...:
        """Predicts the next time step scores given a single predictor state.

        Args:
            state: A single predictor state.
            scores: A `[vocab_size,]` tensor containing the scores passed
                through from the previous predictor.

        Returns:
            A `[vocab_size,]` tensor of scores.
        """
        raise NotImplementedError(
            "Neither predict_next() nor predict_next_single() are implemented.")

    def postprocess_output_features(self, state: Dict[str, Any],
                                    output_features: Dict[str, Any]) -> None:
        """Postprocesses the final output features.

        Args:
            state: Final predictor state
            output_features: Output feature dict postprocessed by previous
                predictors.
        """
        pass


class Processor:
    """Base class for preprocessors and postprocessors."""

    @classmethod
    def setup_processor(cls, config):
        """Factory function for processors."""
        return cls(config)

    def __init__(self, config):
        pass

    def process(self, features: Dict[str, Any]) -> None:
        """Processes a single feature dictionary.

        Args:
            features: Feature dictionary passed through from the previous
                processor. Features are modified in-place in the dictionary.
        """
        pass


class Preprocessor(Processor):
    """Base class for Preprocessors.

    Preprocessors modify the feature dictionary prior to running the Decoder.
    Subclasses must implement process().
    """
    pass


class Postprocessor(Processor):
    """Base class for Postprocessors.

    Postprocessors modify the feature dictionaries returned by the Decoder.
    Subclasses must implement either process_all() (processing all
    returned feature dictionaries at once) or process() (processing each
    feature dictionary separately).
    """

    def process_all(
            self, all_features: Sequence[Dict[str, Any]]) -> None:
        """Processes all feature returned by the Decoder."""
        for features in all_features:
            self.process(features)


class Decoder:
    """A decoding strategy."""

    @classmethod
    def setup_decoder(cls, config):
        """Factory function for decoders."""
        return cls(config)

    def __init__(self, config):
        try:
            self.eos_id = config["eos_id"]
        except KeyError:
            self.eos_id = utils.DEFAULT_EOS_ID
        self._predictors = []

    def is_finished(self, hypo: Hypothesis) -> bool:
        """Returns true if the last output ID is equal to end-of-sentence."""
        try:
            return hypo.output_ids[-1] == self.eos_id
        except IndexError:
            return False

    def best_hypo_finished(self, hypos: Sequence[Hypothesis]) -> bool:
        """Returns true if the first hypothesis in `hypos` is finished."""
        if hypos:
            return self.is_finished(hypos[0])
        return True

    def all_hypos_finished(self, hypos: Sequence[Hypothesis]) -> bool:
        """Returns true if all hypotheses in `hypos` are finished."""
        return all(self.is_finished(hypo) for hypo in hypos)

    def make_initial_hypothesis(
            self, input_features: Dict[str, Any]) -> Hypothesis:
        """Builds the initial `Hypothesis`.

        Iterates through all predictors and calls initialize_state() to build
        the initial predictor states.

        Args:
            input_features: Input feature dictionary.

        Returns:
            The initial `Hypothesis` for timestep 0.
        """
        states = [predictor.initialize_state(input_features)
                  for predictor in self._predictors]
        return Hypothesis(
            states=states, score=0.0, output_ids=[])

    def make_hypothesis(self, prediction: Prediction,
                        lazy: bool = False) -> Hypothesis:
        if prediction.token_id is None:
            return prediction.parent_hypothesis

        if prediction.parent_hypothesis.stale:
            raise ValueError("Trying to expand stale hypothesis!")

        new_states = [predictor.update_single_state(
            state=old_state, prediction=prediction, lazy=lazy) for
            predictor, old_state in zip(
                self._predictors, prediction.parent_hypothesis.states)]
        if lazy:
            prediction.parent_hypothesis.stale = True
            output_ids = prediction.parent_hypothesis.output_ids
            output_ids.append(prediction.token_id)
        else:
            output_ids = prediction.parent_hypothesis.output_ids + [
                prediction.token_id]
        return Hypothesis(
            states=new_states, score=prediction.score,
            output_ids=output_ids)

    def make_final_output_features(
            self,
            input_features: Dict[str, Any],
            hypos: Sequence[Hypothesis]) -> Sequence[Dict[str, Any]]:
        all_output_features = []
        for hypo in sorted(hypos, key=lambda h: -h.score):
            if self.is_finished(hypo):
                output_features = {
                    "score": hypo.score,
                    "output": OrderedDict({"ids": hypo.output_ids})
                }
                output_features.update(input_features)
                for state, predictor in zip(hypo.states, self._predictors):
                    predictor.postprocess_output_features(
                        state, output_features)
                all_output_features.append(output_features)
        all_output_features.sort(key=lambda f: -f["score"])
        return all_output_features

    def get_position_scores(
            self,
            hypos: Sequence[Hypothesis]) -> np.ndarray:
        pos_scores = np.zeros((len(hypos),))
        for index, predictor in enumerate(self._predictors):
            predictor_states = [hypo.states[index] for hypo in hypos]
            pos_scores = predictor.predict_next(predictor_states, pos_scores)
        return pos_scores

    def complete_finished_hypothesis(
            self,
            hypos: Sequence[Hypothesis],
            predictions: Sequence[Prediction]) -> Sequence[Prediction]:
        for hypo in hypos:
            if self.is_finished(hypo):
                predictions.append(
                    Prediction(
                        token_id=None, score=hypo.score,
                        parent_hypothesis=hypo
                    )
                )
        return predictions

    def modify_scores(self, scores: np.ndarray) -> np.ndarray:
        "Method to be overwritten in specific decoders"
        return scores

    def get_predictions(
            self,
            hypos: Sequence[Hypothesis],
            nbest: int) -> Sequence[Prediction]:
        unfinished_hypos = [hypo for hypo in hypos if
                            not self.is_finished(hypo)]
        pos_scores = self.get_position_scores(hypos=unfinished_hypos)
        pos_scores = self.modify_scores(scores=pos_scores)
        base_scores = [hypo.score for hypo in unfinished_hypos]
        accumulated_scores = pos_scores + np.expand_dims(base_scores, 1)
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
        predictions = self.complete_finished_hypothesis(
            hypos=hypos, predictions=predictions)
        predictions.sort(key=lambda p: -p.score)
        return predictions[:nbest]

    def sample_predictions(
            self,
            hypos: Sequence[Hypothesis],
            seed: int,
            make_probs: Callable) -> Sequence[Prediction]:

        unfinished_hypos = [hypo for hypo in hypos if
                            not self.is_finished(hypo)]
        pos_scores = self.get_position_scores(hypos=unfinished_hypos)
        base_scores = [hypo.score for hypo in unfinished_hypos]

        num_unfinished_hypos, vocab_size = pos_scores.shape
        predictions = []
        # Not compatible with batches yet
        for sample_id in range(num_unfinished_hypos):
            np.random.seed(seed + sample_id)
            # Sample token
            sampled_token_id = np.random.choice(
                a=[i for i in range(vocab_size)],
                size=1,
                p=make_probs(pos_scores[sample_id, :]),
                # pos_scores[sample_id,:] needs to be 1-dimensional array
            )
            predictions.append(
                Prediction(
                    token_id=int(sampled_token_id),
                    score=pos_scores[sample_id, int(sampled_token_id)] +
                          base_scores[sample_id],
                    parent_hypothesis=unfinished_hypos[sample_id])
            )
        predictions = self.complete_finished_hypothesis(
            hypos=hypos, predictions=predictions)
        return predictions

    def add_predictor(self, predictor: Predictor) -> None:
        """Adds a predictor to the scoring pipeline."""
        self._predictors.append(predictor)

    def decode(
            self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        """Runs decoding on a single sentence.

        Subclasses must implement this method. This is the main entry point to
        the Decoder. The decoding task is defined by the decoder config, the
        predictors, and the `input_features`. This method must return a list of
        output feature dictionaries, each corresponding to a complete
        (translation) hypotheses.

        Args:
            input_features: Input feature dictionary

        Returns:
            A sorted list of translation hypotheses, represented by output
            feature dictionaries.
        """
        raise NotImplementedError("Decoding strategy is not implemented.")


class Metric:
    """Base class for metrics."""

    @classmethod
    def setup_metric(cls, config):
        """Factory function for metrics."""
        return cls(config)

    def __init__(self, config):
        pass

    def score(self, hypothesis: str, reference: str) -> float:
        """Computes the value of a metric for a hypothesis and a reference."""
        pass

    def score_all(self, hypotheses: Sequence[str], references: Sequence[str]) -> float:
        """Scores a list of hypotheses."""
        for hypothesis, reference in zip(hypotheses, references):
            self.score(hypothesis, reference)
