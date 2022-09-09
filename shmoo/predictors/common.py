"""Basic general-purpose predictors."""

from typing import Any, Dict, Sequence

from shmoo.core import utils
from shmoo.core.interface import Prediction
from shmoo.core.interface import Predictor
from shmoo.predictors import register_predictor


@register_predictor("TokenBoost")
class TokenBoostPredictor(Predictor):
    """Boosts the score of a single (constant) token ID in each time step."""

    def __init__(self, config):
        super().__init__(config)
        self._token_id = utils.get_from_config(config, "token_boost_token_id")
        self._factor = utils.get_from_config(
            config, "token_boost_factor", default=1.0)

    def predict_next(self, states: Sequence[Dict[str, Any]],
                     scores: ...) -> ...:
        scores[:, self._token_id] += self._factor
        return scores


@register_predictor("LengthNorm")
class LengthNormPredictor(Predictor):
    """Length normalization following Wu et al. (2016)."""

    def __init__(self, config):
        super().__init__(config)
        self._alpha = utils.get_from_config(
            config, "length_norm_alpha", default=1.0)

    def predict_next(self, states: Sequence[Dict[str, Any]],
                     scores: ...) -> ...:
        """Length norm is a noop on partial hypotheses."""
        return scores

    def postprocess_output_features(self, state: Dict[str, Any],
                                    output_features: Dict[str, Any]) -> None:
        """Applies length normalization to a final hypothesis."""
        lp = (5.0 + len(output_features["output"]["ids"])) / (5.0 + 1.0)
        output_features["score"] /= lp ** self._alpha


@register_predictor("ScoreRecorder")
class ScoreRecorderPredictor(Predictor):
    """Adds an output feature containing the partial scores at each timestep."""

    def initialize_state(
            self, input_features: Dict[str, Any]) -> Dict[str, Any]:
        return {"partial_scores": []}

    def update_single_state(self, state: Dict[str, Any],
                            prediction: Prediction,
                            lazy: bool = False) -> Dict[str, Any]:
        if lazy:
            state["partial_scores"].append(prediction.score)
            return state
        else:
            return {
                "partial_scores": state["partial_scores"] + [prediction.score]}

    def predict_next(self, states: Sequence[Dict[str, Any]],
                     scores: ...) -> ...:
        """The ScoreRecorder is a noop on partial hypotheses."""
        return scores

    def postprocess_output_features(self, state: Dict[str, Any],
                                    output_features: Dict[str, Any]) -> None:
        """Writes the partial scores to the partial_scores output feature."""
        output_features["partial_scores"] = state["partial_scores"]
