"""Beam search based decoding strategies."""

from absl import logging
from typing import Any, Dict, Sequence

import numpy as np

from shmoo.core.interface import Decoder
from shmoo.core import utils
from shmoo.decoders import register_decoder

# Beam size used if beam_size is not set in the config.
DEFAULT_BEAM_SIZE = 4
DEFAULT_TYPICAL_P = 0.5


@register_decoder("BeamDecoder")
class BeamDecoder(Decoder):
    """Vanilla beam search implementation."""

    def __init__(self, config):
        super().__init__(config)
        self._finished_criterion = self.all_hypos_finished
        self.beam_size = utils.get_from_decoder_config(
            config, 'beam_size', DEFAULT_BEAM_SIZE)
        logging.info(f"Beam Search Decoder successfully initialized.")
        logging.info(f"Beam size: {self.beam_size}")

    def process(
            self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        hypos = [self.make_initial_hypothesis(input_features)]
        while not self._finished_criterion(hypos):
            predictions = self.get_predictions(hypos, nbest=self.beam_size)
            hypos = [self.make_hypothesis(prediction) for prediction in
                     predictions]
        return self.make_final_output_features(input_features, hypos)

@register_decoder("TypicalBeamDecoder")
class TypicalBeamDecoder(BeamDecoder):

    def __init__(self, config):
        super().__init__(config)
        self.cutoff_p = utils.get_from_decoder_config(config, 'cutoff_typical_p', DEFAULT_BEAM_SIZE)
        logging.info(f"Typical Beam Search Decoder successfully initialized.")

    def _normalise_scores_acc_to_prob_mass(self, log_probs: np.ndarray,
                                           scores: np.ndarray,
                                           cutoff_p: float) -> np.ndarray:
        indices = np.argsort(scores, axis=1)
        sorted_probs = np.exp(log_probs[np.arange(log_probs.shape[0])[:,np.newaxis],indices])
        crit_ind = (np.cumsum(sorted_probs, axis=1) < cutoff_p - np.finfo(
            np.float32).eps).argmin(axis=1)
        sparse_probs = np.zeros(log_probs.shape)
        for i in range(log_probs.shape[0]):
            sparse_probs[i,indices[i,:crit_ind[i]+1]] = np.exp(log_probs[i,indices[i,:crit_ind[i]+1]])
        return sparse_probs / np.expand_dims(np.sum(sparse_probs, axis=1), axis=1)

    def normalise_scores_acc_to_expected_info(self, log_probs: np.ndarray,
                                              cutoff_p: float) -> np.ndarray:
        entropy = np.expand_dims(-np.sum(np.exp(log_probs) * log_probs, axis=1), axis=1)
        shifted_abs_log_probs = np.abs(entropy + log_probs)
        return self._normalise_scores_acc_to_prob_mass(
            scores=shifted_abs_log_probs,
            log_probs=log_probs,
            cutoff_p=cutoff_p)

    def modify_scores(self, scores):
        return self.normalise_scores_acc_to_expected_info(log_probs=scores, cutoff_p=self.cutoff_p)
