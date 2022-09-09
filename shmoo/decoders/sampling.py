from absl import logging
from typing import Any, Dict, Sequence
from functools import partial

import numpy as np
from scipy.special import softmax

from shmoo.core import utils
from shmoo.core.interface import Decoder
from shmoo.decoders import register_decoder


@register_decoder("SamplingDecoder")
class SamplingDecoder(Decoder):

    def __init__(self, config):
        super().__init__(config)
        self._finished_criterion = self.all_hypos_finished
        self.seed = utils.get_from_decoder_config(config, 'seed', utils.DEFAULT_SEED)
        self.num_samples = utils.get_from_decoder_config(config, 'num_samples', utils.DEFAULT_NUM_SAMPLES)
        self.strategy = utils.get_from_decoder_config(config, 'strategy', utils.DEFAULT_SAMPLING_STRATEGY)

        if self.strategy == 'temperature':
            self.temp = utils.get_from_decoder_config(config, 'temperature', utils.DEFAULT_TEMPERATURE)
            self.make_probs = partial(self.normalise_scores, temp=self.temp)
        elif self.strategy == 'top k':
            self.k = utils.get_from_decoder_config(config, 'k', utils.DEFAULT_TOP_K)
            self.make_probs = partial(self.normalise_top_k_scores, k=self.k)
        elif self.strategy == 'nucleus':
            self.cutoff_nucleus_p = utils.get_from_decoder_config(config, 'p', utils.DEFAULT_NUCLEUS_P)
            self.make_probs = partial(self.normalise_scores_acc_to_prob_mass, cutoff_p=self.cutoff_nucleus_p)
        elif self.strategy == 'typical':
            self.cutoff_typical_p = utils.get_from_decoder_config(config, 'typical p', utils.DEFAULT_TYPICAL_P)
            self.make_probs = partial(self.normalise_scores_acc_to_expected_info, cutoff_p=self.cutoff_typical_p)
        else:
            raise NotImplementedError(f"{self.strategy} is not implemented.")

        logging.info(f"Sampling Decoder using successfully initialized.")

    def normalise_scores(self, log_probs: np.ndarray, temp: float) -> np.ndarray:
        return np.exp(log_probs) if temp == 1 else softmax(log_probs / temp)

    def normalise_top_k_scores(self, log_probs: np.ndarray, k: int) -> np.ndarray:
        flat_indices = np.argpartition(-log_probs, k, axis=None)
        sparse_probs = np.zeros(log_probs.size)
        sparse_probs[flat_indices[:k]] = np.exp(log_probs[flat_indices[:k]])
        return sparse_probs / np.sum(sparse_probs)

    def _normalise_scores_acc_to_prob_mass(self, log_probs: np.ndarray, scores: np.ndarray, cutoff_p: float) -> np.ndarray:
        flat_indices = np.argsort(scores)
        sorted_probs = np.exp(log_probs[flat_indices])
        crit_ind = (np.cumsum(sorted_probs) < cutoff_p - np.finfo(np.float32).eps).argmin()
        sparse_probs = np.zeros(log_probs.size)
        sparse_probs[flat_indices[:crit_ind+1]] = np.exp(log_probs[flat_indices[:crit_ind+1]])
        return sparse_probs / np.sum(sparse_probs)

    def normalise_scores_acc_to_prob_mass(self, log_probs: np.ndarray, cutoff_p: float) -> np.ndarray:
        return self._normalise_scores_acc_to_prob_mass(scores=-log_probs,
                                                  log_probs=log_probs,
                                                  cutoff_p=cutoff_p)

    def normalise_scores_acc_to_expected_info(self, log_probs: np.ndarray, cutoff_p: float) -> np.ndarray:
        entropy = -np.sum(np.exp(log_probs) * log_probs)
        shifted_abs_log_probs = np.abs(entropy + log_probs)
        return self._normalise_scores_acc_to_prob_mass(scores=shifted_abs_log_probs,
                                                       log_probs=log_probs,
                                                       cutoff_p=cutoff_p)

    def process(
            self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        hypos = [self.make_initial_hypothesis(input_features) for _ in range(self.num_samples)]
        while not self._finished_criterion(hypos):
            predictions = self.sample_predictions(hypos,
                                                  seed=self.seed,
                                                  make_probs=self.make_probs)
            hypos = [self.make_hypothesis(prediction, lazy=True) for prediction in predictions]
        return self.make_final_output_features(input_features, hypos)
