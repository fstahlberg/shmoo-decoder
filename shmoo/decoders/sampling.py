from absl import logging
from typing import Any, Dict, Sequence

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
        logging.info(f"Sampling Decoder successfully initialized.")
        logging.info(f"Number of samples: {self.num_samples}")

    def process(
            self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        hypos = [self.make_initial_hypothesis(input_features) for _ in range(self.num_samples)]
        while not self._finished_criterion(hypos):
            predictions = self.sample_predictions(hypos, seed=self.seed)
            hypos = [self.make_hypothesis(prediction, lazy=True) for prediction in predictions]
        return self.make_final_output_features(input_features, hypos)