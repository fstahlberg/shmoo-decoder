from typing import Any, Dict, Sequence

from shmoo.core import utils
from shmoo.core.interface import Decoder
from shmoo.decoders import register_decoder

@register_decoder("SamplingDecoder")
class SamplingDecoder(Decoder):

    def __init__(self, config):
        super().__init__(config)
        self._finished_criterion = self.all_hypos_finished
        self.seed = config.get('decoder_config', {}).get('seed', utils.DEFAULT_SEED)
        self.num_samples = config.get('decoder_config', {}).get('num_samples', utils.DEFAULT_NUM_SAMPLES)

    def process(
            self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        hypos = [self.make_initial_hypothesis(input_features) for i in range(self.num_samples)]
        while not self._finished_criterion(hypos):
            predictions = self.sample_predictions(hypos, seed=self.seed)
            hypos = [self.make_hypothesis(prediction) for prediction in predictions]
        return self.make_final_output_features(input_features, hypos)