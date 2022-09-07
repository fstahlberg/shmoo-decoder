from typing import Any, Dict, Sequence

from shmoo.core.interface import Decoder
from shmoo.core import utils
from shmoo.decoders import register_decoder


@register_decoder("BeamDecoder")
class BeamDecoder(Decoder):

    def __init__(self, config):
        super().__init__(config)
        self._finished_criterion = self.all_hypos_finished
        try:
            self.beam_size = config["decoder_config"]["beam_size"]
        except KeyError:
            self.beam_size = utils.DEFAULT_BEAM_SIZE

    def process(
            self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        hypos = [self.make_initial_hypothesis(input_features)]
        while not self._finished_criterion(hypos):
            predictions = self.get_predictions(hypos, nbest=self.beam_size)
            hypos = [self.make_hypothesis(prediction) for prediction in
                     predictions]
        return self.make_final_output_features(input_features, hypos)
