"""Beam search based decoding strategies."""

from absl import logging
from typing import Any, Dict, Sequence

from shmoo.core.interface import Decoder
from shmoo.core import utils
from shmoo.decoders import register_decoder

# Beam size used if beam_size is not set in the config.
DEFAULT_BEAM_SIZE = 4


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
