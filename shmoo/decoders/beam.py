from typing import Any, Dict, Sequence

from shmoo.core.interface import Decoder
from shmoo.core.interface import Hypothesis
from shmoo.decoders import register_decoder


def best_hypo_finished(hypos: Sequence[Hypothesis]) -> bool:
    if hypos:
        return hypos[0].is_finished()
    return True


def all_hypos_finished(hypos: Sequence[Hypothesis]) -> bool:
    return all(hypo.is_finished() for hypo in hypos)


@register_decoder("BeamDecoder")
class BeamDecoder(Decoder):

    def __init__(self):
        super().__init__()
        self._finished_criterion = all_hypos_finished
        self.beam_size = 4

    def process(
            self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        hypos = [self.make_initial_hypothesis(input_features)]
        while not self._finished_criterion(hypos):
            predictions = self.get_predictions(hypos, nbest=self.beam_size)
            hypos = [self.make_hypothesis(prediction) for prediction in
                     predictions]
        return self.make_final_output_features(input_features, hypos)
