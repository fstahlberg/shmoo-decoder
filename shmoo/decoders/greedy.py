from typing import Any, Dict, Sequence

from shmoo.core.interface import Decoder
from shmoo.decoders import register_decoder


@register_decoder("GreedyDecoder")
class GreedyDecoder(Decoder):

    def process(
            self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        hypo = self.make_initial_hypothesis(input_features)
        while not hypo.is_final():
            prediction = self.get_predictions([hypo], nbest=1)[0]
            hypo = hypo.expand(prediction)
        return []
