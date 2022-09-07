from typing import Any, Dict, Sequence

from shmoo.core.interface import Decoder
from shmoo.decoders import register_decoder


@register_decoder("BeamDecoder")
class BeamDecoder(Decoder):

    def process(
            self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        # input_features["output_ids"] = [token_id + 100 for token_id in
        #                                 input_features["input_ids"]]
        input_features["output_ids"] = input_features["input_ids"] + 100
        return [input_features]
