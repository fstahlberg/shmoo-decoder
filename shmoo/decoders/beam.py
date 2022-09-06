from typing import Any, Dict, Sequence

from shmoo.core.interface import Decoder


class BeamDecoder(Decoder):

    def process(self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        input_features["output_ids"] = [token_id + 100 for token_id in
                                  input_features["input_ids"]]
        return [input_features]
