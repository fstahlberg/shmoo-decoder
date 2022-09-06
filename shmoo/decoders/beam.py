from typing import Any, Dict

from shmoo.core.interface import Decoder


class BeamDecoder(Decoder):

    def process(self, features: Dict[str, Any]) -> None:
        features["output_ids"] = [token_id + 100 for token_id in
                                  features["input_ids"]]
