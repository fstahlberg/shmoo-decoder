from typing import Any, Dict

from shmoo.core.interface import Postprocessor
from shmoo.core.interface import Preprocessor


class TrivialTokenPreprocessor(Preprocessor):

    def process(self, features: Dict[str, Any]) -> None:
        features["input_ids"] = [int(token) for token in
                                 features["input_raw"].split()]


class TrivialTokenPostprocessor(Postprocessor):

    def process(self, features: Dict[str, Any]) -> None:
        features["output_raw"] = " ".join(
            [str(token_id) for token_id in features["output_ids"]])
