from typing import Any, Dict

from shmoo.core.interface import Preprocessor
from shmoo.core.interface import Postprocessor
from shmoo.prepostprocessing import register_processor


@register_processor("TrivialTokenPreprocessor")
class TrivialTokenPreprocessor(Preprocessor):

    def process(self, features: Dict[str, Any]) -> None:
        features["input_ids"] = [int(token) for token in
                                 features["input_raw"].split()]


@register_processor("TrivialTokenPostprocessor")
class TrivialTokenPostprocessor(Postprocessor):

    def process(self, features: Dict[str, Any]) -> None:
        features["output_raw"] = " ".join(
            [str(token_id) for token_id in features["output_ids"]])
