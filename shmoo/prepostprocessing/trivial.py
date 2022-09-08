from typing import Any, Dict

from shmoo.core import utils
from shmoo.core.interface import Preprocessor
from shmoo.core.interface import Postprocessor
from shmoo.prepostprocessing import register_processor


@register_processor("TrivialTokenPreprocessor")
class TrivialTokenPreprocessor(Preprocessor):

    def process(self, features: Dict[str, Any]) -> None:
        features["input"]["ids"] = [int(token) for token in
                                    utils.get_last_item(features["input"])[
                                        1].split()]


@register_processor("TrivialTokenPostprocessor")
class TrivialTokenPostprocessor(Postprocessor):

    def process(self, features: Dict[str, Any]) -> None:
        features["output"]["raw"] = " ".join(
            [str(token_id) for token_id in
             utils.get_last_item(features["output"])[1]])


@register_processor("RemoveEOSPostprocessor")
class RemoveEOSPostprocessor(Postprocessor):

    def process(self, features: Dict[str, Any]) -> None:
        features["output"]["ids_without_eos"] = utils.get_last_item(
            features["output"])[1][:-1]
