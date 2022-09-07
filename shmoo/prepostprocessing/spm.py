from typing import Any, Dict

from shmoo.core.interface import Postprocessor
from shmoo.core.interface import Preprocessor
from shmoo.prepostprocessing import register_processor

try:
    import sentencepiece as spm
except ModuleNotFoundError:
    pass


@register_processor("SPMPreprocessor")
class SPMPreprocessor(Preprocessor):

    def __init__(self, spm_path: str):
        self._spm = spm.SentencePieceProcessor(model_file=spm_path)

    def process(self, features: Dict[str, Any]) -> None:
        features["input_ids"] = self._spm.encode(features['input_raw'])


@register_processor("SPMPostprocessor")
class SPMPostprocessor(Postprocessor):

    def __init__(self, spm_path: str):
        self._spm = spm.SentencePieceProcessor(model_file=spm_path)

    def process(self, features: Dict[str, Any]) -> None:
        features["output_raw"] = self._spm.decode([features["output_ids"]])[0]
