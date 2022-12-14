"""Encoding and decoding with a sentencepiece model.

Uses the sentencepiece library
    https://github.com/google/sentencepiece/
to convert between plain text and token ID sequences.
"""

from absl import logging
from typing import Any, Dict

from shmoo.core import utils
from shmoo.core.interface import Postprocessor
from shmoo.core.interface import Preprocessor
from shmoo.prepostprocessing import register_processor

try:
    import sentencepiece as spm
except ImportError:
    logging.info("SPM not available.")
else:
    logging.info("SPM imports successful.")


@register_processor("SPMPreprocessor")
class SPMPreprocessor(Preprocessor):
    """Encodes a plain text string with an SPM model."""

    def __init__(self, config):
        super().__init__(config)
        self._spm = spm.SentencePieceProcessor(
            model_file=utils.get_from_config(config, "spm_path"))

    def process(self, features: Dict[str, Any]) -> None:
        features["input"]["ids"] = self._spm.encode(
            utils.get_last_item(features["input"])[1])


@register_processor("SPMPostprocessor")
class SPMPostprocessor(Postprocessor):
    """Converts a sequence of SPM token IDs to a plain text string."""

    def __init__(self, config):
        super().__init__(config)
        self._spm = spm.SentencePieceProcessor(
            model_file=utils.get_from_config(config, "spm_path"))

    def process(self, features: Dict[str, Any]) -> None:
        features["output"]["raw"] = self._spm.decode(
            [utils.get_last_item(features["output"])[1]])[0]
