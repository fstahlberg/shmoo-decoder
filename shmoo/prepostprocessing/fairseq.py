"""Pre- and postprocessing adaptors for fairseq.

This module provides access to pre- and postprocessing strategies like
tokenizers and subword unit segmentations from the fairseq library.

https://github.com/pytorch/fairseq
"""

from typing import Any, Dict

from shmoo.core.interface import Preprocessor
from shmoo.core.interface import Postprocessor
from shmoo.core import utils
from shmoo.prepostprocessing import register_processor


@register_processor("FairseqTokenizerPreprocessor")
class FairseqTokenizerPreprocessor(Preprocessor):

    def __init__(self, config):
        super().__init__(config)
        task, args = utils.make_fairseq_task(config["fairseq"])
        self._tokenizer = task.build_tokenizer(args)

    def process(self, features: Dict[str, Any]) -> None:
        features["input"]["fairseq_tokenizer"] = self._tokenizer.encode(
            utils.get_last_item(features["input"])[1])


@register_processor("FairseqTokenizerPostprocessor")
class FairseqTokenizerPostprocessor(Postprocessor):

    def __init__(self, config):
        super().__init__(config)
        task, args = utils.make_fairseq_task(config['fairseq'])
        self._tokenizer = task.build_tokenizer(args)

    def process(self, features: Dict[str, Any]) -> None:
        features["output"]["fairseq_tokenizer"] = self._tokenizer.decode(
            utils.get_last_item(features["output"])[1])


@register_processor("FairseqBPEPreprocessor")
class FairseqBPEPreprocessor(Preprocessor):

    def __init__(self, config):
        super().__init__(config)
        task, args = utils.make_fairseq_task(config['fairseq'])
        self._bpe = task.build_bpe(args)
        self._src_dict = task.src_dict

    def process(self, features: Dict[str, Any]) -> None:
        bpe_codes = self._bpe.encode(utils.get_last_item(features["input"])[1])
        features["input"]["fairseq_bpe"] = bpe_codes
        features["input"]["ids"] = self._src_dict.encode_line(bpe_codes)


@register_processor("FairseqBPEPostprocessor")
class FairseqBPEPostprocessor(Postprocessor):

    def __init__(self, config):
        super().__init__(config)
        task, args = utils.make_fairseq_task(config['fairseq'])
        self._bpe = task.build_bpe(args)
        self._tgt_dict = task.tgt_dict

    def process(self, features: Dict[str, Any]) -> None:
        bpe_symbols = [self._tgt_dict[token_id] for token_id in
                       utils.get_last_item(features["output"])[1]]
        features["output"]["fairseq_bpe"] = bpe_symbols
        features["output"]["fairseq_postbpe"] = self._bpe.decode(
            " ".join(bpe_symbols))


@register_processor("FairseqSplitPreprocessor")
class FairseqSplitPreprocessor(Preprocessor):
    """Simply splits the string and maps to ids"""

    def __init__(self, config):
        super().__init__(config)
        task, args = utils.make_fairseq_task(config['fairseq'])
        self._src_dict = task.src_dict

    def process(self, features: Dict[str, Any]) -> None:
        features["input"]["ids"] = self._src_dict.encode_line(
            utils.get_last_item(features["input"])[1])


@register_processor("FairseqSplitPostprocessor")
class FairseqSplitPostprocessor(Postprocessor):

    def __init__(self, config):
        super().__init__(config)
        task, args = utils.make_fairseq_task(config['fairseq'])
        self._tgt_dict = task.tgt_dict

    def process(self, features: Dict[str, Any]) -> None:
        # convert from output_ids to "raw" output, meaning a string
        output_symbols = [self._tgt_dict[token_id] for token_id in
                          utils.get_last_item(features["output"])[1]]
        features["output"]["raw"] = " ".join(output_symbols)
