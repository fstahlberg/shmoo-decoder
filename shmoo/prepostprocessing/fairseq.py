from typing import Any, Dict

from shmoo.core.interface import Preprocessor
from shmoo.core.interface import Postprocessor
from shmoo.core import utils
from shmoo.prepostprocessing import register_processor


@register_processor("FairseqTokenizerPreprocessor")
class FairseqTokenizerPreprocessor(Preprocessor):

    def __init__(self, config):
        super().__init__(config)
        task, args = utils.make_fairseq_task(
            [config['fairseq']['model_dir'],
             '--source-lang', config['fairseq']["src_lang"], "--target-lang",
             config['fairseq']["trg_lang"], '--tokenizer',
             config['fairseq']["tokenizer"]])
        self._tokenizer = task.build_tokenizer(args)

    def process(self, features: Dict[str, Any]) -> None:
        features["input"]["fairseq_tokenizer"] = self._tokenizer.encode(
            utils.get_last_item(features["input"])[1])


@register_processor("FairseqTokenizerPostprocessor")
class FairseqTokenizerPostprocessor(Postprocessor):

    def __init__(self, config):
        super().__init__(config)
        task, args = utils.make_fairseq_task(
            [config['fairseq']['model_dir'],
             '--source-lang', config['fairseq']["src_lang"], '--target-lang',
             config['fairseq']["trg_lang"], '--tokenizer',
             config['fairseq']["tokenizer"]])
        self._tokenizer = task.build_tokenizer(args)

    def process(self, features: Dict[str, Any]) -> None:
        features["output"]["fairseq_tokenizer"] = self._tokenizer.decode(
            utils.get_last_item(features["output"])[1])


@register_processor("FairseqBPEPreprocessor")
class FairseqBPEPreprocessor(Preprocessor):

    def __init__(self, config):
        super().__init__(config)
        task, args = utils.make_fairseq_task(
            [config['fairseq']['model_dir'],
             '--source-lang', config['fairseq']['src_lang'], '--target-lang',
             config['fairseq']['trg_lang'], '--bpe',
             config['fairseq']['bpe'], '--bpe-codes',
             f"{config['fairseq']['model_dir']}/bpecodes"])
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
        task, args = utils.make_fairseq_task(
            [config['fairseq']['model_dir'],
             '--source-lang', config['fairseq']['src_lang'], '--target-lang',
             config['fairseq']['trg_lang'], '--bpe',
             config['fairseq']['bpe'], '--bpe-codes',
             f"{config['fairseq']['model_dir']}/bpecodes"])
        self._bpe = task.build_bpe(args)
        self._tgt_dict = task.tgt_dict

    def process(self, features: Dict[str, Any]) -> None:
        bpe_symbols = [self._tgt_dict[token_id] for token_id in
                       utils.get_last_item(features["output"])[1]]
        features["output"]["fairseq_bpe"] = " ".join(bpe_symbols)
        features["output"]["raw"] = self._bpe.decode(" ".join(bpe_symbols))


@register_processor("FairseqSplitPreprocessor")
class FairseqSplitPreprocessor(Preprocessor):
    """Simply splits the string and maps to ids"""

    def __init__(self, config):
        super().__init__(config)
        task, args = utils.make_fairseq_task(
            [config['fairseq']['model_dir'],
             '--source-lang', config['fairseq']['src_lang'], '--target-lang',
             config['fairseq']['trg_lang']])
        self._src_dict = task.src_dict

    def process(self, features: Dict[str, Any]) -> None:
        features["input"]["ids"] = self._src_dict.encode_line(
            utils.get_last_item(features["input"])[1])


@register_processor("FairseqSplitPostprocessor")
class FairseqSplitPostprocessor(Postprocessor):

    def __init__(self, config):
        super().__init__(config)
        task, args = utils.make_fairseq_task(
            [config['fairseq']['model_dir'],
             '--source-lang', config['fairseq']['src_lang'], '--target-lang',
             config['fairseq']['trg_lang']])
        self._tgt_dict = task.tgt_dict

    def process(self, features: Dict[str, Any]) -> None:
        # convert from output_ids to "raw" output, meaning a string
        output_symbols = [self._tgt_dict[token_id] for token_id in
                          utils.get_last_item(features["output"])[1]]
        features["output"]["raw"] = " ".join(output_symbols)
