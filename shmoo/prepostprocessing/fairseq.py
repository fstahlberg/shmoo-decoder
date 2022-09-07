from typing import Any, Dict

from shmoo.core.interface import Preprocessor
from shmoo.core.interface import Postprocessor
from shmoo.core import utils
from shmoo.prepostprocessing import register_processor


@register_processor("FairseqTokenizerPreprocessor")
class FairseqTokenizerPreprocessor(Preprocessor):

    def __init__(self, config):
        task, args = utils.make_fairseq_task(
            [config['fairseq']['model_dir'],
             '--source-lang', config['fairseq']["src_lang"], "--target-lang",
             config['fairseq']["trg_lang"], '--tokenizer',
             config['fairseq']["tokenizer"]])
        self._tokenizer = task.build_tokenizer(args)

    def process(self, features: Dict[str, Any]) -> None:
        features["input_pretok"] = features["input_raw"]
        features["input_raw"] = self._tokenizer.encode(features["input_raw"])


@register_processor("FairseqTokenizerPostprocessor")
class FairseqTokenizerPostprocessor(Postprocessor):

    def __init__(self, config):
        task, args = utils.make_fairseq_task(
            [config['fairseq']['model_dir'],
             '--source-lang', config['fairseq']["src_lang"], '--target-lang',
             config['fairseq']["trg_lang"], '--tokenizer',
             config['fairseq']["tokenizer"]])
        self._tokenizer = task.build_tokenizer(args)

    def process(self, features: Dict[str, Any]) -> None:
        features["output_predetok"] = features["output_raw"]
        features["output_raw"] = self._tokenizer.decode(features["output_raw"])


@register_processor("FairseqBPEPreprocessor")
class FairseqBPEPreprocessor(Preprocessor):

    def __init__(self, config):
        task, args = utils.make_fairseq_task(
            [config['fairseq']['model_dir'],
             '--source-lang', config['fairseq']['src_lang'], '--target-lang',
             config['fairseq']['trg_lang'], '--bpe',
             config['fairseq']['bpe'], '--bpe-codes',
             f"{config['fairseq']['model_dir']}/bpecodes"])
        self._bpe = task.build_bpe(args)
        self._src_dict = task.src_dict

    def process(self, features: Dict[str, Any]) -> None:
        features["input_bpe"] = self._bpe.encode(features["input_raw"])
        features["input_ids"] = self._src_dict.encode_line(
            features["input_bpe"])


@register_processor("FairseqBPEPostprocessor")
class FairseqBPEPostprocessor(Postprocessor):

    def __init__(self, config):
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
                       features["output_ids"]]
        features["output_bpe"] = " ".join(bpe_symbols)
        features["output_raw"] = self._bpe.decode(features["output_bpe"])
