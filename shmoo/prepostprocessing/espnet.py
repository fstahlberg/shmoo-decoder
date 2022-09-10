"""
Input pre-processing and output post-processing for ESPnet2 models
- Input can be text, speech features in kaldi_ark format
- Output will be text
- Support for speech recogition, text-to-text and speech-to-text translation
"""

from absl import logging
from collections import OrderedDict
from typing import Any, Dict
import os
import pickle
from shmoo.core import utils
from shmoo.core.interface import Preprocessor, Postprocessor, Processor
from shmoo.prepostprocessing import register_processor

try:
    from yaml import full_load
    import sentencepiece as spm
    import kaldiio

    # import soundfile
except ImportError:
    logging.info("ESPnet pre/postprocessing not available.")
else:
    logging.info("ESPnet pre/postprocessing imports successful.")


CHECK = "\u2713"


# @register_processor("ESPnetProcessor")
class ESPnetProcessor(Processor):
    def __init__(self, config: str):
        """

        Args:
            config (dict): The shmoo config
        """

        #  espnet config file
        config_file = config["espnet"]["config"]

        self.config_file = config_file
        self.config = {}
        with open(self.config_file, "r") as fpr:
            self.config = full_load(fpr)

        self.int2token = OrderedDict()
        self.token2int = OrderedDict()

        self.tokenizer_model_file = None
        self.tokenizer_model = None

        self.input_type = None
        self.input_format = None

        self.check()
        self.set_up()

    def check(self):
        """Checks if the ESPnet2 config file has all the necessary keys"""

        logging.info("Checking ESPnet2 config yaml")

        for key in [
            "train_data_path_and_name_and_type",
            "token_list",
            "token_type",
            "model",
        ]:

            if key == "train_data_path_and_name_and_type":
                _, self.input_type, self.input_format = self.config[key][0]
                if self.input_type == "speech":
                    if self.input_format != "kaldi_ark":
                        logging.error(
                            f"Input format_type {self.input_format} not supported yet. Currently supports kaldi_ark formatted speech features.",
                        )
                        raise NotImplementedError

                logging.info("Input modaility: {:s}".format(self.input_type))
                logging.info("Input format   : {:s}".format(self.input_format))

            if key == "model":
                key = self.config["token_type"] + "model"
                if os.path.exists(self.config[key]):
                    self.tokenizer_model_file = self.config[key]
                else:
                    logging.error(
                        "Error: Cannot find {:s} file at {:s}".format(
                            key, self.config[key]
                        )
                    )
                    raise FileNotFoundError(self.config[key])

            if key not in self.config:
                raise KeyError(
                    "Error: {:s} is expected to be found in ESPnet config file {:s}".format(
                        key, self.config_file
                    )
                )

            else:
                logging.info("{:s} {:s}".format(key, CHECK))

    def set_up(self):
        """Load tokenizer, and build token2int, int2token mappings"""

        for i, tok in enumerate(self.config["token_list"]):
            self.token2int[tok] = i
            self.int2token[i] = tok
        logging.info("Loaded token2int.")
        logging.info("Vocab size: {:d}".format(len(self.token2int)))
        # since <sos/eos> is the last token in ESPnet2 pre-processing
        self.bos_index = len(self.token2int)
        self.eos_index = len(self.token2int)
        logging.info("bos/eos index {:d}".format(self.eos_index))

        if self.config["token_type"] == "bpe":

            self.tokenizer_model = spm.SentencePieceProcessor()
            self.tokenizer_model.load(self.tokenizer_model_file)
            logging.info("Loaded sentencepiece model.")

        else:
            # Load tokenizer model (eg: spm or char tokenizer or ..)
            with open(self.tokenizer_model_file, "rb") as fpr:
                self.tokenizer_model = pickle.load(fpr)


@register_processor("ESPnetPreprocessor")
class ESPnetPreprocessor(Preprocessor, ESPnetProcessor):
    """Input preprocessor for ESPnet"""

    def __init__(self, config: dict):
        """Initialize the preprocessor"""

        logging.info("Setting up espnet preprocessor")
        super().__init__(config)
        logging.info("Preprocessor setup done")

    def process(self, features: Dict[str, Any]) -> None:
        """Tokenize raw text sequence to ids"""

        raw_feats = utils.get_last_item(features["input"])[1]

        if self.input_type == "text":
            if self.config["token_type"] == "bpe":
                features["input"]["ids"] = [
                    self.token2int[tok]
                    for tok in self.tokenizer_model.EncodeAsPieces(raw_feats)
                ]
            else:
                raise NotImplementedError("Input type should be either text or speech.")

        elif self.input_type == "speech":
            if self.input_format == "kaldi_ark":

                features["input"]["ids"] = kaldiio.load_mat(raw_feats)

                logging.info(
                    "{:s} ({:d},{:d})".format(
                        features["input"]["raw"], *features["input"]["ids"].shape
                    )
                )

            else:
                raise NotImplementedError(
                    "Input format {:s} for input type {:s} not supported yet".format(
                        self.input_format, self.input_type
                    )
                )


@register_processor("ESPnetPostprocessor")
class ESPnetPostprocessor(Postprocessor, ESPnetProcessor):
    """Post processor for outputs produced by ESPnet2"""

    def __init__(self, config: dict):
        """Initialize the post processor"""

        logging.info("Setting up espnet postprocessor")
        super().__init__(config)
        logging.info("Postprocessor setup done")

    def process(self, features: Dict[str, Any]) -> None:
        """Post process token IDs to sequence of token strings"""

        if self.config["token_type"] == "bpe":

            features["output"]["bpe"] = [
                self.int2token[i] for i in utils.get_last_item(features["output"])[1]
            ]

            features["output"]["raw"] = self.tokenizer_model.DecodePieces(
                features["output"]["bpe"]
            )
