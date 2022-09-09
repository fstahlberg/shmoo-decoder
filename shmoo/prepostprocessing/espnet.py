from absl import logging
from collections import OrderedDict
from typing import Any, Dict
import sys
import os
import pickle
from shmoo.core import utils
from shmoo.core.interface import Preprocessor, Postprocessor, Processor
from shmoo.prepostprocessing import register_processor

try:
    from yaml import full_load
    import sentencepiece as spm
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
            config_file (str): Path to ESPnet2 config.yaml file of a trained model
        """

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
        """Checks if the config file has all the necessary keys"""

        logging.info("Checking config yaml")

        for key in [
            "train_data_path_and_name_and_type",
            "token_list",
            "token_type",
            "model",
        ]:

            if key == "train_data_path_and_name_and_type":
                _, input_type, input_format = self.config[key][0]
                if input_type == "speech":
                    if input_format in ("kaldi_ark", "sound"):
                        self.input_type = input_type
                        self.input_format = input_format
                    else:
                        logging.error(
                            f"Input format_type {input_format} not supported yet. Currently supports kaldi_ark formatted features or raw audio/sound.",
                        )
                        raise NotImplementedError
                logging.info("Input modaility: {:s}".format(input_type))
                logging.info("Input format   : {:s}".format(input_format))

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
    def __init__(self, config_file):

        logging.info("Setting up espnet preprocessor")
        super().__init__(config_file)
        logging.info("Preprocessor setup done")

    def process(self, features: Dict[str, Any]) -> None:
        """Tokenize text from `input_raw` and save the ids as values for `input_ids`"""

        if self.input_type == "text":
            if self.config["token_type"] == "bpe":
                features["input"]["ids"] = [
                    self.token2int[tok]
                    for tok in self.tokenizer_model.EncodeAsPieces(
                        features["input"]["raw"]
                    )
                ]
                print(features["input"]["raw"], features["input"]["ids"])
            else:
                raise NotImplementedError

        elif self.input_type == "speech":
            if self.input_format == "kaldi_ark":

                try:
                    import kaldiio

                    features["input"]["ids"] = kaldiio.load_mat(
                        features["input"]["raw"]
                    )

                    logging.info(
                        "{:s} ({:d},{:d})".format(
                            features["input"]["raw"], *features["input"]["ids"].shape
                        )
                    )

                except ModuleNotFoundError:
                    pass

            elif self.input_format == "sound":
                try:
                    import soundfile
                except ImportError:
                    logging.info("soundfile not found. pip install soundfile")

            else:
                raise NotImplementedError


@register_processor("ESPnetPostprocessor")
class ESPnetPostprocessor(Postprocessor, ESPnetProcessor):
    def __init__(self, config_file: str):

        logging.info("Setting up espnet postprocessor")
        super().__init__(config_file)
        logging.info("Postprocessor setup done")

    def process(self, features: Dict[str, Any]) -> None:
        """Post process bpe IDs to sequence of tokens"""

        if self.config["token_type"] == "bpe":

            features["output"]["bpe"] = [
                self.int2token[i] for i in utils.get_last_item(features["output"])[1]
            ]

            features["output"]["raw"] = self.tokenizer_model.DecodePieces(
                features["output"]["bpe"]
            )
