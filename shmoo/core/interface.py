from collections import OrderedDict
from typing import Any, Dict, Sequence
from yaml import full_load
import sys
import os
import pickle
import sentencepiece as spm
import numpy as np


class Prediction:
    pass


class Predictor:

    def initialize_state(self, input_features: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def update_states(self, states: Sequence[Dict[str, Any]],
                      predictions: Sequence[Prediction]) -> None:
        for state, prediction in zip(states, predictions):
            self.update_single_state(state, prediction)

    def update_single_state(self, state: Dict[str, Any],
                            prediction: Prediction) -> None:
        pass

    def predict_next(self, states: Sequence[Dict[str, Any]]):
        all_scores = []
        for state in enumerate(states):
            all_scores.append(self.predict_next_single(state))
        return np.stack(all_scores)

    def predict_next_single(self, state: Dict[str, Any]):
        pass


class Processor:
    def process(self, features: Dict[str, Any]) -> None:
        pass


class ESPnetProcessor(Processor):

    def __init__(self, config_file: str):
        """

        Args:
            config_file (str): Path to ESPnet2 config.yaml file of a trained model
        """

        self.config_file = config_file
        self.config = {}
        with open(self.config_file, "r") as fpr:
            self.config = full_load(fpr)

        self.int2token = OrderedDict()
        self.token2int = OrderedDict()

        self.tokenizer_model_file = None
        self.tokenizer_model = None
        self.eos_id = None

        self.input_type = None
        self.input_format = None

        self.check()
        self.set_up()


    def check(self):
        """Checks if the config file has all the necessary keys"""

        print("- Checking config yaml")

        for key in ['train_data_path_and_name_and_type', 'token_list', 'token_type', 'model']:

            if key == "train_data_path_and_name_and_type":
                _, input_type, input_format = self.config[key][0]
                if input_type == "speech":
                    if input_format in ("kaldi_ark", "sound"):
                        self.input_type = input_type
                        self.input_format = input_format
                    else:
                        print(f"Input format_type {input_format} not supported yet. Currently support kaldi_ark formatted features or raw audio/sound.", file=sys.stderr)
                        raise NotImplementedError
                print("  - Input modaility:", input_type)
                print("  - Input format   :", input_format)

            if key == "model":
                key = self.config['token_type'] + "model"
                if os.path.exists(self.config[key]):
                    self.tokenizer_model_file = self.config[key]
                else:
                    print(f"Error: Cannot find {key} file at", self.config[key], file=sys.stderr)
                    raise FileNotFoundError(self.config[key])

            if key not in self.config:
                print(f"Error: {key} is expected to be found in ESPnet config file", self.config_file, file=sys.stderr
                )
                raise KeyError

            else:
                print("  -", key, u'\u2713')

    def set_up(self):

        for i, tok in enumerate(self.config['token_list']):
            self.token2int[tok] = i
            self.int2token[i] = tok
        print("- Loaded token2int.")
        print("- Vocab size:", len(self.token2int))
        # since <sos/eos> is the last token in ESPnet2 pre-processing
        self.eos_id = len(self.token2int)
        print("- Eos ID", self.eos_id)

        if self.config["token_type"] == "bpe":

            self.tokenizer_model = spm.SentencePieceProcessor()
            self.tokenizer_model.load(self.tokenizer_model_file)
            print("- Loaded sentencepiece model.")

        else:
            # Load tokenizer model (eg: spm or char tokenizer or ..)
            with open(self.tokenizer_model_file, 'rb') as fpr:
                self.tokenizer_model = pickle.load(fpr)

    def process(self, features: Dict[str, Any]) -> None:
        """Tokenize text from `input_raw` and save the ids as values for `input_ids` """

        if self.input_type == "text":
            if self.config['token_type'] == 'bpe':
                features['input_ids'] = [
                    self.token2int[tok]
                    for tok in self.tokenizer_model.EncodeAsPieces(features['input_raw'])
                ]
                print(features["input_raw"], features["input_ids"])
            else:
                raise NotImplementedError

        elif self.input_type == "speech":
            if self.input_format == "kaldi_ark":

                try:
                    import kaldiio
                    features["input_ids"] = kaldiio.load_mat(features['input_raw'])

                    print(features["input_raw"], features["input_ids"].shape)

                except ModuleNotFoundError:
                    pass

            elif self.input_format == "sound":
                try:
                    import soundfile
                except ModuleNotFoundError:
                    pass

            else:
                raise NotImplementedError


class Preprocessor(Processor):
    pass


class Postprocessor(Processor):
    pass


class Decoder:

    def __init__(self):
        self._predictors = []

    def add_predictor(self, predictor: Predictor):
        self._predictors.append(predictor)

    def decode(self, input_features: Dict[str, Any]) -> Sequence[
        Dict[str, Any]]:
        pass
