"""

This is the interface to the fairseq library.

https://github.com/pytorch/fairseq

It is based on the fairseq predictor for SGNMT from
https://github.com/bpopeters/sgnmt/blob/master/cam/sgnmt/predictors/pytorch_fairseq.py
"""

import logging
import os
from typing import Dict, List, Optional, Any

from shmoo.core.interface import Predictor
from shmoo.core.interface import Prediction
from shmoo.core import utils
from shmoo.predictors import register_predictor

try:
    # Requires fairseq
    from fairseq import checkpoint_utils, options, tasks
    from fairseq import utils as fairseq_utils
    from fairseq.sequence_generator import EnsembleModel
    import torch
    from torch import Tensor
except ImportError:
    pass  # Deal with it in decode.py

# here: constants and functions copied from SGNMT's utils module: they
# ultimately belong somewhere else
GO_ID = 1
"""Reserved word ID for the start-of-sentence symbol. """

EOS_ID = 2
"""Reserved word ID for the end-of-sentence symbol. """

UNK_ID = 0
"""Reserved word ID for the unknown word (UNK). """

NOTAPPLICABLE_ID = 3
"""Reserved word ID which is currently not used. """

NEG_INF = float("-inf")

INF = float("inf")

EPS_P = 0.00001


def common_get(obj, key, default):
    """Can be used to access an element via the index or key.
    Works with numpy arrays, lists, and dicts.

    Args:
        ``obj`` (list,array,dict):  Mapping
        ``key`` (int): Index or key of the element to retrieve
        ``default`` (object): Default return value if ``key`` not found

    Returns:
        ``obj[key]`` if ``key`` in ``obj``, otherwise ``default``
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    else:
        return obj[key] if key < len(obj) else default


def oov_to_unk(seq, vocab_size, unk_idx=None):
    if unk_idx is None:
        unk_idx = UNK_ID
    return [x if x < vocab_size else unk_idx for x in seq]


# end things from SGNMT's utils module


FAIRSEQ_INITIALIZED = False
"""Set to true by _initialize_fairseq() after first constructor call."""


def _initialize_fairseq(user_dir):
    global FAIRSEQ_INITIALIZED
    if not FAIRSEQ_INITIALIZED:
        logging.info("Setting up fairseq library...")
        if user_dir:
            args = type("", (), {"user_dir": user_dir})()
            fairseq_utils.import_user_module(args)
        FAIRSEQ_INITIALIZED = True


@register_predictor("FairseqPredictor")
class FairseqPredictor(Predictor):

    def __init__(self):
        """
        Check https://github.com/bpopeters/sgnmt/blob/master/cam/sgnmt/predictors/pytorch_fairseq.py
        for an idea of how the model can actually be loaded
        """
        model_path = "/home/fstahlberg/work/shmoo/wmt14.en-fr.fconv-py/model.pt"
        user_dir = ""
        self._lang_pair = "en-fr"

        _initialize_fairseq(user_dir)

        self.device = torch.device("cpu")
        task = self._load_task(model_path)
        self.src_vocab_size = len(task.source_dictionary)

        self.model = self._build_ensemble(model_path, task)

    def _load_task(self, model_path):
        input_args = ["--path", model_path, os.path.dirname(model_path)]

        if self._lang_pair:
            src, trg = self._lang_pair.split("-")
            input_args.extend(["--source-lang", src, "--target-lang", trg])
        input_args.extend(
            ["--tokenizer", "moses", "--bpe", "subword_nmt", "--bpe-codes", "/home/fstahlberg/work/shmoo/wmt14.en-fr.fconv-py/bpecodes"])

        task, args = utils.make_fairseq_task(input_args)
        return task

    def _build_ensemble(self, model_path, task):
        models, _ = checkpoint_utils.load_model_ensemble(
            model_path.split(':'), task=task
        )

        # Optimize ensemble for generation
        for model in models:
            model.make_generation_fast_(
                beamable_mm_beam_size=1, need_attn=False
            )
        ensemble_model = EnsembleModel(models)
        ensemble_model.eval()
        return ensemble_model

    def initialize_state(self, input_features: Dict[str, Any]) -> Dict[
        str, Any]:

        # init predictor state with "consumed" sequence containing only BOS
        state = {"consumed": [GO_ID or EOS_ID]}

        # add incremental states (is jit necessary? just copying from SGNMT)
        state["incremental_states"] = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )

        # tensorize src
        src_sentence = input_features["input_ids"]
        src_tokens = torch.LongTensor(
            [oov_to_unk(src_sentence + [EOS_ID], self.src_vocab_size)],
            device=self.device
        )
        src_lengths = torch.LongTensor(
            [len(src_sentence) + 1], device=self.device
        )

        # encode src and add to state
        encoder_input = {"src_tokens": src_tokens, "src_lengths": src_lengths}
        with torch.no_grad():
            encoder_outs = self.model.forward_encoder(encoder_input)
        state["encoder_outs"] = encoder_outs

        import pdb;
        pdb.set_trace()
        return state

    def update_single_state(self, state: Dict[str, Any],
                            prediction: Prediction) -> None:
        # so this should be similar to consume, right?
        state["consumed"].append(prediction.token_id)

    def predict_next_single(self, state: Dict[str, Any]):
        # assuming this is analogous to predict_next in sgnmt
        with torch.no_grad():
            consumed = torch.tensor(
                state["consumed"], dtype=torch.long, device=self.device
            )
            lprobs, _ = self.model.forward_decoder(
                consumed,
                state["encoder_outs"],
                state["incremental_states"]
            )
            lprobs[0, self.pad_id] = NEG_INF
            return lprobs[0].cpu().numpy()
