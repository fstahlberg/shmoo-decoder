"""

This is the interface to the fairseq library.

https://github.com/pytorch/fairseq

It is based on the fairseq predictor for SGNMT from
https://github.com/bpopeters/sgnmt/blob/master/cam/sgnmt/predictors/pytorch_fairseq.py
"""

from absl import logging
import copy
import os
from typing import Dict, List, Optional, Any

from shmoo.core.interface import Predictor
from shmoo.core.interface import Prediction
from shmoo.core import utils
from shmoo.predictors import register_predictor

try:
    # fairseq imports
    from fairseq import checkpoint_utils, options, tasks
    from fairseq import utils as fairseq_utils
    from fairseq.sequence_generator import EnsembleModel
    import torch
    from torch import Tensor
except ImportError:
    logging.info("Fairseq predictor not available.")
else:
    logging.info("Fairseq predictor imports successful.")

# here: constants and functions copied from SGNMT's utils module: they
# ultimately belong somewhere else
GO_ID = 1
"""Reserved word ID for the start-of-sentence symbol. """

EOS_ID = 2
"""Reserved word ID for the end-of-sentence symbol. """

NEG_INF = float("-inf")


@register_predictor("FairseqPredictor")
class FairseqPredictor(Predictor):

    def __init__(self):
        """
        Check https://github.com/bpopeters/sgnmt/blob/master/cam/sgnmt/predictors/pytorch_fairseq.py
        for an idea of how the model can actually be loaded
        """
        model_path = "/home/fstahlberg/work/shmoo/wmt14.en-fr.fconv-py/model.pt"
        input_args = ["--path", model_path, os.path.dirname(model_path),
                      "--source-lang", "en", "--target-lang", "fr"]

        # user_dir = ""
        # self._lang_pair = "en-fr"

        # _initialize_fairseq(user_dir)

        self.device = torch.device("cpu")
        # task = self._load_task(model_path)

        task, args = utils.make_fairseq_task(input_args)
        self.src_vocab_size = len(task.source_dictionary)

        self.model = self._build_ensemble(model_path, task)
        self.encoder_outs = None

    #
    # def _load_task(self, model_path):
    #     input_args = ["--path", model_path, os.path.dirname(model_path)]
    #
    #     if self._lang_pair:
    #         src, trg = self._lang_pair.split("-")
    #         input_args.extend(["--source-lang", src, "--target-lang", trg])
    #     input_args.extend(
    #         ["--tokenizer", "moses", "--bpe", "subword_nmt", "--bpe-codes", "/home/fstahlberg/work/shmoo/wmt14.en-fr.fconv-py/bpecodes"])
    #
    #     task, args = utils.make_fairseq_task(input_args)
    #     return task

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
        state = {"consumed": [GO_ID]}

        # add incremental states (is jit necessary? just copying from SGNMT)
        state["incremental_states"] = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )

        # tensorize src
        src_sentence = [token_id for token_id in input_features["input_ids"]]
        src_tokens = torch.LongTensor(
            [src_sentence], device=self.device)
        src_lengths = torch.LongTensor(
            [len(src_sentence)], device=self.device
        )

        # encode src and add to state
        encoder_input = {"src_tokens": src_tokens, "src_lengths": src_lengths}
        with torch.no_grad():
            encoder_outs = self.model.forward_encoder(encoder_input)
        self.encoder_outs = encoder_outs

        return state

    def update_single_state(self, state: Dict[str, Any],
                            prediction: Prediction,
                            lazy: bool = False) -> Dict[str, Any]:
        if lazy:
            new_state = state
            new_state["consumed"].append(prediction.token_id)
        else:
            new_state = {
                "consumed": state["consumed"] + [prediction.token_id]
            }
        return new_state

    def predict_next_single(self, state: Dict[str, Any]):
        with torch.no_grad():
            consumed = torch.tensor(
                [state["consumed"]], dtype=torch.long, device=self.device)
            log_probs, _ = self.model.forward_decoder(
                consumed, self.encoder_outs, state["incremental_states"])
            return log_probs[0].cpu().numpy()
