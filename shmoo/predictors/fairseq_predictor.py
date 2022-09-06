"""
So, the idea is to sketch a predictor class for fairseq. This should implement
the MT model API and abstract away the internal details of fairseq, so that
the search algorithm doesn't need to know anything about it.

Starting this from code copied from fairseq.
"""

import logging
import os
from typing import Dict, List, Optional


from cam.sgnmt import utils
from cam.sgnmt.predictors.core import Predictor


# using this predictor requires torch and fairseq
try:
    # Requires fairseq
    from fairseq import checkpoint_utils, options, tasks
    from fairseq import utils as fairseq_utils
    from fairseq.sequence_generator import EnsembleModel
    import torch
    from torch import Tensor
    import numpy as np
except ImportError:
    pass  # hmm?


class FairseqPredictor:
    """Predictor for using fairseq models."""

    def __init__(self):
        """
        Initializes a fairseq predictor.
        
        No idea what goes here...maybe a path to a yaml config file? We will
        probably need things like that to recover
    
        """
        # _initialize_fairseq(user_dir)

        self.use_cuda = False

        # a bunch of stuff that parses the fairseq task -- not sure how this
        # will work, it's all interface stuff
        parser = options.get_generation_parser()
        input_args = ["--path", model_path, os.path.dirname(model_path)]
        if lang_pair:
            src, trg = lang_pair.split("-")
            input_args.extend(["--source-lang", src, "--target-lang", trg])
        args = options.parse_args_and_arch(parser, input_args)

        # Set up task, e.g. translation
        task = tasks.setup_task(args)
        self.src_vocab_size = len(task.source_dictionary)
        self.trg_vocab_size = len(task.target_dictionary)
        self.pad_id = task.source_dictionary.pad()

        # Load ensemble
        logging.info('Loading fairseq model(s) from {}'.format(model_path))
        self.models, _ = checkpoint_utils.load_model_ensemble(
            model_path.split(':'),
            task=task,
        )

        # Optimize ensemble for generation
        # Note that fairseq likes to treat all models as ensembles at
        # generation time
        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=1,
                need_attn=False,
            )
            if self.use_cuda:
                model.cuda()
        self.model = EnsembleModel(self.models)
        self.model.eval()

    def get_unk_probability(self, posterior):
        """Fetch posterior[utils.UNK_ID]"""
        # what is this for?
        return utils.common_get(posterior, utils.UNK_ID, utils.NEG_INF)

    def predict_next(self, **pred_args):
        """
        Compute p(\cdot | x, y_{<i}) 

        """
        with torch.no_grad():
            consumed = torch.tensor([self.consumed], dtype=torch.long, device="cuda" if self.use_cuda else "cpu")
            # note that this does not currently support decoding with temperature
            lprobs, _ = self.model.forward_decoder(
                consumed,
                self.encoder_outs,
                self.incremental_states,
                alpha=self.alpha
            )
            lprobs[0, self.pad_id] = utils.NEG_INF
            return lprobs[0].cpu().numpy()

    def initialize(self, src_sentences):
        """
        Big question here: should tensorizing happen here, or somewhere else?
        
        Reset predictor state, run encode sequence
        
        I do not like initializing attributes outside of __init__ (some linters
        don't like this either).
        """

        # what should this except, a tensor or a list of list of tokens?
        # damn, so many choices

        # reset state
        self.batch_size = len(src_sentences)  # treating it like a list of lists for now

        # actually, it should start with the BOS ID I think
        self.consumed = [[] for b in range(self.batch_size)]
        # self.consumed = [utils.GO_ID or utils.EOS_ID]

        # tensorize source sequence
        src_batch, src_lengths = self._tensorize(src_sentences)

        # run encoder over the batch, saving it as part of the predictor state
        with torch.no_grad():
            self.encoder_outs = self.model.forward_encoder(
                {'src_tokens': src_batch, 'src_lengths': src_lengths}
            )

        # Reset incremental states
        # do we want this jit?
        self.incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )

    def _tensorize(self, src_sentences):
        """
        src_tokens = torch.LongTensor([
            utils.oov_to_unk(src_sentence + [utils.EOS_ID],
                             self.src_vocab_size)])
        src_lengths = torch.LongTensor([len(src_sentence) + 1])

        # cuda boilerplate?
        if self.use_cuda:
            src_battch = src_batch.cuda()
            src_lengths = src_lengths.cuda()
        """
        pass

    def consume(self, word):
        """Append ``word`` to the current history."""
        self.consumed.append(word)

    def get_state(self):
        """The predictor state is the complete history."""
        return self.consumed, [state for state in self.incremental_states]

    def set_state(self, state):
        """The predictor state is the complete history."""
        self.consumed, inc_states = state
        self.incremental_states = inc_states

    def is_equal(self, state1, state2):
        """Returns true if the history is the same """
        return state1[0] == state2[0]
