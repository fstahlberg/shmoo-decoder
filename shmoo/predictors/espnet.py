from absl import logging

import argparse
import copy
from typing import Dict, Any

from shmoo.core.interface import Predictor
from shmoo.core.interface import Prediction
from shmoo.predictors import register_predictor

try:
    from yaml import full_load
    import torch
    from espnet.nets.scorers.ctc import CTCPrefixScorer
    from espnet.nets.scorers.ctc import CTCPrefixScoreTH
    from espnet2.bin.st_inference import Speech2Text as ST
    from espnet2.bin.asr_inference import Speech2Text as ASR
    from espnet2.bin.mt_inference import Text2Text as MT
    ESPNET_TASK_MAP = {"st": ST, "asr": ASR, "mt": MT}
except ImportError:
    logging.info("ESPnet predictor not available.")
else:
    logging.info("ESPnet predictor imports successful.")



@register_predictor("Espnet")
class ESPnetPredictor(Predictor):
    def __init__(self, config):
        """Initialize ESPnet model"""

        config_file = config["espnet"]["config"]
        model_path = config["espnet"]["model_path"]
        self.task = config["espnet"]["task"]

        self.inference = ESPNET_TASK_MAP[self.task](config_file, model_path)

        self.config = {}
        with open(config_file, "r") as fpr:
            self.config = full_load(fpr)

        # only for ASR: ctc weight and decoder weight
        self.weights = {}
        self.scorers = {}

        ctc_weight = config["decoder_config"]["ctc_weight"]
        self.pre_beam_score_key = "full"
        beam_size = config["decoder_config"]["beam_size"]
        self.pre_beam_size = int(
            config["decoder_config"]["pre_beam_factor"] * beam_size
        )

        # we should set it from the information from pre-processor
        self.bos_index = len(self.config["token_list"]) - 1
        self.eos_index = len(self.config["token_list"]) - 1

        if self.task == "st":
            self.model = self.inference.st_model

        elif self.task == "asr":
            self.model = self.inference.asr_model
            self.weights = {"ctc": ctc_weight, "decoder": 1.0 - ctc_weight}

            ctc = CTCPrefixScorer(ctc=self.model.ctc, eos=self.model.eos)
            decoder = self.model.decoder
            self.scorers = {"ctc": ctc, "decoder" :decoder}

        elif self.task == "mt":
            self.model = self.inference.mt_model

        else:
            raise NotImplementedError("{:s} task is not implemented yet.".format(task))

    def initialize_state(self, input_features: Dict[str, Any]) -> Dict[str, Any]:

        # init predictor state with "consumed" sequence containing only BOS
        state = {"consumed": [self.bos_index], "decoder": [None], "ctc": [None]}

        state["incremental_states"] = []

        input_feats = torch.from_numpy(input_features["input"]["ids"])
        input_feats = input_feats.unsqueeze(0)
        lengths = input_feats.new_full(
            [1], dtype=torch.long, fill_value=input_feats.size(1)
        )

        encoder_outs, _ = self.model.encode(input_feats, lengths)
        self.encoder_outs = encoder_outs[0]

        print("enc outs:", self.encoder_outs.shape)

        if self.task == "asr":
            if "ctc" in self.scorers:
                logp = self.scorers["ctc"].ctc.log_softmax(encoder_outs)
                xlen = torch.LongTensor([logp.size(1)])
                self.scorers["ctc"].impl = CTCPrefixScoreTH(logp, xlen, 0, self.eos_index)

        return state

    def update_single_state(
        self, state: Dict[str, Any], prediction: Prediction, lazy: bool = False
    ) -> Dict[str, Any]:
        if lazy:
            new_state = state
            new_state["consumed"].append(prediction.token_id)
        else:
            new_state = {
                "consumed": state["consumed"] + [prediction.token_id],
                "ctc": copy.deepcopy(state["ctc"]),
                "decoder": copy.deepcopy(state["decoder"]),
            }
        return new_state

    def predict_next_single(self, state: Dict[str, Any], scores):

        with torch.no_grad():

            consumed = torch.LongTensor([state["consumed"]])

            if self.task == "asr":

                pre_beam_scores = []

                # might contain multiple scorers i.e.,
                # ctc and a transformer auto-regressive decoder
                weighted_scores = torch.zeros([1, self.model.vocab_size])

                if "decoder" in self.scorers:
                    dec_scores, state["decoder"] = self.scorers["decoder"].batch_score(
                        consumed,
                        state["decoder"],
                        self.encoder_outs.expand(1, *self.encoder_outs.shape),
                    )
                    weighted_scores += self.weights["decoder"] * dec_scores

                # import ipdb
                # ipdb.set_trace()
                # if "ctc" in self.scorers:
                #     if self.pre_beam_score_key == "full":
                #         pre_beam_scores = weighted_scores
                #     else:
                #         pre_beam_scores = dec_scores[self.pre_beam_score_key]

                #     part_ids = torch.topk(
                #         pre_beam_scores, self.pre_beam_size, dim=-1
                #     )[1]

                #     ctc_scores, state["ctc"] = self.scorers["ctc"].batch_score_partial(
                #         consumed, part_ids, state["ctc"], self.encoder_outs
                #     )
                #     weighted_scores += self.weights["ctc"] * ctc_scores

        return scores + weighted_scores.squeeze().cpu().numpy()


def main(args):

    predictor = ESPnetPredictor(args.config_file, args.model_pth)


if __name__ == "__main__":

    parser = argparse.ArgumenParser()
    parser.add_argument("config_file")
    parser.add_argument("model_pth")
    args = parser.parser_args()
    main(args)
