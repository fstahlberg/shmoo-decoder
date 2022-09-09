from typing import Dict, Any, List
from collections import defaultdict

from sacrebleu import BLEU, CHRF
import numpy as np

from shmoo import metrics
from shmoo.core import utils
from shmoo.core.interface import Postprocessor
from shmoo.prepostprocessing import register_processor

DEFAULT_METRIC = "chrf"


@register_processor("MBRPostprocessor")
class MBRPostprocessor(Postprocessor):

    def __init__(self, config):
        super().__init__(config)
        self.metric = metrics.setup_metric(utils.get_from_decoder_config(config, 'MBR metric', DEFAULT_METRIC), config)

    def deduplicate_hypotheses(self, all_features: Dict[str, Any]) -> Dict[str, List[int]]:

        hypotheses_dict = defaultdict(list)
        for i, sample in enumerate(all_features):
            hypo = utils.get_last_item(sample['output'])[1]
            assert type(hypo) == str  # Needs to be applied on a str level
            hypotheses_dict[hypo].append(i)
        return hypotheses_dict

    def score_hypotheses(self, hypos: List[str]) -> np.ndarray:

        num_unique_hypos = len(hypos)
        res = np.zeros([num_unique_hypos, num_unique_hypos])
        for i, hypo in enumerate(hypos):
            for j, reference in enumerate(hypos):
                if i != j:
                    res[i, j] = self.metric.score(hypothesis=hypo, reference=reference)
        return np.mean(res, axis=1)


    def map_back_to_hypotheses(self, unique_hypos: List[str], unique_scores: List[float],
                               hypotheses_dict: Dict[str, List[int]]) -> List[float]:

        scores = np.empty(sum(map(len, list(hypotheses_dict.values()))))
        for i, hypo in enumerate(unique_hypos):
            for hyp_id in hypotheses_dict[hypo]:
                scores[hyp_id] = unique_scores[i]
        return scores

    def process_all(self, all_features: Dict[str, Any]) -> None:

        # Deduplicate hypotheses
        hypotheses_dict = self.deduplicate_hypotheses(all_features=all_features)
        unique_hypos = list(hypotheses_dict.keys())
        unique_scores = self.score_hypotheses(hypos=unique_hypos)

        # Map back to original hypotheses
        if len(unique_hypos) != len(all_features):
            scores = self.map_back_to_hypotheses(unique_hypos=unique_hypos,
                                                 unique_scores=unique_scores,
                                                 hypotheses_dict=hypotheses_dict)
        else:
            scores = unique_scores

        # Modify the outputs
        for i, _ in enumerate(all_features):
            all_features[i]['mbr_score'] = scores[i]

        # Sort according to the MBR score
        all_features.sort(key=lambda h: -h["mbr_score"])

