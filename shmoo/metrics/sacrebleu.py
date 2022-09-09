from shmoo.core.interface import Metric
from shmoo.metrics import register_metric

from sacrebleu import BLEU, CHRF

@register_metric("bleu")
class BLEUMetric(Metric):

    def __init__(self, config):
        super().__init__(config)
        self.scorer = BLEU(effective_order=True).sentence_score

    def score(self, hypothesis: str, reference: str) -> float:
        return self.scorer(hypothesis=hypothesis, references=[reference]).score


@register_metric("chrf")
class CHRFMetric(Metric):

    def __init__(self, config):
        super().__init__(config)
        self.scorer = CHRF().sentence_score

    def score(self, hypothesis: str, reference: str) -> float:
        return self.scorer(hypothesis=hypothesis, references=[reference]).score