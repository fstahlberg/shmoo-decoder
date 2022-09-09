"""
DFS, B(readth)FS, B(est)FS
"""

from typing import Any, Dict, Sequence
import heapq

from shmoo.core.interface import Decoder
from shmoo.decoders import register_decoder


class PriorityQueue:
    # why? Because I don't like the heapq interface
    def __init__(self, maxheap=True):
        self.maxheap = maxheap
        self.heap = []

    def pop(self):
        score, data = heapq.heappop(self.heap)
        if self.maxheap:
            score = -score
        return score, data

    def append(self, item):
        score, data = item
        if self.maxheap:
            score = -score
        heapq.heappush(self.heap, (score, data))

    def __len__(self):
        return len(self.heap)

    def __bool__(self):
        return bool(self.heap)


class _TreeSearchDecoder(Decoder):
    """
    An algorithm that generalizes depth-first, breadth-first, and best-first
    search.
    """

    def __init__(self, config):
        super().__init__(config)

    def build_open_set(self):
        raise NotImplementedError("Need to use a concrete implementation")

    def process(
            self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        initial_hypo = self.make_initial_hypothesis(input_features)

        open_set = self.build_open_set()
        open_set.append((0.0, initial_hypo))
        finished_hypos = []
        while open_set:
            # pop from the open set
            curr_score, hypo = open_set.pop()

            if self.is_finished(hypo):
                finished_hypos.append(hypo)
                # todo: real stopping conditions
                # (currently it terminates on the first finished hypothesis)
                break

            # todo: don't hard-code this nbest, and consider thresholds for
            # these things (as in the cat-got-your-tongue paper)
            # (also, it may be worthwhile to put the eos prediction at the
            # front)
            predictions = self.get_predictions([hypo], nbest=10)
            predictions.sort(key=lambda h: h.score, reverse=True)
            for pred in predictions:
                open_set.append((pred.score, self.make_hypothesis(pred)))

        return self.make_final_output_features(input_features, finished_hypos)


@register_decoder("BestFirstDecoder")
class BestFirstDecoder(_TreeSearchDecoder):
    def build_open_set(self):
        return PriorityQueue()


@register_decoder("DepthFirstDecoder")
class DepthFirstDecoder(_TreeSearchDecoder):
    def build_open_set(self):
        return []
