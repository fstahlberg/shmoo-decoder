"""
DFS, B(readth)FS, B(est)FS
"""

from typing import Any, Dict, Sequence
import heapq

from shmoo.core.interface import Decoder, Hypothesis
from shmoo.decoders import register_decoder


class PriorityQueue:
    # why? Because I don't like the heapq interface
    def __init__(self, maxheap=True):
        self.maxheap = maxheap
        self.heap = []

    def peek(self):
        score, data = self.heap[0]
        if self.maxheap:
            score = -score
        return score, data

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

    def stop_early(self, open_set, best_complete):
        """
        Stop search early because the current node has a worse score than the
        best found so far. This is possible only BestFS
        """
        return False

    def process(
            self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        initial_hypo = self.make_initial_hypothesis(input_features)

        open_set = self.build_open_set()
        open_set.append((0.0, initial_hypo))
        finished_hypos = []
        best_complete = float('-inf')
        while open_set:
            if self.stop_early(open_set, best_complete):
                break

            # pop from the open set
            curr_priority, hypo = open_set.pop()

            if not isinstance(hypo, Hypothesis):
                hypo = self.make_hypothesis(hypo)

            if self.is_finished(hypo):
                finished_hypos.append(hypo)
                best_complete = max(best_complete, hypo.score)
                # covered_mass += math.exp(hypo.score)
                continue

            # todo: don't hard-code this nbest
            predictions = self.get_predictions([hypo], nbest=87)

            # DFS: you want scores in ascending order, except for EOS which
            # you put last so it will always be popped first
            # BestFS: the order doesn't matter
            for pred in reversed(predictions):
                # this is very not-optimized
                if pred.score < best_complete:
                    continue
                priority = pred.score  # different for ad-hoc completion
                open_set.append((priority, pred))

        return self.make_final_output_features(input_features, finished_hypos)


@register_decoder("BestFirstDecoder")
class BestFirstDecoder(_TreeSearchDecoder):
    def build_open_set(self):
        return PriorityQueue()

    def stop_early(self, open_set, best_complete):
        curr_hyp = open_set.peek()[1]
        return curr_hyp.score < best_complete


@register_decoder("DepthFirstDecoder")
class DepthFirstDecoder(_TreeSearchDecoder):
    def build_open_set(self):
        return []
