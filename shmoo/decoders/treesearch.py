"""
DFS, B(readth)FS, B(est)FS
"""

from typing import Any, Dict, Sequence
import heapq

import numpy as np

from shmoo.core.interface import Decoder, Hypothesis, Prediction
from shmoo.core import utils
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

        self.nbest_predictions = utils.get_from_decoder_config(
            config, 'nbest_predictions', 5)

    def build_open_set(self):
        raise NotImplementedError("Need to use a concrete implementation")

    def stop_early(self, open_set, best_complete):
        """
        Stop search early because the current node has a worse score than the
        best found so far. This is possible only BestFS
        """
        return False

    def order_predictions(self):
        """
        Sort the predictions in the order in which they should be appended to
        the open set.
        """
        pass

    def get_predictions(
            self,
            hypos: Sequence[Hypothesis],
            nbest: int = 0,
            score_threshold: float = float("-inf")) -> Sequence[Prediction]:
        # score threshold should be a batch-sized array, not a single float.

        if nbest > 0:
            return super().get_predictions(hypos, nbest)

        unfinished_hypos = [hypo for hypo in hypos if
                            not self.is_finished(hypo)]

        pos_scores = self.get_position_scores(hypos=unfinished_hypos)
        base_scores = [hypo.score for hypo in unfinished_hypos]
        accumulated_scores = pos_scores + np.expand_dims(base_scores, 1)
        # if not using nbest, it should not be necessary to call argpartition

        predictions = []
        # todo: actual score threshold
        indices = np.nonzero(accumulated_scores > score_threshold)
        for hypo_index, token_id in zip(indices[0], indices[1]):
            predictions.append(
                Prediction(
                    token_id=token_id,
                    score=accumulated_scores[hypo_index, token_id],
                    parent_hypothesis=unfinished_hypos[hypo_index]))
        predictions = self.complete_finished_hypothesis(
            hypos=hypos, predictions=predictions)

        # subclass decides whether/how to sort
        self.order_predictions(predictions)
        return predictions

    def process(
            self, input_features: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        initial_hypo = self.make_initial_hypothesis(input_features)

        open_set = self.build_open_set()
        open_set.append((0.0, initial_hypo))
        finished_hypos = []
        best_complete = float('-inf')
        pred_calls = 0
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

            predictions = self.get_predictions(
                [hypo],
                nbest=self.nbest_predictions,
                score_threshold=best_complete
            )
            pred_calls += 1

            for pred in predictions:
                priority = pred.score  # different for ad-hoc completion
                open_set.append((priority, pred))
        print(pred_calls)

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

    def order_predictions(self, predictions):
        """
        Scores should be in ascending order except for the eos, which should
        occur last so that it is always at the top of the stack.
        """
        predictions.sort(key=lambda p: float("inf")
                         if p.token_id == self.eos_id else p.score)
