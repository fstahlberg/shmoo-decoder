"""Pre- and postprocessors for input and output (e.g. to the file system)."""

from typing import Any, Dict, Sequence

from shmoo.core import utils
from shmoo.core.interface import Postprocessor
from shmoo.prepostprocessing import register_processor


@register_processor("StdoutPostprocessor")
class StdoutPostprocessor(Postprocessor):
    """Prints the input and output features to stdout."""

    def __init__(self, config):
        super().__init__(config)
        self._verbose = utils.get_from_config(config, "verbose", default=True)

    def process_all(
            self, all_features: Sequence[Dict[str, Any]]) -> None:
        """Prints output features to stdout."""
        if not all_features:
            print("!!! No hypotheses found !!!")
            return

        input_features = all_features[0]["input"]
        if self._verbose:
            print("INPUT:")
            for step, (key, value) in enumerate(
                    input_features.items()):
                print("  Step %d (%s): %r" % (step + 1, key, value))
        else:
            print("INPUT: %r" % (list(input_features.items())[0][1],))
        print("")
        for rank, features in enumerate(all_features):
            if self._verbose:
                print("%d. OUTPUT (%f):" % (rank + 1, features["score"]))
                for step, (key, value) in enumerate(
                        features["output"].items()):
                    print("  Step %d (%s): %r" % (step + 1, key, value))
                additional_features = [item for item in sorted(features.items())
                                       if item[0] not in ["score", "input",
                                                          "output"]]
                if additional_features:
                    print("  ADDITIONAL FEATURES:")
                    for item in additional_features:
                        print("    %s: %r" % item)
                print("")
            else:
                print(
                    "%d. OUTPUT (%f): %r" % (
                        rank + 1, features["score"], utils.get_last_item(
                            features["output"])[1]))
