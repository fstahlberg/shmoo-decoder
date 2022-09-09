"""Main interface to Shmoo.

The `Shmoo` class is the main entry point to the Shmoo decoder. Example usage:

    shmoo_decoder = api.Shmoo()
    shmoo_decoder.set_up_with_yaml(config_path=<path-to-config-yaml>)
    shmoo.decode("<sentence to translate>")

More detailed information can be found on github:
  https://github.com/fstahlberg/shmoo-decoder
"""

from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence
from absl import logging

from shmoo import prepostprocessing, decoders, predictors

try:
    from ruamel.yaml import YAML
    from pathlib import Path
except ImportError:
    logging.info("YAML configuration not available.")
else:
    logging.info("YAML imports successful.")


class Shmoo:

    def __init__(self):
        self._preprocessors = []
        self._postprocessors = []
        self._decoder = None

    def set_up(self, config: Dict[str, Any]) -> None:
        """Set up the Shmoo API with a config dictionary."""
        self._decoder = decoders.setup_decoder(config["decoder"], config)

        # TODO: Extend to support multiple predictors
        self._decoder.add_predictor(
            predictors.setup_predictor(config["framework"], config)
        )

        for preprocessor in config["preprocessors"]:
            # preprocessor is a str if no parameters are specified
            self._preprocessors.append(
                prepostprocessing.setup_processor(
                    preprocessor if type(preprocessor) == str else
                    list(preprocessor.keys())[0],
                    config)
            )

        for postprocessor in config["postprocessors"]:
            # postprocessor is a str if no parameters are specified
            self._postprocessors.append(
                prepostprocessing.setup_processor(
                    postprocessor if type(postprocessor) == str else
                    list(postprocessor.keys())[0],
                    config)
            )

    def set_up_with_yaml(self, config_path: str) -> None:
        """Set up the Shmoo API with a YAML file."""
        yaml = YAML(typ='safe')
        config = yaml.load(Path(config_path))
        logging.info("Loaded YAML config from %s: %s", config_path, config)
        self.set_up(config)

    def decode(
            self, raw: Any, input_features: Optional[Dict[str, Any]] = None
    ) -> Sequence[Dict[str, Any]]:
        """Performs a single decoding run.

        Args:
            raw: The input to the first preprocessor.
            input_features: A dictionary with addition input features.

        Returns:
            The feature dictionaries returned by the final postprocessor.
        """
        if input_features is None:
            input_features = {}
        input_features["input"] = OrderedDict({"raw": raw})
        for preprocessor in self._preprocessors:
            preprocessor.process(input_features)
        all_output_features = self._decoder.process(input_features)
        for postprocessor in self._postprocessors:
            postprocessor.process_all(all_output_features)
        return all_output_features
