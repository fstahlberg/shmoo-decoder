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

from shmoo.core.interface import Decoder
from shmoo.core.interface import Processor
from shmoo.core import utils
from shmoo import decoders
from shmoo import predictors
from shmoo import prepostprocessing

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

    def set_decoder(self, decoder: Decoder) -> None:
        self._decoder = decoder

    def add_preprocessor(self, preprocessor: Processor) -> None:
        self._preprocessors.append(preprocessor)

    def add_postprocessor(self, postprocessor: Processor) -> None:
        self._postprocessors.append(postprocessor)

    def set_up(self, config: Dict[str, Any]) -> None:
        """Set up the Shmoo API with a config dictionary."""
        decoder = decoders.setup_decoder(
            utils.get_from_config(config, "decoder"), config)

        # TODO: Extend to support multiple predictors
        decoder.add_predictor(
            predictors.setup_predictor(
                utils.get_from_config(config, "framework"), config)
        )

        self.set_decoder(decoder)

        for preprocessor in utils.get_from_config(config, "preprocessors"):
            # preprocessor is a str if no parameters are specified
            self.add_preprocessor(
                prepostprocessing.setup_processor(
                    preprocessor if type(preprocessor) == str else
                    list(preprocessor.keys())[0],
                    config)
            )

        for postprocessor in utils.get_from_config(config, "postprocessors"):
            # postprocessor is a str if no parameters are specified
            self.add_postprocessor(
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
