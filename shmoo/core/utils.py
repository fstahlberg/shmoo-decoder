"""Utility functions used throughout the Shmoo framework."""

from typing import Any, Sequence, Tuple
from collections import OrderedDict

from absl import logging

try:
    from fairseq import options
    from fairseq import tasks
    from fairseq import utils as fairseq_utils
except ImportError:
    logging.info("Fairseq not available.")
else:
    logging.info("Fairseq imports successful.")

# Set to true by _initialize_fairseq() after first call.
FAIRSEQ_INITIALIZED = False

# End-of-sentence symbol used if eos_id is not set in the config.
DEFAULT_EOS_ID = 2

# Beam size used if beam_size is not set in the config.
DEFAULT_BEAM_SIZE = 4

# Sampling hyperparameters
DEFAULT_SAMPLING_STRATEGY = "nucleus"

# Default number of samples used in sampling-based decoders.
DEFAULT_NUM_SAMPLES = 5

DEFAULT_SEED = 1
DEFAULT_TEMPERATURE = 1
DEFAULT_TOP_K = 10
DEFAULT_NUCLEUS_P = 0.8
DEFAULT_TYPICAL_P = 0.8


def _initialize_fairseq(user_dir: str) -> None:
    """Sets up the fairseq library by importing a fairseq user directory.

    Args:
        user_dir: Path to the directory with fairseq modules.
    """
    global FAIRSEQ_INITIALIZED
    if not FAIRSEQ_INITIALIZED:
        logging.info("Setting up fairseq library...")
        if user_dir:
            args = type("", (), {"user_dir": user_dir})()
            fairseq_utils.import_user_module(args)
        FAIRSEQ_INITIALIZED = True


def get_from_decoder_config(config, argument, default):
    param = config.get('decoder_config', {}).get(argument, default)
    logging.info(f"{argument}: {param}")
    return param


def make_fairseq_task(input_args: Sequence[str]) -> Tuple[Any, Any]:
    """Creates a Fairseq task for accessing tokenizers and models.

    Args:
        input_args: Fairseq command line arguments.

    Returns:
        A tuple (fairseq_task, args).
    """
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser, input_args)
    return tasks.setup_task(args), args


def get_last_item(ordered_dict: OrderedDict) -> ...:
    """Gets the last key-value pair added to a OrderedDict."""
    return list(ordered_dict.items())[-1]
