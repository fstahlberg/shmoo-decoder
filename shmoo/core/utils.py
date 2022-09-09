"""Utility functions used throughout the Shmoo framework."""

from typing import Any, Dict, Optional, Tuple
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


def get_from_decoder_config(config, key: str, default: Optional[Any] = None):
    return get_from_config(
        config, key, subsection='decoder_config', default=default)


def get_from_config(config, key: str, subsection: Optional[str] = None,
                    default: Optional[Any] = None) -> Any:
    if subsection is not None:
        try:
            return config[subsection][key]
        except KeyError:
            if default is None:
                logging.fatal(
                    "Did not find required key '%s -> %s' in the config.",
                    subsection, key)
                raise ValueError("Required key not found in config.")
            logging.info(
                "Did not find key '%s -> %s' in the config. Using default: %r",
                subsection, key, default)
            return default
    else:
        try:
            return config[key]
        except KeyError:
            if default is None:
                logging.fatal(
                    "Did not find required key '%s' in the config.", key)
                raise ValueError("Required key not found in config.")
            logging.info(
                "Did not find key '%s' in the config. Using default value %r",
                key, default)
            return default


# Cache for fairseq tasks (see make_fairseq_task()).
_FAIRSEQ_TASK_CACHE = {}


def make_fairseq_task(fairseq_config: Dict[str, Any]) -> Tuple[Any, Any]:
    """Creates a Fairseq task for accessing tokenizers and models.

    Args:
        fairseq_config: Fairseq config dictionary.

    Returns:
        A tuple (fairseq_task, args).
    """
    global _FAIRSEQ_TASK_CACHE
    config_cache_key = str(fairseq_config)
    if config_cache_key in _FAIRSEQ_TASK_CACHE:
        return _FAIRSEQ_TASK_CACHE[config_cache_key]

    model_path = f"{fairseq_config['model_dir']}/model.pt"
    input_args = [fairseq_config['model_dir'],
                  "--path", model_path]
    if "src_lang" in fairseq_config:
        input_args.extend(['--source-lang', fairseq_config["src_lang"]])
    if "trg_lang" in fairseq_config:
        input_args.extend(["--target-lang", fairseq_config["trg_lang"]])
    if "tokenizer" in fairseq_config:
        input_args.extend(['--tokenizer', fairseq_config["tokenizer"]])
    if "bpe" in fairseq_config:
        input_args.extend(
            ['--bpe', fairseq_config['bpe'],
             '--bpe-codes', f"{fairseq_config['model_dir']}/bpecodes"])
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser, input_args)
    task = tasks.setup_task(args)

    _FAIRSEQ_TASK_CACHE[config_cache_key] = (task, args)
    return task, args


def get_last_item(ordered_dict: OrderedDict) -> ...:
    """Gets the last key-value pair added to a OrderedDict."""
    return list(ordered_dict.items())[-1]
