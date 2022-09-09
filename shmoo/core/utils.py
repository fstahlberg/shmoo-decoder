from absl import logging

try:
    # Requires fairseq
    from fairseq import options
    from fairseq import tasks
    from fairseq import utils as fairseq_utils
except ImportError:
    logging.info("Fairseq not available.")
else:
    logging.info("Fairseq imports successful.")

# Set to true by _initialize_fairseq() after first call.
FAIRSEQ_INITIALIZED = False

DEFAULT_EOS_ID = 2
DEFAULT_BEAM_SIZE = 4

# Sampling hyperparameters
DEFAULT_SAMPLING_STRATEGY = "temperature"
DEFAULT_NUM_SAMPLES = 5
DEFAULT_SEED = 1
DEFAULT_TEMPERATURE = 1
DEFAULT_TOP_K = 10
DEFAULT_NUCLEUS_P = 0.8

def _initialize_fairseq(user_dir):
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


def make_fairseq_task(input_args):
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser, input_args)
    return tasks.setup_task(args), args
