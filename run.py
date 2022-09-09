"""Example script for running the Shmoo decoder on string input."""

from absl import app
from absl import flags
import sys

from shmoo.core import api
from shmoo.prepostprocessing.io import StdoutPostprocessor

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "config_path", None,
    "Path to a yaml file detailing model path, framework, decoding method "
    "and potential pre- or postprocessors.")
flags.DEFINE_string(
    "single_sentence", None,
    "If specified, decode this single sentence and exit.")
flags.DEFINE_string(
    "kaldi_ark", None,
    "Load speech features from kaldi_ark file")
flags.mark_flag_as_required("config_path")


def main(argv):
    del argv  # Unused.

    shmoo_decoder = api.Shmoo()
    shmoo_decoder.set_up_with_yaml(config_path=FLAGS.config_path)
    shmoo_decoder.add_postprocessor(StdoutPostprocessor({"verbose": True}))

    if FLAGS.kaldi_ark:
        shmoo_decoder.decode(FLAGS.kaldi_ark)
    else:
        lines = [FLAGS.single_sentence] if FLAGS.single_sentence else sys.stdin
        for line in lines:
            shmoo_decoder.decode(line.strip())


if __name__ == "__main__":
    app.run(main)
