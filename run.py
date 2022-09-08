from absl import app
from absl import flags
import sys

from shmoo.core import api

FLAGS = flags.FLAGS

flags.DEFINE_string("config_path", None, "Path to a yaml file detailing model path, model framework, decoding method "
                                         "and potential pre- or postprocessors.")
flags.DEFINE_string("single_sentence", None, "If specified, decode this single sentence and exit.")
flags.mark_flag_as_required("config_path")


def main(argv):
    del argv  # Unused.

    shmoo_decoder = api.Shmoo()
    shmoo_decoder.set_up_with_yaml(config_path=FLAGS.config_path)

    if FLAGS.single_sentence:
        source_sentences = [FLAGS.single_sentence]
    else:
        source_sentences = sys.stdin
    # output_features = shmoo_decoder.decode_raw("Why is it rare to discover new marine mammal species?")
    for line in source_sentences:
        source_sentence = line.strip()
        all_output_features = shmoo_decoder.decode_raw(source_sentence)
        for index, output_features in enumerate(all_output_features):
            print("\n%d. BEST OUTPUT" % (index + 1,))
            for key, val in sorted(output_features.items()):
                print("%s: %s" % (key, val))


if __name__ == "__main__":
    app.run(main)
