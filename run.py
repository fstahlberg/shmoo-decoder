from absl import app
from absl import flags

from shmoo.core import api

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("preprocessor", None, "Preprocessors.")
flags.DEFINE_string("decoder", None, "Decoder.")
flags.DEFINE_multi_string("predictor", None, "Predictor.")
flags.DEFINE_multi_string("postprocessor", None, "Postprocessors.")
flags.DEFINE_integer("bla", 1, "blub")

flags.mark_flag_as_required("decoder")


def main(argv):
    del argv  # Unused.

    shmoo_decoder = api.Shmoo()
    shmoo_decoder.set_up(
        preprocessor_specs=FLAGS.preprocessor,
        decoder_spec=FLAGS.decoder,
        predictor_specs=FLAGS.predictor,
        postprocessor_specs=FLAGS.postprocessor)
    all_output_features = shmoo_decoder.decode_raw(
        "Why is it rare to discover new marine mammal species?")
    for index, output_features in enumerate(all_output_features):
        print("\n%d. BEST OUTPUT" % (index + 1,))
        for key, val in sorted(output_features.items()):
            print("%s: %s" % (key, val))


if __name__ == "__main__":
    app.run(main)
