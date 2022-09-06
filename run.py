from absl import app
from absl import flags

from shmoo.core import api

FLAGS = flags.FLAGS
flags.DEFINE_string("config_path", None, "Your name.")
flags.DEFINE_integer("bla", 1, "blub")

flags.mark_flag_as_required("config_path")


def main(argv):
    del argv  # Unused.

    shmoo_decoder = api.Shmoo()
    shmoo_decoder.set_up(FLAGS.config_path)
    output_features = shmoo_decoder.decode_raw("11 22 33 44")
    print(output_features)


if __name__ == "__main__":
    app.run(main)
