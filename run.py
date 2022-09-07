from absl import app
from absl import flags
from ruamel.yaml import YAML
from pathlib import Path
import sys

from shmoo.core import api

FLAGS = flags.FLAGS

flags.DEFINE_string("config_path", None, "Path to a yaml file detailing model path, model framework, decoding method "
                                         "and potential pre- or postprocessors.")
flags.mark_flag_as_required("config_path")


def main(argv):
    del argv  # Unused.

    # Parse config file
    yaml = YAML(typ='safe')
    config = yaml.load(Path(FLAGS.config_path))

    print(config)

    shmoo_decoder = api.Shmoo()
    shmoo_decoder.set_up(config=config)
    # output_features = shmoo_decoder.decode_raw("Why is it rare to discover new marine mammal species?")
    for line in sys.stdin:
        source_sentence = line.strip()
        output_features = shmoo_decoder.decode_raw(source_sentence)
        print(output_features)


if __name__ == "__main__":
    app.run(main)
