from absl import app
from absl import flags
from ruamel.yaml import YAML
from pathlib import Path

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
    all_output_features = shmoo_decoder.decode_raw(
        "Why is it rare to discover new marine mammal species?")
    for index, output_features in enumerate(all_output_features):
        print("\n%d. BEST OUTPUT" % (index + 1,))
        for key, val in sorted(output_features.items()):
            print("%s: %s" % (key, val))


if __name__ == "__main__":
    app.run(main)
