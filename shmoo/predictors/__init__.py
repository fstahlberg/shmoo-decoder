import importlib
import os

from shmoo.core.interface import Predictor

# register dataclass
PREDICTOR_REGISTRY = {}
PREDICTOR_CLASS_NAMES = set()


def setup_predictor(predictor_name, config):

    if isinstance(predictor_name, str):
        predictor = PREDICTOR_REGISTRY[predictor_name]
    else:
        return NotImplementedError

    return predictor


def register_predictor(name):
    """
    New predictors can be added to shmoo with the
    :func:`~shmoo.predictors.register_predictor` function decorator.
    For example::
        @register_predictor('fairseq')
        class FairseqPredictor(Predictor):
            (...)
    .. note::
        All Predictors must implement the :class:`~shmoo.interface.Predictor`
        interface.
    Args:
        name (str): the name of the framework
    """

    def register_predictor_cls(cls):
        if name in PREDICTOR_REGISTRY:
            raise ValueError("Cannot register duplicate predictor ({})".format(name))
        if not issubclass(cls, Predictor):
            raise ValueError(
                "Predictors ({}: {}) must extend the Predictor interface".format(name, cls.__name__)
            )
        if cls.__name__ in PREDICTOR_CLASS_NAMES:
            raise ValueError(
                "Cannot register predictor with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        PREDICTOR_REGISTRY[name] = cls
        PREDICTOR_CLASS_NAMES.add(cls.__name__)

        return cls

    return register_predictor_cls


def import_predictors(predictors_dir, namespace):
    for file in os.listdir(predictors_dir):
        path = os.path.join(predictors_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            predictor_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + predictor_name)


# automatically import any Python files in the tasks/ directory
predictors_dir = os.path.dirname(__file__)
import_predictors(predictors_dir, "shmoo.predictors")
