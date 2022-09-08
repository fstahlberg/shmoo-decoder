"""This module contains all predictors supported by Shmoo."""

import os

from shmoo.core import import_classes
from shmoo.core.interface import Predictor

# register predictor class
PREDICTOR_REGISTRY = {}
PREDICTOR_CLASS_NAMES = set()


def setup_predictor(predictor_name: str, config) -> Predictor:
    """Looks up `predictor_name` and calls it's `setup_predictor(config)`."""

    if isinstance(predictor_name, str):
        predictor = PREDICTOR_REGISTRY[predictor_name]
    else:
        return NotImplementedError

    return predictor.setup_predictor(config)


def register_predictor(name: str):
    """Adds a new predictor to the registry.

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
            raise ValueError(
                "Cannot register duplicate predictor ({})".format(name))
        if not issubclass(cls, Predictor):
            raise ValueError(
                "Predictors ({}: {}) must extend the Predictor interface".format(
                    name, cls.__name__)
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


# automatically import any Python files in the predictors/ directory
predictors_dir = os.path.dirname(__file__)
import_classes(predictors_dir, "shmoo.predictors")
