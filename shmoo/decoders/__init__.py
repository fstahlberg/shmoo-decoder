"""This module contains all decoding strategies supported by Shmoo."""

import os

from shmoo.core import import_classes
from shmoo.core.interface import Decoder

# register processor class
DECODER_REGISTRY = {}
DECODER_CLASS_NAMES = set()


def setup_decoder(decoder_name: str, config) -> Decoder:
    """Looks up `decoder_name` and calls it's `setup_decoder(config)`."""

    if isinstance(decoder_name, str):
        decoder = DECODER_REGISTRY[decoder_name]
    else:
        return NotImplementedError

    return decoder.setup_decoder(config)


def register_decoder(name: str):
    """Adds a new decoder to the registry.

    New decoders can be added to Shmoo with the
    :func:`~shmoo.decoders.register_decoder` function decorator.
    For example::
        @register_decoder('BeamSearch')
        class BeamSearch(Decoder):
            (...)
    .. note::
        All decoders must implement the :class:`~shmoo.interface.Decoder`
        interface.

    Args:
        name (str): the name of the decoder
    """

    def register_decoder_cls(cls):
        if name in DECODER_REGISTRY:
            raise ValueError(
                "Cannot register duplicate decoder ({})".format(name))
        if not issubclass(cls, Decoder):
            raise ValueError(
                "Predictors ({}: {}) must extend the Decoder interface".format(
                    name, cls.__name__)
            )
        if cls.__name__ in DECODER_CLASS_NAMES:
            raise ValueError(
                "Cannot register decoder with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        DECODER_REGISTRY[name] = cls
        DECODER_CLASS_NAMES.add(cls.__name__)

        return cls

    return register_decoder_cls


# automatically import any Python files in the tasks/ directory
decoders_dir = os.path.dirname(__file__)
import_classes(decoders_dir, "shmoo.decoders")
