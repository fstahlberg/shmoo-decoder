import os

from shmoo.core import import_classes
from shmoo.core.interface import Processor

# register processor class
PROCESSOR_REGISTRY = {}
PROCESSOR_CLASS_NAMES = set()


def setup_processor(processor_name, config):

    if isinstance(processor_name, str):
        processor = PROCESSOR_REGISTRY[processor_name]
    else:
        return NotImplementedError

    return processor.setup_processor(config)


def register_processor(name):
    """
    New pre- and postprocessors can be added to shmoo with the
    :func:`~shmoo.processors.register_processor` function decorator.
    For example::
        @register_processor('tokenizer')
        class Tokenizer(Process):
            (...)
    .. note::
        All pre- and postprocessor must implement the :class:`~shmoo.interface.Processor`
        interface.
    Args:
        name (str): the name of the processor
    """

    def register_processor_cls(cls):
        if name in PROCESSOR_REGISTRY:
            raise ValueError("Cannot register duplicate processor ({})".format(name))
        if not issubclass(cls, Processor):
            raise ValueError(
                "Processors ({}: {}) must extend the Processor interface".format(name, cls.__name__)
            )
        if cls.__name__ in PROCESSOR_CLASS_NAMES:
            raise ValueError(
                "Cannot register processor with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        PROCESSOR_REGISTRY[name] = cls
        PROCESSOR_CLASS_NAMES.add(cls.__name__)

        return cls

    return register_processor_cls


# automatically import any Python files in the tasks/ directory
processors_dir = os.path.dirname(__file__)
import_classes(processors_dir, "shmoo.prepostprocessing")
