"""This module contains all metrics supported by Shmoo."""

import os

from shmoo.core import import_classes
from shmoo.core.interface import Metric

# register predictor class
METRIC_REGISTRY = {}
METRIC_CLASS_NAMES = set()


def setup_metric(metric_name: str, config) -> Metric:
    """Looks up `metric_name` and calls it's `setup_metric(config)`."""

    if isinstance(metric_name, str):
        metric = METRIC_REGISTRY[metric_name]
    else:
        return NotImplementedError

    return metric.setup_metric(config)


def register_metric(name: str):
    """Adds a new metric to the registry.

    New metrics can be added to shmoo with the
    :func:`~shmoo.metrics.register_metric` function decorator.
    For example::
        @register_metric('sacrebleu')
        class SacrebleuMetric(Metric):
            (...)
    .. note::
        All Metrics must implement the :class:`~shmoo.interface.Metric`
        interface.

    Args:
        name (str): the name of the metric
    """

    def register_predictor_cls(cls):
        if name in METRIC_REGISTRY:
            raise ValueError(
                "Cannot register duplicate metric ({})".format(name))
        if not issubclass(cls, Metric):
            raise ValueError(
                "Metrics ({}: {}) must extend the Metric interface".format(
                    name, cls.__name__)
            )
        if cls.__name__ in METRIC_CLASS_NAMES:
            raise ValueError(
                "Cannot register metric with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        METRIC_REGISTRY[name] = cls
        METRIC_CLASS_NAMES.add(cls.__name__)

        return cls

    return register_predictor_cls


# automatically import any Python files in the predictors/ directory
metrics_dir = os.path.dirname(__file__)
import_classes(metrics_dir, "shmoo.metrics")