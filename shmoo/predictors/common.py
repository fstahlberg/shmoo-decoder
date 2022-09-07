from typing import Any, Dict

import numpy as np

from shmoo.core.interface import Predictor
from shmoo.predictors import register_predictor


@register_predictor("TokenBoost")
class TokenBoostPredictor(Predictor):

    def predict_next_single(self, state: Dict[str, Any]):
        pass
