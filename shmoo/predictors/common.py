from typing import Any, Dict

import numpy as np


from shmoo.core.interface import Predictor


class TokenBoostPredictor(Predictor):

    def predict_next_single(self, state: Dict[str, Any]):
        pass
