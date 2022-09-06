from shmoo.core.interface import Preprocessor
from shmoo.core.interface import Postprocessor
from shmoo.core.interface import Decoder
from shmoo.core.interface import Predictor

from shmoo.decoders.beam import BeamDecoder
from shmoo.predictors.common import TokenBoostPredictor
from shmoo.prepostprocessing.text import TrivialTokenPreprocessor
from shmoo.prepostprocessing.text import TrivialTokenPostprocessor


class Registry:
    pass


def make_preprocessor(preprocessor_name: str,
                      preprocessor_config_path: str) -> Preprocessor:
    return TrivialTokenPreprocessor()


def make_postprocessor(postprocessor_name: str,
                       postprocessor_config_path: str) -> Postprocessor:
    return TrivialTokenPostprocessor()


def make_decoder(decoder_name: str, decoder_config_path: str) -> Decoder:
    return BeamDecoder()


#def make_predictor(predictor_name: str,
#                   predictor_config_path: str) -> Predictor:
#    return BeamDecoder()
