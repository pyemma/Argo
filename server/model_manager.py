from enum import Enum, auto
import os

import torch

from configs.model import OUTPUT_MODEL_PATH, DIN_SMALL_CONFIG, LARGE_TRANSACT_CONFIG
from model import (
    DeepInterestModel,
    TransAct,
)


class ModelType(Enum):
    DIN_SMALL = auto()
    TRANSACT_FULL = auto()


def factory(model_file_name, config, model_class):
    model_file_path = os.path.join(OUTPUT_MODEL_PATH, model_file_name)
    checkpoint = torch.load(model_file_path, weights_only=False)
    model = model_class(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


MODEL_TYPE_TO_FACTORY = {
    ModelType.DIN_SMALL: lambda: factory("din-movie-len-small.pth", DIN_SMALL_CONFIG, DeepInterestModel),
    ModelType.TRANSACT_FULL: lambda: factory("transact-movie-len-full.pth", LARGE_TRANSACT_CONFIG, TransAct),
}


class ModelManager:
    """
    This is a simple class to manage the loading of model file into memory for
    inference; for now we only consider the simple scenarios where the model
    is not disaggregated and could be loaded onto single machine; in the future,
    we might need to support more complicated cases such as distributed-inference
    or PP

    The model loading would run in a lazy mode to load model into memory on demand
    when the request hit the model
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # use the available accelerator
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # hold all models in memory
        # TODO: add eviction mechanism to offload model if exceed memory budget
        self.models = {}
        self._initialized = True

    def get_model(self, model_type: ModelType):
        if model_type not in self.models:
            self.models[model_type] = MODEL_TYPE_TO_FACTORY.get(model_type)().to(self.device)
        return self.models.get(model_type)


model_manager = ModelManager()
