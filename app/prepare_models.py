import importlib
import logging

from config import Configuration

conf = Configuration()


def prepare_models():
    """Pre-downloads the models specified in the configuration object."""
    for model_name in conf.models:
        try:
            module = importlib.import_module("torchvision.models")
            # download model
            _ = module.__getattribute__(model_name)(weights="DEFAULT")
            del _  # free up memory
        except ImportError:
            logging.error("Model {} not found".format(model_name))


if __name__ == "__main__":
    prepare_models()
