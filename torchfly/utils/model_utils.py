import torch
import torch.nn as nn
import logging

from .file_utils import gdrive_download

logger = logging.getLogger(__name__)

def get_pretrained_states(modelname=None, url=None, gdrive=True):
    # TODO add different inputs
    logger.info(f"Loading {modelname} weights >>")

    if modelname == "unified-gpt2-small":
        url = "https://drive.google.com/uc?id=1C5uuC2RNMwIjLC5UInmoEVXbX-U1OEvF"
        filepath = gdrive_download(url, "models", "unified-gpt2-small.pth")
        states_dict = torch.load(filepath)
        return states_dict