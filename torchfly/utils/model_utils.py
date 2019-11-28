import torch
import torch.nn as nn
import logging

from .file_utils import gdrive_download

logger = logging.getLogger(__name__)

def download_gdrive_id(model_name, file_id):
    url = "https://drive.google.com/uc?id=" + file_id
    filepath = gdrive_download(url, "models", f"{model_name}.pth")
    states_dict = torch.load(filepath)
    return states_dict

supported_models_gdrive_map = {
                                "unified-gpt2-distill": "1fb9fK23Cy-Z4nLYBN-UahzKU1Ei3ugJi",
                                "unified-gpt2-small": "1C5uuC2RNMwIjLC5UInmoEVXbX-U1OEvF",
                                "unified-gpt2-medium": "13egcYsbJexXJbdqByvQveUieVLsAcpI8",
                                "unified-gpt2-medium-fp16": "15fNP9CglJ_3IqGNWNziyFJ7xVql7tdTE",
                                "unified-gpt2-large": "1DjKEdyQe5IUf0I1muCResBAUnX92UsYX",
                                "unified-gpt2-large-fp16": "1KiAvm0vcDwNNQIqUPiGtlU39UUgHPmwN",
                                "unified-gpt2-xl": "1UVqVlzuONX3hKCwF6AH0yJ_-2ghR39AH",
                                "unified-gpt2-xl-fp16": "1DGRaNKRuWQ4tGGSosMAyFohlRNCQdsBX",
                                "roberta-base": "1Ai4W4VluXEMI_CuSm55KaCqJyt7Lod4Q",
                                "roberta-large": "1twVERcE-qzgKrq9VEobBserzQfvr_NeF",
                                "roberta-large-fp16": "1j2vH8QVZ1fWkOLdoAf2YjsqN2xJ003uj",
                                }

def get_pretrained_states(modelname=None, url=None, gdrive=True):
    # TODO add different inputs
    logger.info(f"Loading {modelname} weights >>")

    if modelname in supported_models_gdrive_map.keys():
        states_dict = download_gdrive_id(modelname, supported_models_gdrive_map[modelname])
        return states_dict
    else:
        raise NotImplementedError
