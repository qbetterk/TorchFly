from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseTrainer(ABC):
    """
    An abstract class for trainer
    """
    @abstractmethod
    def train_one_step(self, batch):
        pass

    @abstractmethod
    def eval_one_step(self, batch):
        pass

    @abstractmethod
    def train_one_epoch(self, dataloader):
        pass

    @abstractmethod
    def eval_one_epoch(self, dataloader):
        pass


class CommonTrainer(BaseTrainer):
    def __init__(self, args, model:nn.Module):
        pass

    def train_one_step(self, batch)