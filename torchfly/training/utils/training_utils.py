from abc import ABC, abstractmethod
import argparse


class BaseArgParser(ABC):
    """
    An Abstract Class for parsing arguments
    """
    @abstractmethod
    def parse_args(self):
        pass

    @abstractmethod
    def add_args(self):
        pass


class CommonArgParser(BaseArgParser):
    """ 
    Common Argument Parser
    """
    def __init__(
        self, description="Common argumet parser for deep learning training"
    ):
        self.parser = argparse.ArgumentParser(description=description)
        self.init_base_args()
        self.parse_args = self.parser.parse_args
        self.add_argument = self.parser.add_argument

    def init_base_args(self):
        """
        Parse the command line arguments
        """
        # add all arguments
        self.parser.add_argument(
            "--learning_rate",
            default=1e-4,
            type=float,
            help="The initial learning rate."
        )
        self.parser.add_argument(
            "--batch_size", default=1, type=int, help="Set the batch size"
        )
        self.parser.add_argument(
            "--num_train_epochs",
            default=3,
            type=int,
            help="Total number of training epochs to perform."
        )
        self.parser.add_argument(
            "--max_grad_norm",
            default=5.0,
            type=float,
            help="Max gradient norm."
        )
        # warmup settings
        self.parser.add_argument(
            "--warmup_steps",
            default=-1,
            type=int,
            help="Total number of training epochs to perform."
        )
        self.parser.add_argument(
            "--warmup_ratio",
            default=0.1,
            type=float,
            help="Ratio of warmup steps in terms of the training set"
        )
        self.parser.add_argument(
            '--gradient_accumulation_steps',
            type=int,
            default=1,
            help=
            "Number of updates steps to accumulate before performing a backward/update pass."
        )
        # logging
        self.parser.add_argument(
            '--logging_steps',
            type=int,
            default=50,
            help="Log every X updates steps."
        )
        self.parser.add_argument(
            '--save_steps',
            type=int,
            default=50,
            help="Save checkpoint every X updates steps."
        )
        self.parser.add_argument(
            '--checkpoint_dir',
            type=str,
            default="Checkpoint",
            help="Set checkpoint directory."
        )
        # fp 16 training
        self.parser.add_argument(
            '--fp16',
            action='store_true',
            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex)"
        )
        self.parser.add_argument(
            '--fp16_opt_level',
            type=str,
            default='O1',
            help=
            "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        )
        # distributed training
        self.parser.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help="For distributed training: local_rank"
        )