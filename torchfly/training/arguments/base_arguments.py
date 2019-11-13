import argparse

class BaseArguments:
    def __init__(self, notebook=False):
        self._notebook = notebook
        self._parser = argparse.ArgumentParser(description="")

        # Evaluation arguments
        self._parser.add_argument(
            "--do_valid",
            action='store_true',
            help="Whether to do evaluation on the validation set"
        )
        self._parser.add_argument(
            "--do_test",
            action='store_true',
            help="Whether to do evaluation on the test set"
        )

        # Training hyperparameters
        self._parser.add_argument(
            "--learning_rate",
            default=5e-5,
            type=float,
            help="The initial learning rate for AdamW."
        )
        self.add_argument(
            "--batch_size",
            default=1,
            type=int,
            help="Set the batch size for training"
        )
        self._parser.add_argument(
            "--num_train_epochs",
            default=10,
            type=int,
            help="Total number of training epochs to perform."
        )
        self._parser.add_argument(
            "--max_grad_norm",
            default=-1.0,
            type=float,
            help="Max gradient norm for gradient clipping."
        )

        ## optimization settings
        self._parser.add_argument(
            "--warmup_steps",
            default=-1,
            type=int,
            help="Total number of training epochs to perform."
        )
        self._parser.add_argument(
            "--warmup_ratio",
            default=0.1,
            type=float,
            help="Ratio of warmup steps in terms of the training set"
        )
        self._parser.add_argument(
            '--gradient_accumulation_steps',
            type=int,
            default=1,
            help=
            "Number of updates steps to accumulate before performing a backward/update pass."
        )

        # logging settings
        self._parser.add_argument(
            '--logging_steps',
            type=int,
            default=50,
            help="Log every X updates steps."
        )
        self._parser.add_argument(
            '--save_steps',
            type=int,
            default=50,
            help="Save checkpoint every X updates steps."
        )

        # Checkpoint setting
        self._parser.add_argument(
            '--checkpoint_dir',
            type=str,
            default="Cehckpoint",
            help="Set checkpoint directory."
        )

        ## FP 16 training
        self._parser.add_argument(
            '--fp16',
            action='store_true',
            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex)"
        )
        self._parser.add_argument(
            '--fp16_opt_level',
            type=str,
            default='O1',
            help=
            "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        )

        ## Distributed training
        self._parser.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help="For distributed training: local_rank"
        )

    def add_argument(self, **kwargs):
        self._parser.add_argument(**kwargs)

    def parse_args(self):
        if self._notebook:
            import sys; sys.argv=['']; del sys
        return self._parser.parse_args()