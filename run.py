#!/usr/bin/env -S PYTHONPATH=. python3.9

import os
import sys
import warnings
import signal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

from transformers.utils import logging

logging.set_verbosity_error()

from experiments.src.train import train
from experiments.configs.table_1 import config_list as table1_config
from experiments.configs.datasets import *
from experiments.configs.models import *
from tasks import *

exp_configs = table1_config


if __name__ == "__main__":

    #? Cleaner Interrupt of the execution. Takes some seconds to stop,
    #? but it cleans the memory and should not cause core dumps.
    def signal_handler(sig, frame):    
        print('Cancelling Training!')
        sys.exit(0)

    # Registers handler for SIGINT
    signal.signal(signal.SIGINT, signal_handler)


    for config in exp_configs:          
        print("Running Experiment with config:")
        print(config)

        train(
            n_log_steps=10,
            lr_decay_interval=1000,
            **config,
        )
