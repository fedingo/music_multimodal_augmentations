from experiments.configs.models import *
from experiments.configs.datasets import *
import itertools


config_list = [
    {
        **model_config,
        **dataset_config,
        "n_runs": 3,
    } for model_config, dataset_config in itertools.product(models_list, datasets_list)
]