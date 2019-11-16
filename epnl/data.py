import logging, logging.config

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('epnl')

import pandas as pd
import numpy as np


class DataLoader:
    """
    Load a dataset
    """

    def __init__(self, path):
        self._path = path

    def num_training(self):
        return 0

    def num_testing(self):
        return 0

    def num_validation(self):
        return 0


class MetaphorDataLoader(DataLoader):
    """
    Load the metaphor task dataset.
    """

    pass


class MetonymyDataLoader(DataLoader):
    """
    Load the metonymy task dataset.
    """

    pass
