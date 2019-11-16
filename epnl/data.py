import logging, logging.config

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('epnl')

import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class MetaphorDataSet(Dataset):
    """
    Load the metaphor task dataset.
    """

    pass


class MetonymyDataSet(Dataset):
    """
    Load the metonymy task dataset.
    """

    pass

class TestDataSet(Dataset):
    """
    Load a test dataset to ensure functionality
    """

    def __init__(self, path):

        self.sequence = "sentence"
        self.span1L = "key_ix"
        self.span1R = None
        self.span2L = None
        self.span2R = None
        self.target = "metaphor"

        self._data = pd.read_csv(path)
        logger.debug(f"Data loaded for TestDataSet of {self._data.shape} shape")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        select = self._data.iloc[item]
        sample = {
            "sequence": select[self.sequence]
            # "span1L": select[self.span1L],
            # "span1R": select[self.span1R] if self.span1R else None,
            # "span2L": select[self.span2L] if self.span2L else None,
            # "span2R": select[self.span2R] if self.span2R else None,
            # "target": select[self.target]
        }
        return sample
