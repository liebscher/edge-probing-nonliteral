import logging, logging.config

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('epnl')

import json

import pandas as pd
import numpy as np

import torch as tt
import torch.nn as nn

from torch.utils.data import Dataset


def match_tokens(ind, spanL, spanR):
    """
    Get the relevant span bounds given the original span bounds

    Parameters
    ----------
    ind : list

    spanL : int
        Left inclusive beginning of the span
    spanR : int
        Right exclusive end of the span

    Returns
    -------

    """

    # ((0, 'he'), (1, 'couldn'), (1, "'"), (1, 't'), (2, 'have'), (3, 'fought')
    # 1:3

    enc_spanL = ind.index(spanL)
    enc_spanR = ind.index(spanR) if spanR <= max(ind) else len(ind) - 1

    return enc_spanL, enc_spanR


# class MetaphorDataSet(Dataset):
#     """
#     Load the metaphor task dataset.
#     """
#
#     pass
#
#
# class MetonymyDataSet(Dataset):
#     """
#     Load the metonymy task dataset.
#     """
#
#     pass


class DPRDataSet(Dataset):
    """
    Use the DPR datasets
    """

    def __init__(self, path, embedder, padding=64):
        """

        Parameters
        ----------
        path : string
            Path to a data file
        embedder : epnl.Embedder
            An Embedder object to determine how tokens will be tokenizer and embedded into a latent space
        """

        self._embedder = embedder

        self._padding = padding

        self.sequence = "text"
        self.span1L = "span1"
        self.span1R = "span1"
        self.span2L = "span2"
        self.span2R = "span2"
        self.target = "label"

        self._data = []
        for line in open(path).readlines():
            self._data.append(json.loads(line))

        self._data = pd.DataFrame(self._data)

        logger.debug(f"Data loaded for DPR of {self._data.shape} shape")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item : int
            Index of the item to get from the dataset

        Returns
        -------
        sample : dict
            span1 : [torch.tensor] the pooled tokens for the first span
            target : [ndarray] the target output
        """
        select = self._data.iloc[item]
        which = 0 #np.round(np.random.rand()).astype(int) if len(select["targets"]) == 2 else 0

        ind, tokens, enc = self._embedder.tokenize(select[self.sequence])

        embedding = self._embedder.embed(enc)

        padded_embedding1 = tt.zeros((self._padding, self._embedder.get_dims()))
        padded_embedding2 = tt.zeros((self._padding, self._embedder.get_dims()))

        left1 = select["targets"][which][self.span1L][0]
        left2 = select["targets"][which][self.span2L][0]

        right1 = select["targets"][which][self.span1R][1]
        right2 = select["targets"][which][self.span2R][1]

        span1L, span1R = match_tokens(ind, left1, right1)
        span2L, span2R = match_tokens(ind, left2, right2)

        padded_embedding1[:(span1R - span1L), :] += embedding[0][span1L:span1R, :]
        padded_embedding2[:(span2R - span2L), :] += embedding[0][span2L:span2R, :]

        sample = {
            "embedding1": padded_embedding1,
            "embedding2": padded_embedding2,
            "target": 1.0 if select["targets"][which][self.target] == "entailed" else 0.0
        }
        # enc_span1 = ind.index(left1)
        # enc_span2 = ind.index(right1) if right1 <= max(ind) else len(ind) - 1
        #
        # match = tokens[enc_span1:enc_span2]

        # print(select[self.sequence], (left1, right1), (left2, right2), match, sample["target"])
        return sample


class TroFiDataSet(Dataset):
    """
    Use the TroFi datasets
    """

    def __init__(self, path, embedder, padding=64):
        """

        Parameters
        ----------
        path : string
            Path to a data file
        embedder : epnl.Embedder
            An Embedder object to determine how tokens will be tokenizer and embedded into a latent space
        """

        self._embedder = embedder

        self._padding = padding

        self.sequence = "sentence"
        self.span1L = "kw_ix"
        self.span1R = None
        self.span2L = None
        self.span2R = None
        self.target = "label"

        self._data = pd.read_csv(path)
        logger.debug(f"Data loaded for TroFi of {self._data.shape} shape")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        """

        Parameters
        ----------
        item : int
            Index of the item to get from the dataset

        Returns
        -------
        sample : dict
            span1 : [torch.tensor] the pooled tokens for the first span
            target : [ndarray] the target output
        """
        select = self._data.iloc[item]

        ind, tokens, enc = self._embedder.tokenize(select[self.sequence])

        embedding = self._embedder.embed(enc)

        padded_embedding = tt.zeros((self._padding, self._embedder.get_dims()))

        # span start gets shifted left since 0/1 indexing issue
        if self.span1R:
            right = select[self.span1R]
        else:
            right = select[self.span1L]

        assert select[self.span1L]-1 < right, f"Span start not before span end: {select[self.sequence]}"

        span1L, span1R = match_tokens(ind, select[self.span1L]-1, right)

        padded_embedding[:(span1R - span1L), :] += embedding[0][span1L:span1R, :]

        sample = {
            # "span1": span1_pooled,
            "embedding1": padded_embedding,
            "embedding2": -1.,
            # "span1L": span1L,
            # "span1R": span1R,
            # "span2L": -1.,
            # "span2R": -1,
            "target": 1.0 if select[self.target] == "N" else 0.0
        }
        return sample
