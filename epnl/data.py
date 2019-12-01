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


def match_tokens(embedding, ind, spanL, spanR):
    """
    Get the relevant list of embedded tokens given the span bounds

    Parameters
    ----------
    embedding : torch.tensor ()
        The embedded sequence before extracting the span
    ind : list

    spanL : int
        Left inclusive beginning of the span
    spanR : int
        Right exclusive end of the span

    Returns
    -------
    span : torch.tensor ()
        A sequence of embedded tokens corresponding to the span
    """

    # ((0, 'he'), (1, 'couldn'), (1, "'"), (1, 't'), (2, 'have'), (3, 'fought')
    # 1:3

    enc_span1 = ind.index(spanL)
    enc_span2 = ind.index(spanR) if spanR <= max(ind) else len(ind) - 1

    return embedding[enc_span1:enc_span2, :]


class Pooler(nn.Module):
    """
    Perform pooling, potentially with a projection layer beforehand.
    We need pooling because the tokenizer may split tokens into multiple parts, so we merge those different parts
    together when necessary.
    """

    def __init__(self, project=True, inp_dim=512, out_dim=512, pooling_type="max"):
        """

        Parameters
        ----------
        project : bool
            Should a single layer of linear projection be added __ pooling together span elements
        inp_dim : int
            Number of dimensions for input spans
        out_dim : int
            Number of dimensions for the output span
        pooling_type : string
            The type of pooling to perform. Either "mean", "max", or "sum" (default)
        """
        super(Pooler, self).__init__()
        # projection to reduce token size
        self._projector = nn.Linear(inp_dim, out_dim) if project else lambda x: x
        self._pooling_type = pooling_type

        logger.debug(f"Initializing new Pooler - project: {project}, input dims: {inp_dim}, output dims: {out_dim}, " +
                     f"type: {pooling_type}")

    def forward(self, span_sequence):
        """

        Parameters
        ----------
        span_sequence : torch.tensor (S, d)
            S embedded tokens of d dimensions to be pooled

        Returns
        -------
        pool : torch.tensor (d, )
            Single tensor of pooled token spans

        """
        if self._pooling_type == "mean":
            return tt.mean(span_sequence, dim=0)
        elif self._pooling_type == "max":
            return tt.max(span_sequence, dim=0)[0]

        return tt.sum(span_sequence, dim=0)


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


class DPRDataSet(Dataset):
    """
    Use the DPR datasets
    """

    def __init__(self, path, embedder):
        """

        Parameters
        ----------
        path : string
            Path to a data file
        embedder : epnl.Embedder
            An Embedder object to determine how tokens will be tokenizer and embedded into a latent space
        """

        self._embedder = embedder

        self.sequence = "text"
        self.span1L = "span1"
        self.span1R = "span1"
        self.span2L = "span2"
        self.span2R = "span2"
        self.target = "label"

        # either create a pooling object or don't
        self._pooler = Pooler(
            True,
            self._embedder.get_dims(),
            int(self._embedder.get_dims() / 2),
            "mean"
        )

        self._data = []
        for line in open(path).readlines():
            self._data.append(json.loads(line))

        self._data = pd.DataFrame(self._data)

        logger.debug(f"Data loaded for DPR of {self._data.shape} shape")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        """
        {
            "text": "The bee landed on the flower because it had pollen .",
            "info":
                {"split": "train",
                "source": "recast-dpr"},
            "targets": [
                {
                    "span1": [7, 8],
                    "span2": [0, 2],
                    "label": "not-entailed",
                    "span1_text": "it",
                    "span2_text": "The bee"
                },
                {
                    "span1": [7, 8],
                    "span2": [4, 6],
                    "label": "entailed",
                    "span1_text": "it",
                    "span2_text": "the flower"
                }
            ]
        }


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
        which = np.round(np.random.rand()).astype(int) if len(select["targets"]) == 2 else 0

        ind, tokens, enc = self._embedder.tokenize(select[self.sequence])

        embedding = self._embedder.embed(enc)

        left1 = select["targets"][which][self.span1L][0]
        left2 = select["targets"][which][self.span2L][0]

        right1 = select["targets"][which][self.span1R][1]
        right2 = select["targets"][which][self.span2R][1]

        span1 = match_tokens(embedding[0], ind, left1, right1)
        span2 = match_tokens(embedding[0], ind, left2, right2)

        span1_pooled = self._pooler(span1)
        span2_pooled = self._pooler(span2)

        sample = {
            "span1": span1_pooled,
            "span2": span2_pooled,
            "target": 1.0 if select["targets"][which][self.target] == "entailed" else 0.0
        }
        return sample


class TroFiDataSet(Dataset):
    """
    Use the TroFi datasets
    """

    def __init__(self, path, embedder):
        """

        Parameters
        ----------
        path : string
            Path to a data file
        embedder : epnl.Embedder
            An Embedder object to determine how tokens will be tokenizer and embedded into a latent space
        """

        self._embedder = embedder

        self.sequence = "sentence"
        self.span1L = "kw_ix"
        self.span1R = None
        self.span2L = None
        self.span2R = None
        self.target = "label"

        # either create a pooling object or don't
        self._pooler = Pooler(
            True,
            self._embedder.get_dims(),
            self._embedder.get_dims(),
            "max"
        )

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

        # print(select[self.sequence])
        # print(tuple(zip(ind, tokens)))
        # print(enc)
        # print(detokens)
        # print(embedding.size())

        # span start gets shifted left since 0/1 indexing issue
        if self.span1R:
            right = select[self.span1R]
        else:
            right = select[self.span1L]

        span1 = match_tokens(embedding[0], ind, select[self.span1L]-1, right)

        span1_pooled = self._pooler(span1)

        sample = {
            "span1": span1_pooled,
            "target": 1.0 if select[self.target] == "N" else 0.0
        }
        return sample


class TestDataSet(Dataset):
    """
    Load a test dataset to ensure functionality
    """

    def __init__(self, path, embedder):
        """

        Parameters
        ----------
        path : string
            Path to a data file
        embedder : epnl.Embedder
            An Embedder object to determine how tokens will be tokenizer and embedded into a latent space
        """

        self._embedder = embedder

        self.sequence = "sentence"
        self.span1L = "key_ix"
        self.span1R = None
        self.span2L = None
        self.span2R = None
        self.target = "metaphor"

        # either create a pooling object or don't
        self._pooler = Pooler(
            True,
            self._embedder.get_dims(),
            self._embedder.get_dims(),
            "max"
        )

        self._data = pd.read_csv(path)
        logger.debug(f"Data loaded for TestDataSet of {self._data.shape} shape")

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
        # detokens = self._embedder.detokenize(enc)

        embedding = self._embedder.embed(enc)

        # print(select[self.sequence])
        # print(tuple(zip(ind, tokens)))
        # print(enc)
        # print(detokens)
        # print(embedding.size())

        # span start gets shifted left since 0/1 indexing issue
        if self.span1R:
            right = select[self.span1R]
        else:
            right = select[self.span1L]

        span1 = match_tokens(embedding[0], ind, select[self.span1L]-1, right)

        span1_pooled = self._pooler(span1)

        # print(span1.size(), span1_pooled.size())

        sample = {
            "span1": span1_pooled,
            "target": select[self.target].astype(float)
        }
        return sample
