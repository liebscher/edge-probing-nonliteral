import logging, logging.config

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('epnl')

import json

import pandas as pd
import numpy as np

import torch as tt

from torch.utils.data import Dataset

import time
from datetime import timedelta


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


def embed_sentences(embedder, sentences):
    indices = []
    tokens = []
    embeddings = []
    for sentence in sentences:
        index, token, enc = embedder.tokenize(sentence)

        embedding = embedder.embed(enc)

        indices.append(index)
        tokens.append(token)
        embeddings.append(embedding)

    return indices, tokens, embeddings


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

        logger.debug(f"Embedding {len(self._data):,} sentences for DPR")

        st = time.time()
        indices, tokens, embedding = embed_sentences(self._embedder, self._data[self.sequence])
        self._data["indices"] = pd.Series(indices)
        self._data["embeddings"] = pd.Series(embedding)

        logger.debug(f"Data loaded for DPR in {timedelta(seconds=time.time() - st)}")

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
        which = np.round(np.random.rand()).astype(int) if len(select["targets"]) == 2 else 0

        # since we don't create the datasets, we'll lowercase them now
        # ind, tokens, enc = self._embedder.tokenize(select[self.sequence].lower())

        # embedding = self._embedder.embed(enc)

        padded_embedding1 = tt.zeros((self._padding, self._embedder.get_dims()))
        padded_embedding2 = tt.zeros((self._padding, self._embedder.get_dims()))

        left1 = select["targets"][which][self.span1L][0]
        left2 = select["targets"][which][self.span2L][0]

        right1 = select["targets"][which][self.span1R][1]
        right2 = select["targets"][which][self.span2R][1]

        span1L, span1R = match_tokens(select["indices"], left1, right1)
        span2L, span2R = match_tokens(select["indices"], left2, right2)

        padded_embedding1[:(span1R - span1L), :] += select["embeddings"][0][span1L:span1R, :]
        padded_embedding2[:(span2R - span2L), :] += select["embeddings"][0][span2L:span2R, :]

        sample = {
            "embedding1": padded_embedding1,
            "embedding2": padded_embedding2,
            "target": 1.0 if select["targets"][which][self.target] == "entailed" else 0.0
        }
        return sample


class MetonymyDataSet(Dataset):
    """
    Use the Metonymy datasets from https://www.aclweb.org/anthology/P17-1115/
    """

    def __init__(self, path, embedder, padding=128):
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
        self.span1L = "span1L"
        self.span1R = "span1R"
        self.span2L = None
        self.span2R = None
        self.target = "group"

        self._data = pd.read_csv(path)

        logger.debug(f"Embedding {len(self._data):,} sentences for Metonymy")

        st = time.time()
        indices, tokens, embedding = embed_sentences(self._embedder, self._data[self.sequence])
        self._data["indices"] = pd.Series(indices)
        self._data["embeddings"] = pd.Series(embedding)

        logger.debug(f"Data loaded for Metonymy in {timedelta(seconds=time.time() - st)}")

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

        # ind, tokens, enc = self._embedder.tokenize(select[self.sequence])

        # embedding = self._embedder.embed(enc)

        padded_embedding = tt.zeros((self._padding, self._embedder.get_dims()))

        right = select[self.span1R]

        assert select[self.span1L] < right, f"Span start not before span end: {select[self.sequence]}"

        span1L, span1R = match_tokens(select["indices"], select[self.span1L], right)

        padded_embedding[:(span1R - span1L), :] += select["embeddings"][0][span1L:span1R, :]

        sample = {
            "embedding1": padded_embedding,
            "embedding2": -1.,
            "target": 1.0 if select[self.target] == "metonymic" else 0.0
        }
        return sample


class TroFiDataSet(Dataset):
    """
    Use the TroFi datasets from https://github.com/sfu-natlang/trofi-metaphor-data
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

        logger.debug(f"Embedding {len(self._data):,} sentences for TroFi")

        st = time.time()
        indices, tokens, embedding = embed_sentences(self._embedder, self._data[self.sequence])
        self._data["indices"] = pd.Series(indices)
        self._data["embeddings"] = pd.Series(embedding)

        logger.debug(f"Data loaded for TroFi in {timedelta(seconds=time.time() - st)}")

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

        # ind, tokens, enc = self._embedder.tokenize(select[self.sequence])

        # embedding = self._embedder.embed(enc)

        padded_embedding = tt.zeros((self._padding, self._embedder.get_dims()))

        # span start gets shifted left since 0/1 indexing issue
        right = select[self.span1L]

        assert select[self.span1L]-1 < right, f"Span start not before span end: {select[self.sequence]}"

        span1L, span1R = match_tokens(select["indices"], select[self.span1L]-1, right)

        padded_embedding[:(span1R - span1L), :] += select["embeddings"][0][span1L:span1R, :]

        sample = {
            "embedding1": padded_embedding,
            "embedding2": -1.,
            "target": 1.0 if select[self.target] == "L" else 0.0
        }
        return sample
