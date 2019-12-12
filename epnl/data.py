import json

import pandas as pd
import numpy as np

import torch as tt

from torch.utils.data import Dataset

import time
from datetime import timedelta

import logging, logging.config

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('epnl')


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
    """
    Given a list of string sentences, embed it into a high-dimensional space with an Embedder (e.g. BERT)

    Parameters
    ----------
    embedder : epnl.Embedder
    sentences : list[str]

    Returns
    -------
    indices : list[int]
    tokens : list[int]
    embeddings : list[torch.tensor]
    """
    indices = []
    tokens = []
    embeddings = []
    for sentence in sentences:
        index, token, enc = embedder.tokenize(sentence.lower())

        embedding = embedder.embed(enc)

        indices.append(index)
        tokens.append(token)
        embeddings.append(embedding)

    return indices, tokens, embeddings


class DPRDataSet(Dataset):
    """
    Use the DPR datasets
    """

    def embed_data(self):
        logger.debug(f"Embedding {len(self._data):,} sentences for DPR")

        st = time.time()
        indices, tokens, embedding = embed_sentences(self._embedder, self._data[self.sequence])
        self._data["indices"] = pd.Series(indices)
        self._data["embeddings"] = pd.Series(embedding)

        logger.debug(f"Data loaded for DPR in {timedelta(seconds=time.time() - st)}")

    def __init__(self, path, embedder, padding=64, embed=True):
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

        import copy

        self._data = []
        for line in open(path).readlines():
            parse = json.loads(line)

            if len(parse["targets"]) == 2:
                a = copy.deepcopy(parse)
                del a["targets"]
                a.update(parse["targets"][0])
                self._data.append(a)

                b = copy.deepcopy(parse)
                del b["targets"]
                b.update(parse["targets"][1])
                self._data.append(b)
            elif len(parse["targets"]) == 1:
                a = copy.deepcopy(parse)
                del a["targets"]
                a.update(parse["targets"][0])
                self._data.append(a)

        self._data = pd.DataFrame(self._data)

        if embed:
            self.embed_data()

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

        padded_embedding1 = tt.zeros((self._padding, self._embedder.get_dims()))
        padded_embedding2 = tt.zeros((self._padding, self._embedder.get_dims()))

        left1 = select[self.span1L][0]
        left2 = select[self.span2L][0]

        right1 = select[self.span1R][1]
        right2 = select[self.span2R][1]

        span1L, span1R = match_tokens(select["indices"], left1, right1)
        span2L, span2R = match_tokens(select["indices"], left2, right2)

        padded_embedding1[:(span1R - span1L), :] += select["embeddings"][0][span1L:span1R, :]
        padded_embedding2[:(span2R - span2L), :] += select["embeddings"][0][span2L:span2R, :]

        sample = {
            "embedding1": padded_embedding1,
            "embedding2": padded_embedding2,
            "target": 1.0 if select[self.target] == "entailed" else 0.0
        }
        return sample


class MetonymyDataSet(Dataset):
    """
    Use the Metonymy datasets from https://www.aclweb.org/anthology/P17-1115/
    """

    def embed_data(self):
        logger.debug(f"Embedding {len(self._data):,} sentences for Metonymy")

        st = time.time()
        indices, tokens, embedding = embed_sentences(self._embedder, self._data[self.sequence])
        self._data["indices"] = pd.Series(indices)
        self._data["embeddings"] = pd.Series(embedding)

        logger.debug(f"Data loaded for Metonymy in {timedelta(seconds=time.time() - st)}")

    def __init__(self, path, embedder, padding=128, embed=True):
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

        if embed:
            self.embed_data()

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

    def embed_data(self):
        logger.debug(f"Embedding {len(self._data):,} sentences for TroFi")

        st = time.time()
        indices, tokens, embedding = embed_sentences(self._embedder, self._data[self.sequence])
        self._data["indices"] = pd.Series(indices)
        self._data["embeddings"] = pd.Series(embedding)

        logger.debug(f"Data loaded for TroFi in {timedelta(seconds=time.time() - st)}")

    def __init__(self, path, embedder, padding=64, embed=True):
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

        if embed:
            self.embed_data()

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


class RelDataSet(Dataset):
    """
    Use the Relation Classification datasets from
    https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview
    """

    def embed_data(self):
        logger.debug(f"Embedding {len(self._data):,} sentences for Relation Classification")

        st = time.time()
        indices, tokens, embedding = embed_sentences(self._embedder, self._data[self.sequence])
        self._data["indices"] = pd.Series(indices)
        self._data["embeddings"] = pd.Series(embedding)

        logger.debug(f"Data loaded for Relation Classification in {timedelta(seconds=time.time() - st)}")

    def __init__(self, path, embedder, padding=64, embed=True):
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
        self.span1L = "left1"
        self.span1R = "right1"
        self.span2L = "left2"
        self.span2R = "right2"
        self.target = "relation"

        self.labels = ['Product-Producer12', 'Member-Collection12',
                       'Entity-Destination12', 'Member-Collection21', 'Component-Whole12',
                       'Other', 'Message-Topic12', 'Product-Producer21',
                       'Entity-Origin12', 'Instrument-Agency12', 'Component-Whole21',
                       'Cause-Effect21', 'Entity-Origin21', 'Message-Topic21',
                       'Content-Container12', 'Cause-Effect12', 'Instrument-Agency21',
                       'Content-Container21', 'Entity-Destination21']

        self._data = pd.read_csv(path)

        if embed:
            self.embed_data()

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

        padded_embedding1 = tt.zeros((self._padding, self._embedder.get_dims()))
        padded_embedding2 = tt.zeros((self._padding, self._embedder.get_dims()))

        left1 = select[self.span1L]
        left2 = select[self.span2L]

        right1 = select[self.span1R]
        right2 = select[self.span2R]

        span1L, span1R = match_tokens(select["indices"], left1, right1)
        span2L, span2R = match_tokens(select["indices"], left2, right2)

        padded_embedding1[:(span1R - span1L), :] += select["embeddings"][0][span1L:span1R, :]
        padded_embedding2[:(span2R - span2L), :] += select["embeddings"][0][span2L:span2R, :]

        label = tt.zeros(len(self.labels))
        label[self.labels.index(select[self.target])] = 1.0

        sample = {
            "embedding1": padded_embedding1,
            "embedding2": padded_embedding2,
            "target": label
        }
        return sample
