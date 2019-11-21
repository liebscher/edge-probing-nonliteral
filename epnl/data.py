import logging, logging.config

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('epnl')

import pandas as pd
import numpy as np

import torch as tt
import torch.nn as nn

from torch.utils.data import Dataset


def match_tokens(embedding, ind, spanL, spanR):
    """
    Get the relevant list of embedded tokens given the span bounds
    """

    # ((0, 'he'), (1, 'couldn'), (1, "'"), (1, 't'), (2, 'have'), (3, 'fought')
    # 1:3

    enc_span1 = ind.index(spanL)
    enc_span2 = ind.index(spanR)

    return embedding[enc_span1:enc_span2, :]


class Pooler(nn.Module):
    """
    Perform pooling, potentially with a projection layer beforehand.
    We need pooling because the tokenizer may split tokens into multiple parts, so we merge those different parts
    together when necessary.
    """

    def __init__(self, project=True, inp_dim=512, out_dim=512, pooling_type="max"):
        super(Pooler, self).__init__()
        # projection to reduce token size
        self._projector = nn.Linear(inp_dim, out_dim) if project else lambda x: x
        self._pooling_type = pooling_type

        logger.debug(f"Initializing new Pooler - project: {project}, input dims: {inp_dim}, output dims: {out_dim}, " +
                     f"type: {pooling_type}")

    def forward(self, span_sequence):

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


class TestDataSet(Dataset):
    """
    Load a test dataset to ensure functionality
    """

    def __init__(self, path, embedder):

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
        right = select[self.span1R] if self.span1R else select[self.span1L]

        span1 = match_tokens(embedding[0], ind, select[self.span1L]-1, right)

        span1_pooled = self._pooler(span1)

        # print(span1.size(), span1_pooled.size())

        sample = {
            "span1": span1_pooled,
            "target": select[self.target].astype(float)
        }
        return sample
