import torch as tt
import torch.nn as nn
import transformers as tm
import numpy as np

class Pooler(nn.Module):
    """
    Perform pooling, potentially with a projection layer beforehand.
    """

    def __init__(self, project=True, pooling_type="mean"):
        self._project = project
        self._pooling_type = pooling_type

    def forward(self):
        if self._pooling_type == "mean":
            pass
        elif self._pooling_type == "max":
            pass

        return None


class Classifier(nn.Module):
    """
    Create a classifier for model output
    """

    def __init__(self, inp_dim, n_classes):
        self._classifier = None

    def forward(self):
        logits = self._classifier()

        return logits


class EdgeProbingModel(nn.Module):
    """
    Given a task and dataset, train a model of the contextual embeddings encoded information.
    """

    def __init__(self):
        self._span_extractor = None
        self._classifier = None


    def forward(self):
        """
        Forward pass for Edge Probing models. Extract the appropriate span(s) and classify them, compute loss.
        """
        out = dict()

        span_emb = None
        logits = self._classifier(span_emb)

        out["logits"] = logits

        return out