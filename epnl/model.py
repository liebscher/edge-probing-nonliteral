import logging, logging.config

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('epnl')

import numpy as np
import torch as tt
import torch.nn as nn
import torch.optim as op
import transformers as tm

import torch as tt
import torch.nn as nn

def get_criterion_loss_function(model, task):
    critf = nn.CrossEntropyLoss()
    lossf = op.SGD(model.parameters(), lr=0.001)

    return critf, lossf


def train_model(model, args, task, train_params):

    logger.info("Training Model")

    data_gen = task.load_data_generator()

    criterion, optimizer = get_criterion_loss_function(model, task)

    stop = False
    while not stop:
        # batch train

        for x in data_gen:
            pass

            # optimizer.zero_grad()
            #
            # outputs = model(x)
            # loss = criterion(outputs, y)
            #
            # loss.backward()
            #
            # optimizer.step()

        break

def save_model(model, optimizer, path):

    logger.info("Saved Model")


class Pooler(nn.Module):
    """
    Perform pooling, potentially with a projection layer beforehand.
    """

    def __init__(self, project=True, inp_dim=512, out_dim=512, pooling_type="max"):
        super(Pooler, self).__init__()
        self._projector = nn.Linear(inp_dim, out_dim) if project else lambda x: x
        self._pooling_type = pooling_type
        self._sequence = None

        logger.debug(f"Initializing new Pooler - project: {project}, input dims: {inp_dim}, output dims: {out_dim}, " +
                     f"type: {pooling_type}")

    def forward(self, sequence, mask):

        self._sequence = self._projector(sequence)

        if self._pooling_type == "mean":
            pass
        elif self._pooling_type == "max":
            pass

        return self._sequence


class Classifier(nn.Module):
    """
    Create a classifier for model output
    """

    def __init__(self, inp_dim, n_classes):
        super(Classifier, self).__init__()
        # Start with a simple linear classifier
        self._classifier = nn.Linear(inp_dim, n_classes)

        logger.debug(f"Initializing new Classifier - input dims: {inp_dim}, output dims: {n_classes}")

    def forward(self, embedding):
        logits = self._classifier(embedding)

        return logits


class EdgeProbingModel(nn.Module):
    """
    Given a task and dataset, train a model of the contextual embeddings encoded information.
    """

    def __init__(self, task, embedder, args):
        super(EdgeProbingModel, self).__init__()
        self._task = task

        self._embedder = embedder

        # either create a pooling object or don't
        self._pooler = Pooler(
            args["pooling-project"]
        ) if args["pooling"] else None

        # create a MLP classifier for the task
        self._classifier = Classifier(512, task.get_output_dims())

        logger.debug(f"Initializing new EdgeProbingModel - task: {task.get_name()}")


    def forward(self, batch):
        """
        Forward pass for Edge Probing models. Extract the appropriate span(s) and classify them, compute loss.
        """
        out = dict()

        span_embedding = self._pooler() if self._pooler else batch
        logits = self._classifier(span_embedding)

        out["logits"] = logits

        return out

    def compute_loss(self):
        """
        Compute loss for the given batch on the specified task
        """

        return 0

    def __repr__(self):
        return f"<epnl.model.EdgeProbingModel object [N: {self._n_classes} E: {self._embedder} D: {self._data}]>"

    def __str__(self):
        return f"EdgeProbingModel <T: {str(self._task)}>"
