import logging, logging.config

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('epnl')

import numpy as np
import torch as tt
import torch.nn as nn
import torch.optim as op
from torch.utils.data import DataLoader

import torch as tt
import torch.nn as nn
import torch.nn.functional as F

tt.manual_seed(43)

def get_criterion_loss_function(model, task):
    critf = nn.CrossEntropyLoss()
    lossf = op.SGD(model.parameters(), lr=0.001)

    return critf, lossf


def train_model(model, args, task, train_params):

    logger.info("Training Model")

    # fetch data for this task
    data = task.get_data()
    # load the data into a PyTorch loader
    dataloader = DataLoader(data, batch_size=train_params["batch_size"], shuffle=True, num_workers=2)

    criterion, optimizer = get_criterion_loss_function(model, task)

    epoch = 1



    stop = False
    while not stop:
        # batch train

        t_loss = tt.zeros(1)

        logger.debug(f"Training task {task.get_name()}: epoch [{epoch}]")

        for i_batch, batch in enumerate(dataloader):
            # print(f"Batch: {i_batch}")

            optimizer.zero_grad()

            outputs = model(batch)

            loss = outputs['loss']
            t_loss += loss

            # print(loss)

            loss.backward()

            optimizer.step()

        epoch += 1

        logger.debug(f"Loss: [{t_loss.item() / train_params['batch_size']:.2f}]")

        if epoch == 10:
            stop = True

    logger.info(f"Finished training task {task.get_name()}: epochs [{epoch}], " +
                f"loss [{t_loss.item() / train_params['batch_size']:.2f}]")

def save_model(model, optimizer, path):

    logger.info("Saved Model")


class Classifier(nn.Module):
    """
    Create a classifier for model output
    """

    def __init__(self, inp_dim, n_classes):
        super(Classifier, self).__init__()

        # Make a simple linear classifier
        self._classifier = nn.Linear(inp_dim, n_classes)

        logger.debug(f"Initializing new Classifier - input dims: {inp_dim}, output dims: {n_classes}")

    def forward(self, embedding):
        logits = self._classifier(embedding)

        return logits


class EdgeProbingModel(nn.Module):
    """
    Given a task and dataset, train a model of the contextual embeddings encoded information.
    """

    def __init__(self, task, embedder):
        super(EdgeProbingModel, self).__init__()
        self._task = task

        self._embedder = embedder

        # # either create a pooling object or don't
        # self._pooler = Pooler(
        #     args["pooling-project"],
        #     self._embedder.get_dims(),
        #     self._embedder.get_dims(),
        #     args["pooling-type"]
        # ) if args["pooling"] else None

        # create a MLP classifier for the task
        self._classifier = Classifier(self._embedder.get_dims(), task.get_output_dims())

        logger.debug(f"Initializing new EdgeProbingModel - task: {task.get_name()}")

    def forward(self, batch):
        """
        Forward pass for Edge Probing models. Extract the appropriate span(s) and classify them, compute loss.

        We take in a list of contextual vectors and integer spans, then use a projection layer, then pool.

        These span representations are concated and fed into MLP with sigmoid output
        """
        out = dict()

        # we start with sentence
        # then we tokenize
        # then we embed
        # from this embedding, we need _only_ the embeddings within spans
        # project these sets of spans
        # pool each span into itself
        # ^^ happens in data.py ^^
        # concat the spans
        # MLP and output

        # pools = [self._pooler(span) for span in batch["span1"]]

        logits = [tt.sigmoid(self._classifier(sp)) for sp in batch["span1"]]

        out["logits"] = logits

        # print(logits)
        # print(batch["target"])

        if "target" in batch:
            out["loss"] = self.compute_loss(tt.cat(logits), batch["target"])

        return out

    def compute_loss(self, logits, targets):
        """
        Compute loss for the given batch on the specified task
        """

        # print(logit, tt.sigmoid(logit), target)

        return F.binary_cross_entropy(logits, targets)

    def __repr__(self):
        return f"<epnl.model.EdgeProbingModel object [N: {self._n_classes} E: {self._embedder} D: {self._data}]>"

    def __str__(self):
        return f"EdgeProbingModel <T: {str(self._task)}>"
