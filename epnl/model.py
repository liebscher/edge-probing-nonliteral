import logging, logging.config

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('epnl')

import time
from datetime import timedelta, datetime

import os

import torch.optim as op
from torch.utils.data import DataLoader

import torch as tt
import torch.nn as nn
import torch.nn.functional as F

from epnl.tasks import Task


def get_optimizer(model):
    """

    Parameters
    ----------
    model
    task

    Returns
    -------

    """
    lossf = op.Adam(model.parameters(), lr=0.001)

    return lossf


def validate_model(model, loader, task):
    start_time = time.time()

    t_metric = {m: 0.0 for m in task.metrics}

    logger.debug(f"Validating task {task.get_name()}")

    model.eval()

    for i_batch, batch in enumerate(loader):
        with tt.no_grad():
            outputs = model(batch)

            for m in outputs["metrics"]:
                t_metric[m] += outputs["metrics"][m]

    # take each metric and divide it by the batch size for a macro metric
    metrics = [f"{m}: {t_metric[m] / (i_batch+1):.4f}" for m in t_metric]
    logger.debug(f"Validation Metrics [{timedelta(seconds=time.time() - start_time)}]: {', '.join(metrics)}")


def train_model(model, optimizer, task, train_params):
    """
    Given a task, train a model

    Parameters
    ----------
    model
    optimizer
    task
    train_params

    Returns
    -------

    """
    logger.info("Training Model")

    # fetch data for this task
    data_train = task.get_data(Task.TRAIN)
    data_dev = task.get_data(Task.DEV)

    # load the data into a PyTorch loader
    loader_train = DataLoader(data_train, batch_size=train_params["batch_size"], shuffle=True, num_workers=4)
    loader_dev = DataLoader(data_dev, batch_size=train_params["validation_batch_size"], shuffle=True, num_workers=4)

    epoch = 1

    stop = False
    # continue to train until stopping condition is met
    while not stop:
        # batch train

        start_time = time.time()

        t_loss = tt.zeros(1)
        t_metric = {m: 0.0 for m in task.metrics}

        logger.debug(f"Training task {task.get_name()}: epoch [{epoch}]")

        for i_batch, batch in enumerate(loader_train):
            optimizer.zero_grad()

            outputs = model(batch)

            loss = outputs['loss']

            for m in outputs["metrics"]:
                t_metric[m] += outputs["metrics"][m]

            t_loss += loss

            loss.backward()

            optimizer.step()

        logger.debug(f"Loss [{timedelta(seconds=time.time() - start_time)}]: [{t_loss.item() / train_params['batch_size']:.4f}]")

        # take each metric and divide it by the batch size for a macro metric
        metrics = [f"{m}: {t_metric[m] / (i_batch+1):.4f}" for m in t_metric]
        logger.debug(f"Metrics: {', '.join(metrics)}")

        # validation
        if epoch % train_params["validation_interval"] == 0:
            validate_model(model, loader_dev, task)

        epoch += 1

        if epoch == 20:
            stop = True

    # always validate the model as a last step before quitting
    validate_model(model, loader_dev, task)

    logger.info(f"Finished training task {task.get_name()}: epochs [{epoch}], " +
                f"loss [{t_loss.item() / train_params['batch_size']:.4f}]")


def save_model(model, optimizer, task_name, path):
    """
    Save a local version of the model and the optimizer

    Parameters
    ----------
    model
    optimizer
    path
    """

    model_name = f"model-{task_name}-{datetime.strftime(datetime.now(), '%d%b%y-%H-%M')}.pt"
    opt_name = f"optimizer-{task_name}-{datetime.strftime(datetime.now(), '%d%b%y-%H-%M')}.pt"

    tt.save(model.state_dict(), os.path.join(path, model_name))
    logger.info(f"Model Saved - {model_name}")

    tt.save(optimizer.state_dict(),  os.path.join(path, opt_name))
    logger.info(f"Model Saved - {opt_name}")


class Classifier(nn.Module):
    """
    Create a classifier for model output
    """

    def __init__(self, inp_dim, n_classes):
        """

        Parameters
        ----------
        inp_dim
        n_classes
        """
        super(Classifier, self).__init__()

        # Make a simple linear classifier
        self._classifier = nn.Linear(inp_dim, n_classes)

        logger.debug(f"Initializing new Classifier - input dims: {inp_dim}, output dims: {n_classes}")

    def forward(self, embedding):
        """

        Parameters
        ----------
        embedding : torch.tensor ()

        Returns
        -------
        logits : torch.tensor (n_classes, )

        """
        logits = self._classifier(embedding)

        return logits


class EdgeProbingModel(nn.Module):
    """
    Given a task and dataset, train a model of the contextual embeddings encoded information.
    """

    def __init__(self, task, embedder):
        """

        Parameters
        ----------
        task : epnl.Task
        embedder : epnl.Embedding
        """
        super(EdgeProbingModel, self).__init__()
        self._task = task

        self._embedder = embedder

        # create a MLP classifier for the task
        self._classifier = Classifier(self._embedder.get_dims() * (int(task.double) + 1),
                                      task.get_output_dims())

        logger.debug(f"Initializing new EdgeProbingModel - task: {task.get_name()}")

    def forward(self, batch):
        """

        Parameters
        ----------
        batch : torch.tensor

        Returns
        -------
        out : dict
            logits : [torch.tensor (n_classes, )]
            loss : [torch.tensor (1, )]
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

        if self._task.double:
            assert "span2" in batch, "Task marked as double spans, but no second span found"
            logits = [
                tt.sigmoid(self._classifier(tt.cat([sp1, sp2]))) for sp1, sp2 in zip(batch["span1"], batch["span2"])
            ]
        else:
            logits = [tt.sigmoid(self._classifier(sp)) for sp in batch["span1"]]

        out["logits"] = logits

        if "target" in batch:
            out["loss"] = self.compute_loss(tt.cat(logits), batch["target"].float())

        out["metrics"] = {}
        for metric in self._task.metrics:
            out["metrics"][metric] = self.compute_metric(metric, tt.cat(logits), batch["target"].float())

        return out

    def compute_loss(self, logits, targets):
        """
        Compute loss for the given batch on the specified task

        Parameters
        ----------
        logits : torch.tensor (n_classes, )
        targets : torch.tensor (n_classes, )

        Returns
        -------
        loss : torch.tensor (1, )
        """

        return F.binary_cross_entropy(logits, targets)

    def compute_metric(self, metric, pred, target):
        pred = tt.ge(pred, 0.5).to(tt.float32)
        tp = (target * pred).sum().to(tt.float32)
        tn = ((1 - target) * (1 - pred)).sum().to(tt.float32)
        fp = ((1 - target) * pred).sum().to(tt.float32)
        fn = (target * (1 - pred)).sum().to(tt.float32)

        eps = 1e-7

        if metric == "acc":
            return (tp + tn) / (tp + tn + fp + fn)

        elif metric == "f1":
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * (precision * recall) / (precision + recall + eps)

            return f1

        return 0.0

    def __repr__(self):
        return f"<epnl.model.EdgeProbingModel object [N: {self._n_classes} E: {self._embedder} D: {self._data}]>"

    def __str__(self):
        return f"EdgeProbingModel <T: {str(self._task)}>"
