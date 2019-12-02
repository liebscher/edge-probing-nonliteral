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


def get_optimizer(model, args):
    """

    Parameters
    ----------
    model
    args

    Returns
    -------

    """
    return op.Adam(model.parameters(), lr=args["learning_rate"])


def validate_model(model, loader, task):
    """

    Parameters
    ----------
    model
    loader
    task

    Returns
    -------

    """
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


class Pooler(nn.Module):
    """
    Perform pooling, potentially with a projection layer beforehand.
    We need pooling because the tokenizer may split tokens into multiple parts, so we merge those different parts
    together when necessary.
    """

    def __init__(self, project=True, inp_dim=512, out_dim=256, pooling_type="max"):
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
        self._project = project
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
        if self._project:
            span_sequence = self._projector(span_sequence)

        if self._pooling_type == "mean":
            raise NotImplementedError
            # return tt.mean(span_sequence, dim=0)
        elif self._pooling_type == "max":
            return tt.max(span_sequence, dim=1)[0]

        return tt.sum(span_sequence, dim=1)


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

        self._out_dims = 256

        # create a MLP classifier for the task
        self._classifier = Classifier(self._out_dims * (int(task.double) + 1),
                                      task.get_output_dims())

        # create span1 pooling object
        self._pooler1 = Pooler(
            project=True,
            inp_dim=self._embedder.get_dims(),
            out_dim=self._out_dims,
            pooling_type="max"
        )
        # create span2 pooling object
        self._pooler2 = Pooler(
            project=True,
            inp_dim=self._embedder.get_dims(),
            out_dim=self._out_dims,
            pooling_type="max"
        )

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
        # ^^ happens in data.py ^^
        # project these sets of spans
        # pool each span into itself
        # concat the spans
        # MLP and output

        # print(batch["embedding1"].size())
        # print(batch["target"].size())

        if self._task.double:
            assert "embedding2" in batch, "Task marked as double spans, but no second span found"

            span1_pool = self._pooler1(batch["embedding1"])
            span2_pool = self._pooler2(batch["embedding2"])

            # print(span1_pool.size(), span2_pool.size())

            cat_pool = tt.cat([span1_pool, span2_pool], dim=1)

            # print(cat_pool.size())

            logits = tt.sigmoid(self._classifier(cat_pool)).flatten()

            # print(logits.size())

        else:
            span1_pool = self._pooler1(batch["embedding1"])

            # print(span1_pool.size()) # 32 x 256

            logits = tt.sigmoid(self._classifier(span1_pool)).flatten()

            # print(logits.size())

        out["logits"] = logits

        assert "target" in batch, "Target must be included in batch to calculate loss"

        out["loss"] = self.compute_loss(logits, batch["target"].float())

        out["metrics"] = {}
        for metric in self._task.metrics:
            out["metrics"][metric] = self.compute_metric(metric, logits, batch["target"].float())

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
