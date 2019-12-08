import time
from datetime import timedelta, datetime

import os

import json

import torch.optim as op
from torch.utils.data import DataLoader

import torch as tt
import torch.nn as nn
import torch.nn.functional as F

from epnl.tasks import Task

import logging, logging.config

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('epnl')


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


def get_metric(rates, metric):
    eps = 1e-7

    if metric == "acc":
        return (rates["tp"] + rates["tn"] + eps) / (rates["tp"] + rates["tn"] + rates["fp"] + rates["fn"] + eps)
    elif metric == "f1":
        precision = rates["tp"] / (rates["tp"] + rates["fp"] + eps)
        recall = rates["tp"] / (rates["tp"] + rates["fn"] + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)

        return f1
    elif metric == "mcc":
        num = rates["tp"] * rates["tn"] - rates["fp"] * rates["fn"]
        dem = ((rates["tp"] + rates["fp"])*(rates["tp"] + rates["fn"])*(rates["tn"] + rates["fp"])*(rates["tn"] + rates["fn"]))**0.5

        return (num + eps) / (dem + eps)

    return 0.0


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

    rates = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    logger.debug(f"Validating task {task.get_name()}")

    model.eval()

    for i_batch, batch in enumerate(loader):
        with tt.no_grad():
            outputs = model(batch)

            tp, tn, fp, fn = outputs["rates"]
            rates["tp"] += tp
            rates["tn"] += tn
            rates["fp"] += fp
            rates["fn"] += fn

    metrics = {
        "f1": get_metric(rates, "f1"),
        "acc": get_metric(rates, "acc"),
        "mcc": get_metric(rates, "mcc")
    }

    # take each metric and divide it by the batch size for a macro metric
    logger.debug(f"Validation F1 [{timedelta(seconds=time.time() - start_time)}]: {metrics['f1']:.4f}")

    return {
        "time": time.time(),
        "dur": time.time() - start_time,
        "metrics": metrics
    }


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
    loader_dev = DataLoader(data_dev, batch_size=train_params["validation_batch_size"], shuffle=False, num_workers=4)

    epoch = 1
    train_reports = []
    max_val = 0.0
    val_reports = []

    rates = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    stop = False
    # continue to train until stopping condition is met
    while not stop:
        # batch train

        start_time = time.time()

        t_loss = tt.zeros(1)

        logger.debug(f"Training task {task.get_name()}: epoch [{epoch}]")

        for i_batch, batch in enumerate(loader_train):
            optimizer.zero_grad()

            outputs = model(batch)

            loss = outputs['loss']

            tp, tn, fp, fn = outputs["rates"]
            rates["tp"] += tp
            rates["tn"] += tn
            rates["fp"] += fp
            rates["fn"] += fn

            t_loss += loss

            loss.backward()
            optimizer.step()

        report = {
            "epoch": epoch,
            "time": time.time(),
            "dur": time.time() - start_time,
            "loss": t_loss.item() / (i_batch+1)
        }

        logger.debug(f"Loss [{timedelta(seconds=report['dur'])}]: [{report['loss']:.4f}]")

        metrics = {
            "f1": get_metric(rates, "f1"),
            "acc": get_metric(rates, "acc"),
            "mcc": get_metric(rates, "mcc")
        }

        logger.debug(f"F1: {metrics['f1']:.4f}")

        report["metrics"] = metrics

        train_reports.append(report)

        # validation
        if epoch % train_params["validation_interval"] == 0:

            val_report = validate_model(model, loader_dev, task)

            val_report["epoch"] = epoch

            val_reports.append(val_report)

            if "f1" in val_report["metrics"] and val_report["metrics"]["f1"] > max_val:
                max_val = val_report["metrics"]["f1"]
                save_model(model, optimizer, task.get_name(), train_params["output_path"])

        epoch += 1

        if epoch == 25:
            stop = True

    # always validate the model as a last step before quitting
    val_report = validate_model(model, loader_dev, task)

    val_report["epoch"] = epoch

    val_reports.append(val_report)

    if "f1" in val_report["metrics"] and val_report["metrics"]["f1"] > max_val:
        save_model(model, optimizer, task.get_name(), train_params["output_path"])

    save_history(train_reports, val_reports, train_params["output_path"])

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


def save_history(training_reports, validation_reports, path):
    for r in training_reports:
        r["set"] = "train"

    for r in validation_reports:
        r["set"] = "val"

    training_reports.extend(validation_reports)

    name = f"history-{datetime.strftime(datetime.now(), '%d%b%y-%H-%M')}.json"

    with open(os.path.join(path, name), "w") as f:
        json.dump(training_reports, f)

    logger.info(f"History saved: {name}")


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

        # Make a linear classifier
        self._classifier = nn.Sequential(
            nn.Linear(inp_dim, 512),
            nn.Tanh(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, n_classes)
        )

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
        self._n_classes = task.get_output_dims()

        self._out_dims = 256

        # create a MLP classifier for the task
        self._classifier = Classifier(self._out_dims * (int(task.double) + 1),
                                      self._n_classes)

        # create span1 pooling object
        self._pooler1 = Pooler(
            project=True,
            inp_dim=self._embedder.get_dims(),
            out_dim=self._out_dims,
            pooling_type="max"
        )
        if self._task.double:
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
        # linear project of these sets of spans
        # pool each span into itself
        # concat the spans
        # MLP and output

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
            # print(span1_pool.size()) # batch size x out_dims

            logits = tt.sigmoid(self._classifier(span1_pool)).flatten()
            # print(logits.size())

        out["logits"] = logits
        out["loss"] = self.compute_loss(logits, batch["target"].float())
        out["rates"] = self.compute_rates(logits, batch["target"].float())

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

    def compute_rates(self, pred, target):
        """
        Compute True Pos., True Neg., False Pos., and False Neg.

        Parameters
        ----------
        pred
        target

        Returns
        -------
        tuple
        """
        pred = tt.ge(pred, 0.5).to(tt.float32)
        tp = (target * pred).sum().to(tt.float32)
        tn = ((1 - target) * (1 - pred)).sum().to(tt.float32)
        fp = ((1 - target) * pred).sum().to(tt.float32)
        fn = (target * (1 - pred)).sum().to(tt.float32)

        return tp.item(), tn.item(), fp.item(), fn.item()

    def __repr__(self):
        return f"<epnl.model.EdgeProbingModel object [N: {self._n_classes} E: {self._embedder}]>"

    def __str__(self):
        return f"EdgeProbingModel <T: {str(self._task)}>"
