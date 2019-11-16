import logging, logging.config

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('epnl')

from os import path

import numpy as np
import torch as tt

from epnl import data

predetermined_data = {
    'metaphor': {'path': 'data/test.csv', 'set': data.TestDataSet}
}


class Task:

    def __init__(self, name, n_classes, embedder):
        self._name = name
        self._n_classes = n_classes
        self._embedder = embedder
        self._data_path = ""

    def setup_data_path(self):
        try:
            return path.join(path.dirname(path.abspath(__file__)), predetermined_data[self._name]["path"])
        except:
            logger.error(f"Task name unknown: {self._name}")

    def get_data(self):
        loader = predetermined_data[self._name]["set"]
        return loader(self._data_path)

    def get_name(self):
        return self._name

    def get_metric(self, metric):
        if metric == "acc":
            return 0.0
        elif metric == "f1":
            return 0.0
        elif metric == "mcc":
            return 0.0
        else:
            raise Exception(f"Metric not available: \"{metric}\"")


class EdgeProbingTask(Task):

    def __init__(self, name, n_classes, path, **kwargs):

        super(EdgeProbingTask, self).__init__(name, n_classes, **kwargs)

        self._data_path = path if path else self.setup_data_path()

        logger.debug("Initializing new EdgeProbingTask")

    def get_output_dims(self):
        return self._n_classes

    def __repr__(self):
        return f"<epnl.tasks.EdgeProbingTask object [N: {self._n_classes} E: {self._embedder} D: {self._data}]>"

    def __str__(self):
        return f"EdgeProbingTask <N: {self._n_classes} E: {self._embedder} D: {self._data}>"
