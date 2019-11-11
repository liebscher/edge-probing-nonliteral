import numpy as np
import torch as tt


class Task:

    def __init__(self, name, n_classes):
        self._name = name
        self._n_classes = n_classes

    def load_data(self):
        pass


class MetaphorTask(Task):

    def __init__(self, name, n_classes):

        super(MetaphorTask, self).__init__(name, n_classes)