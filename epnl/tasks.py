import logging, logging.config

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('epnl')

from os import path

from epnl import data

predetermined_data = {
    'trofi': {
        'path_train': 'data/trofi/train.csv',
        'path_test': 'data/trofi/test.csv',
        'path_dev': 'data/trofi/dev.csv',
        'double': False,
        'set': data.TroFiDataSet
    },
    'metonymy': {
        'path_train': 'data/metonymy/metonymy_train.csv',
        'path_test': 'data/metonymy/metonymy_testing.csv',
        'path_dev': 'data/metonymy/metonymy_dev.csv',
        'double': False,
        'set': data.MetonymyDataSet
    },
    'dpr': {
        'path_train': 'data/dpr/train.json',
        'path_test': 'data/dpr/test.json',
        'path_dev': 'data/dpr/dev.json',
        'double': True,
        'set': data.DPRDataSet
    },
    'rel': {
        'path_train': 'data/rel/training.csv',
        'path_test': 'data/rel/testing.csv',
        'path_dev': 'data/rel/dev.csv',
        'double': True,
        'set': data.RelDataSet
    }
}


def get_statistics(task):

    def count_examples(data):
        return len(data)

    def count_vocab(data):
        vocab = 0
        for sentence in data._data[data.sequence]:
            spl = sentence.split()
            vocab += len(spl)

        return vocab

    def count_unique(data):
        vocab = set()
        for sentence in data._data[data.sequence]:
            spl = set(sentence.split())
            vocab = vocab | spl

        return len(vocab)

    partitions = {
        Task.TRAIN: {},
        Task.DEV: {},
        Task.TEST: {}
    }
    for partition in partitions:
        data = task.get_data(partition, embed=False)
        partitions[partition] = {
            "n_ex": count_examples(data),
            "n_tk": count_vocab(data),
            "n_uq": count_unique(data)
        }

    return partitions


class Task:
    """
    Interface (probably inherited) for a generic Task
    """

    TRAIN = "train"
    TEST = "test"
    DEV = "dev"

    def __init__(self, name, n_classes, embedder):
        self._name = name
        self._n_classes = n_classes
        self._embedder = embedder
        self._data_paths = {}
        self.double = predetermined_data[name]["double"]

    def setup_data_path(self, partition):
        """
        Establish a path to the dataset.

        Parameters
        ------
        partition : str
            One of the class constants

        Returns
        -------
        path : string
            An absolute filepath to the data file using the map at the head of this file
        """
        return path.join(path.dirname(path.abspath(__file__)), predetermined_data[self._name][f"path_{partition}"])

    def get_data(self, partition, embed=True):
        """
        Instantiate a dataset object using the map at the head of this file

        Returns
        -------
        set : torch.utils.data.DataSet
            A child of a DataSet class
        """
        # these sets are defined at the top of this file
        loader = predetermined_data[self._name]["set"]
        return loader(self._data_paths[partition], self._embedder, embed=embed)

    def get_name(self):
        """
        Returns
        -------
        N : string
            Name of the task

        """
        return self._name


class EdgeProbingTask(Task):
    """
    Interface for an Edge Probing Task
    """

    def __init__(self, name, n_classes, **kwargs):
        """

        Parameters
        ----------
        name : string
            Name of the individual task
        n_classes : int
            Number of output classes
        kwargs
            Keywords passed to parent `Task` class
        """

        super(EdgeProbingTask, self).__init__(name, n_classes, **kwargs)

        assert path.isdir(
            path.join(path.join(path.dirname(path.abspath(__file__)), "data/"), name)
        ), f"Task data directory must be of the same name: data/{name}"

        self._data_paths = {
            self.TRAIN: self.setup_data_path(self.TRAIN),
            self.TEST: self.setup_data_path(self.TEST),
            self.DEV: self.setup_data_path(self.DEV),
        }

        logger.debug("Initializing new EdgeProbingTask")

    def get_output_dims(self):
        """
        Returns
        -------
        dims : int
            Number of output dimensions
        """
        return self._n_classes

    def __repr__(self):
        return f"<epnl.tasks.EdgeProbingTask object [N: {self._n_classes} E: {self._embedder}]>"

    def __str__(self):
        return f"EdgeProbingTask <N: {self._n_classes} E: {self._embedder}>"
