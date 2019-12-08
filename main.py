import sys, logging, logging.config

from epnl import model, tasks, embedding, util

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('epnl')

def check_args(args):
    """

    Parameters
    ----------
    args

    Returns
    -------

    """
    accepted = [
        "--tasks",
        "--embedder",
        "--pooling",
        "--pooling-project",
        "--pooling-type"
    ]

    for arg in args:
        assert arg in accepted, f"Unaccepted argument given: \"{arg}\""


def build_model(args, tasks, embedder):
    """
    Based upon the task and embedding schema, build a model to be trained

    Parameters
    ----------
    args
    tasks
    embedder

    Returns
    -------

    """

    if "pooling-project" not in args:
        args["pooling-project"] = True

    if "pooling-type" not in args:
        args["pooling-type"] = "max"

    _model = model.EdgeProbingModel(tasks, embedder)

    return _model


def build_embedder(args):
    """
    Build an embedding object. Acceptable embeddings may be found in embedding.py

    Parameters
    ----------
    args

    Returns
    -------

    """

    if "embedder" in args:
        _embedder = embedding.embedder_from_str(args["embedder"])
    else:
        # use default embedder transformer
        _embedder = embedding.BERTEmbedder()

    return _embedder


def build_tasks(args):
    """
    From the CL arguments, determine the structure of the tasks to be completed

    Parameters
    ----------
    args

    Returns
    -------

    """

    _embedder = build_embedder(args)

    if "tasks" in args:
        _strtasks = args["tasks"]
    else:
        raise ValueError("Task(s) must be defined as arguments")

    _tasks = []

    for t in _strtasks:
        if t == "trofi":
            _tasks.append(tasks.EdgeProbingTask("trofi", 1, embedder=_embedder))
        elif t == "dpr":
            _tasks.append(tasks.EdgeProbingTask("dpr", 1, embedder=_embedder))
        elif t == "metonymy":
            _tasks.append(tasks.EdgeProbingTask("metonymy", 1, embedder=_embedder))
        else:
            raise Exception("Task not recognized: \"{t}\"")

    return _tasks, _embedder


def main():
    """
    Edge Probing
    """
    logger.info("Run main")

    args = len(sys.argv)

    params = {sys.argv[i]: sys.argv[i + 1] for i in range(1, args, 2)}

    check_args(params.keys())

    model_setup = dict()

    for param in params:
        value = params[param]

        if param == "--tasks":
            model_setup["tasks"] = value.split(",")
        elif param == "--embedder":
            model_setup["embedder"] = value
        elif param == "--pooling-project":
            if value == "True":
                model_setup["pooling-project"] = True
            elif value == "False":
                model_setup["pooling-project"] = False
            else:
                raise TypeError(f"Invalid parameter for pooling-project: \"{value}\"")
        elif param == "--pooling-type":
            model_setup["pooling-type"] = value

    logger.debug(f"epnl arguments: {model_setup}")

    _tasks, _embedder = build_tasks(model_setup)

    for _task in _tasks:
        _model = build_model(model_setup, _task, _embedder)

        train_params = {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "validation_batch_size": 32,
            "validation_interval": 5,
            "output_path": "epnl/output/"
        }

        _optimizer = model.get_optimizer(_model, train_params)

        model.train_model(_model, _optimizer, _task, train_params)

        model.save_model(_model, _optimizer, _task.get_name(), train_params["output_path"])

    logger.info("Finished")

    exit(1)


if __name__ == "__main__":
    main()
