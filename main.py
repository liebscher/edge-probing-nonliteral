import sys, logging, logging.config

from epnl import model, tasks, embedding, util

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('epnl')

def check_args(args):
    """
    Ensure the arguments provided are valid

    Parameters
    ----------
    args
    """
    accepted = [
        "--tasks",
        "--embedder",
        "--pooling-project",
        "--pooling-type",
        "--warm"
    ]

    for arg in args:
        assert arg in accepted, f"Unaccepted argument given: \"{arg}\""


def build_model(args, task, embedder):
    """
    Based upon the task and embedding schema, build a model to be trained

    Parameters
    ----------
    args
    task
    embedder

    Returns
    -------
    model
    """

    if "pooling-project" not in args:
        args["pooling-project"] = True

    if "pooling-type" not in args:
        args["pooling-type"] = "max"

    # TODO: accept a model path for warm starting
    if "warm" in args:
        pass

    _model = model.EdgeProbingModel(task, embedder)

    return _model


def build_embedder(args):
    """
    Build an embedding object. Acceptable embeddings may be found in embedding.py

    Parameters
    ----------
    args

    Returns
    -------
    embedder
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
    tasks
    embedder
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
        elif t == "rel":
            _tasks.append(tasks.EdgeProbingTask("rel", 19, embedder=_embedder))
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
            if value in ["True", "true", "t", 1]:
                model_setup["pooling-project"] = True
            elif value in ["False", "false", "f", 0]:
                model_setup["pooling-project"] = False
            else:
                raise TypeError(f"Invalid parameter for pooling-project: \"{value}\"")
        elif param == "--pooling-type":
            model_setup["pooling-type"] = value

    logger.debug(f"epnl arguments: {model_setup}")

    _tasks, _embedder = build_tasks(model_setup)

    # cycle through each task to train and validate a model
    for _task in _tasks:
        _model = build_model(model_setup, _task, _embedder)

        # set up the training parameters for each task
        train_params = {
            "batch_size": 32,
            "learning_rate": 1e-5,
            "validation_batch_size": 32,
            "validation_interval": 5,
            "output_path": "epnl/output/"
        }

        _optimizer, _scheduler = model.get_optimizer(_model, train_params)

        model.train_model(_model, _optimizer, _scheduler, _task, train_params)

        model.save_model(_model, _optimizer, _task.get_name(), train_params["output_path"])

    logger.info("Finished")

    exit(1)


if __name__ == "__main__":
    main()
