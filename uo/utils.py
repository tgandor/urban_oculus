import bz2
import gzip
import json
from itertools import islice


def is_notebook():
    # https://stackoverflow.com/a/39662359/1338797
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def load(path: str) -> dict:
    if path.endswith(".json.gz"):
        with gzip.open(path) as fs:
            return json.load(fs)
    if path.endswith(".json.bz2"):
        with bz2.open(path) as fs:
            return json.load(fs)
    if path.endswith(".json"):
        with open(path) as fs:
            return json.load(fs)
    raise ValueError("unknown file type")


def top(iterable, n=10):
    return list(islice(iterable, n))
