import bz2
import gzip
import inspect
import json
from functools import wraps
from itertools import islice


def logged(func):
    # based on: https://stackoverflow.com/questions/3467526/
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("call:", func.__name__, args)
        return func(*args, **kwargs)

    return wrapper


def aspectize(cls, decorator):
    # based on: https://stackoverflow.com/questions/3467526/
    for name, fn in inspect.getmembers(cls, inspect.isfunction):
        if name.startswith('__'):
            # printing *args would cause infinite recursion with __repr__
            continue
        setattr(cls, name, decorator(fn))


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
