import bz2
import glob
import gzip
import inspect
import json
import logging
import os
import pickle
import zlib
from functools import wraps
from itertools import islice


def dirbasename(path: str) -> str:
    if path.endswith("/"):
        return os.path.basename(os.path.dirname(path))
    return os.path.basename(path)


def cached(path):
    """TODO: implement smarter caching based on f, *args."""

    def wrap(f):
        @wraps(f)
        def wrapper(*args):
            return f(*args)

        return wrapper

    return wrap


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
        if name.startswith("__"):
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
    if not os.path.exists(path):
        if os.path.exists(path + ".gz"):
            logging.getLogger().debug(f"Loading {path}.gz instead of missing {path}")
            path = path + ".gz"
        else:
            raise ValueError(f"File not found: {path}")

    if path.endswith(".json.gz"):
        with gzip.open(path) as fs:
            return json.load(fs)
    if path.endswith(".json.bz2"):
        with bz2.open(path) as fs:
            return json.load(fs)
    if path.endswith(".json"):
        with open(path) as fs:
            return json.load(fs)
    if path.endswith(".pkl"):
        with open(path, "rb") as pkl:
            return pickle.load(pkl)
    if path.endswith(".pkl.gz"):
        with gzip.open(path) as pkl:
            return pickle.load(pkl)

    raise ValueError(f"unknown file type: {path}")


def save(obj: dict, path: str) -> None:
    if path.endswith(".json.gz"):
        with gzip.open(path, "w") as fs:
            return json.dump(obj, fs)
    if path.endswith(".json.bz2"):
        with bz2.open(path, "w") as fs:
            return json.dump(obj, fs)
    if path.endswith(".json"):
        with open(path) as fs:
            return json.dump(obj, fs)
    if path.endswith(".pkl"):
        with open(path, "wb") as pkl:
            return pickle.dump(obj, pkl)
    if path.endswith(".pkl.gz"):
        try:
            with gzip.open(path, "wb") as pkl:
                return pickle.dump(obj, pkl)
        except zlib.error:
            logging.getLogger().exception(f"Error unzipping: {path}")
            raise

    raise ValueError(f"unknown file type: {path}")


def top(iterable, n=10):
    return list(islice(iterable, n))


def warm_cache(glob_expr, function, sequential=False):
    """Execute function for each subdirectory matching glob_expr."""
    try:
        from multiprocess import Pool
    except ImportError:
        from multiprocessing import Pool

    dirs = sorted(glob.glob(os.path.expanduser(glob_expr)))

    if sequential:
        for dir in dirs:
            function(dir)
        return

    with Pool() as p:
        p.map(function, dirs)
