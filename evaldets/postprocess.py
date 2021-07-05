import argparse
from functools import wraps
import glob
import logging
import os
import warnings

import numpy as np
import pandas as pd

from uo.utils import dirbasename, load, save
from .results import DetectionResults

logger = logging.getLogger()


def cached_directory_data(f):
    @wraps(f)
    def wrapper(directory, *args):
        directory = os.path.expanduser(directory)
        cache_file = os.path.join(directory, f"{f.__name__}.pkl.gz")
        if os.path.exists(cache_file):
            logger.info(f"Loading cached {f.__name__} results from {cache_file}.")
            return load(cache_file)
        value = f(directory, *args)
        save(value, cache_file)
        logger.info(f"Cached {f.__name__} results to {cache_file}.")
        return value

    return wrapper


SCORE_TRHS = np.arange(0.05, 1, 0.05)


def get_summary(subdir, t_score=0):
    dr = DetectionResults(subdir)
    summary = dr.summary(t_score=t_score)
    # logger.debug(f"{subdir=}, {dirbasename(subdir)=}")
    summary["subdir"] = dirbasename(subdir)  # subdirs now end in /
    return summary


@cached_directory_data
def summaries_by_tc(result_dir):
    dr = DetectionResults(result_dir)
    data = []
    for t_score in SCORE_TRHS:
        t_score = round(t_score, 2)
        data.append(dr.summary(t_score=t_score))
    return pd.DataFrame(data=data)


def load_meta(subdir):
    results = os.path.join(subdir, "rich_results.json")
    return load(results)


@cached_directory_data
def subdir_summaries(top_dir):
    subdirs = sorted(glob.glob(os.path.join(top_dir, "*/")))
    data = []
    for subdir in subdirs:
        summary = get_summary(subdir)
        meta = load_meta(subdir)
        # logger.debug(f"{meta=}")
        summary["model"] = meta["model"]
        summary["quality"] = meta["quality"]
        data.append(summary)
    return pd.DataFrame(data)


class Summary:
    """For baseline subdirectory."""

    def __init__(self, reval_dir: str) -> None:
        self.reval_dir = reval_dir
        self.phys_dir = os.path.expanduser(reval_dir)
        self.subdirs = sorted(glob.glob(os.path.join(self.phys_dir, "*/")))
        self.metadata = {subdir: load_meta(subdir) for subdir in self.subdirs}

    def get_summaries(self):
        return subdir_summaries(self.phys_dir)

    def tc_summaries(self):
        for s in self.subdirs:
            model = self.metadata[s]["model"]
            df = summaries_by_tc(s)
            yield model, df

    def plot_tc_summaries(self, axes=None, **kwargs):
        if axes is not None:
            axes = iter(axes.ravel())
        subplot_ord = ord("A")
        for model, df in self.tc_summaries():
            if axes is not None:
                ax = next(axes)
                kwargs["ax"] = ax
            df.plot(
                x="T_c",
                y=["PPV", "TPR", "F1"],
                ylim=(0, 1),
                ylabel="value",
                title=f"{chr(subplot_ord)}: {model}",
                **kwargs,
            )
            subplot_ord += 1


class GrandSummary:
    """For top level reval directory."""

    def __init__(self, reval_dir: str) -> None:
        self.reval_dir = reval_dir
        self.phys_dir = os.path.expanduser(reval_dir)
        self.subdirs = sorted(
            subdir
            for subdir in glob.glob(os.path.join(self.phys_dir, "*/"))
            if not dirbasename(subdir).startswith("baseline")
        )
        # logger.debug(f"{self.subdirs=}")

    def __repr__(self):
        return f"GrandSummary('{self.reval_dir}')"

    def q_summaries(self):
        for s in self.subdirs:
            df = subdir_summaries(s)
            yield df

    def save_q_summaries(self, excel=False):
        for df in self.q_summaries():
            model = df['model'][0]
            if excel:
                out = os.path.join(self.phys_dir, f"{model}.xlsx")
                df.to_excel(out)
            else:
                csv = os.path.join(self.phys_dir, f"{model}.csv")
                df.to_csv(out)
            logger.info(f"Saved Q summaries to: {out}")

    def plot_q_summaries(self, axes=None, **kwargs):
        if axes is not None:
            axes = iter(axes.ravel())
        subplot_ord = ord("A")
        for df in self.q_summaries():
            model = df['model'][0]
            if axes is not None:
                ax = next(axes)
                kwargs["ax"] = ax
            df.plot(
                x="quality",
                y=["PPV", "TPR", "F1"],
                ylim=(0, 1),
                ylabel="value",
                title=f"{chr(subplot_ord)}: {model}",
                **kwargs,
            )
            subplot_ord += 1


def load_rich_results(reval_dir):
    rich_results = [
        load(rr) for rr in sorted(glob.glob(f"{reval_dir}/*/rich_results.json"))
    ]
    if not rich_results:
        warnings.warn(
            f"No rich results found in {reval_dir}. "
            "Make sure to pass a directory with evaluator_dump_<model>_<quality> subdirectories."
        )
    return rich_results


def baseline_table(reval_dir, header=False):
    # previously as evaluate_coco.py side-effect
    metrics = load_rich_results(reval_dir)

    if header:
        # \newcommand\tsub[1]{\textsubscript{#1}}
        print(
            r"Model & AP & mAP\tsub{.5} & mAP\tsub{.75} & AP\tsub{l} & AP\tsub{m}"
            r" & AP\tsub{s} & TPR & PPV & TP & FP & EX \\"
        )
        print(r"\midrule")

    for row in metrics:
        row["precision"] *= 100
        row["recall"] *= 100
        row["model"] = row["model"].replace("_", r"\_")  # LaTeX excape
        print(
            "{model} & {AP:.1f} & {AP50:.1f} & {AP75:.1f} & {APl:.1f} & {APm:.1f}"
            r" & {APs:.1f} & {recall:.1f} & {precision:.1f} & {tp:,} & {fp:,} & {ex:,} \\".format(
                **row
            )
        )


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reval_dir")
    parser.add_argument("--header", "-g", action="store_true")
    args = parser.parse_args()
    baseline_table(args.reval_dir, args.header)


if __name__ == "__main__":
    _main()
