import argparse
import glob
import itertools
import logging
import operator
import os
import sys
from typing import Optional, TextIO
import warnings
from datetime import datetime
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from uo.utils import dirbasename, load, save

from .names import T_
from .results import CROWD_ID_T, DetectionResults

logger = logging.getLogger()

DEFAULT_ORDER = "R101 R101_C4 R101_DC5 R101_FPN X101 R50 R50_C4 R50_DC5 R50_FPN".split()

TRAINED_ORDER = [
    # Row 1: Faster R-CNN
    "F50-STD",
    "F50-Q20",
    "F50-Q40",
    "F50-T20",
    "F50-T40",
    # Row 2: RetinaNet
    "R50-STD",
    "R50-Q20",
    "R50-Q40",
    "R50-T20",
    "R50-T40",
    # Row 3: ZOO models: F-RCNN, Retina; 2 empty panes, legend.
    "F50-D2",
    "R50-D2",
]


def cached_directory_data(f=None, *, compress=True):
    def decorator(f):
        @wraps(f)
        def wrapper(directory, *args):
            directory = os.path.expanduser(directory)
            cache_file = os.path.join(
                directory, f"{f.__name__}.pkl{'.gz' if compress else ''}"
            )
            if os.path.exists(cache_file):
                logger.info(f"Loading cached {f.__name__} results from {cache_file}.")
                return load(cache_file)
            value = f(directory, *args)
            save(value, cache_file)
            logger.info(f"Cached {f.__name__} results to {cache_file}.")
            return value

        return wrapper

    if f is None:
        return decorator
    return decorator(f)


def cached_with_args(f=None, *, compress=True):
    def decorator(f):
        @wraps(f)
        def wrapper(directory, *args):
            directory = os.path.expanduser(directory)
            signature = f.__name__
            if len(args):
                signature = "_".join([signature] + [str(a) for a in args])
            cache_file = os.path.join(
                directory, f"{signature}.pkl{'.gz' if compress else ''}"
            )
            if os.path.exists(cache_file):
                logger.info(
                    f"Loading cached {f.__name__}{args} results from {cache_file}."
                )
                return load(cache_file)
            value = f(directory, *args)
            save(value, cache_file)
            logger.info(f"Cached {f.__name__}{args} results to {cache_file}.")
            return value

        return wrapper

    if f is None:
        return decorator
    return decorator(f)


SCORE_TRHS = np.arange(0.05, 1, 0.05)


@cached_with_args
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


@cached_directory_data
def tp_fp_ex_by_tc(result_dir):
    dr = DetectionResults(result_dir)
    return pd.DataFrame(
        {
            "T_c": dr.scores_all(),
            "TP": dr.tp_sum_all(),
            "FP": dr.fp_sum_all(),
            "EX": dr.ex_sum_all(),
        }
    )


def load_meta(subdir, rich=True, model_is_basename=False):
    ret = None

    if rich:
        results = os.path.join(subdir, "rich_results.json")
        if os.path.exists(results):
            ret = load(results)
        else:
            logger.debug(f"No rich_results.json for {subdir}. Trying results.json")

    if ret is None:
        ret = load(os.path.join(subdir, "results.json"))
        ret.update(results["results"]["bbox"])
        del ret["results"]

    if model_is_basename:
        ret["model"] = dirbasename(subdir, 2)

    return ret


@cached_with_args
def subdir_summaries(top_dir: str, t_score=0):
    subdirs = sorted(glob.glob(os.path.join(top_dir, "*/")))
    data = []
    for subdir in subdirs:
        logger.debug(f"{subdir=}")
        summary = get_summary(subdir, t_score)
        meta = load_meta(subdir)
        summary["model"] = meta["model"]
        summary["quality"] = meta["quality"]

        data.append(summary)
    return pd.DataFrame(data)


def subdir_meta_df(model_dir: str, model_is_basename=False) -> pd.DataFrame:
    model_dir = os.path.expanduser(model_dir)
    subdirs = sorted(glob.glob(os.path.join(model_dir, "*/")))
    data = []

    for subdir in subdirs:
        meta = load_meta(subdir)
        data.append(meta)

    df = pd.DataFrame(data)
    if model_is_basename:
        df["model"] = dirbasename(model_dir)

    return df


def subdir_summaries_with_ap(model_dir, t_score=0, model_is_basename=False, raw=False):
    """Combine subdir_summaries(model_dir, t_score) and results.json."""
    model_dir = os.path.expanduser(model_dir)
    subdirs = sorted(glob.glob(os.path.join(model_dir, "*/")))
    data = []

    for subdir in subdirs:
        meta = load_meta(subdir, rich=False)
        summary = get_summary(subdir, t_score)
        meta.update(summary)
        if model_is_basename:
            meta["model"] = dirbasename(model_dir)

        data.append(meta)

    if raw:
        return data

    df = pd.DataFrame(data)
    return df


def _get_model_subdirectories(top_dir):
    return sorted(
        subdir
        for subdir in glob.glob(os.path.join(top_dir, "*/"))
        if not dirbasename(subdir).startswith("baseline")
        and not dirbasename(subdir).startswith("quality_")
    )


class Summary:
    """For baseline subdirectory."""

    def __init__(self, reval_dir: str, model_is_basename=False) -> None:
        self.reval_dir = reval_dir
        self.phys_dir = os.path.expanduser(reval_dir)
        self.model_is_basename = model_is_basename
        self._post_init()

    def _post_init(self):
        self.subdirs = sorted(glob.glob(os.path.join(self.phys_dir, "*/")))
        self.metadata = {subdir: load_meta(subdir) for subdir in self.subdirs}

    def get_summaries(self, t_score=0):
        return subdir_summaries(self.phys_dir, t_score)

    def tc_summaries(self):
        for s in self.subdirs:
            model = self.metadata[s]["model"]
            df = summaries_by_tc(s)
            yield model, df

    def plot_tc_summaries(
        self, axes=None, xlim=(1, 0), *, order=None, i18n=None, **kwargs
    ):
        if axes is not None:
            axes = iter(axes.ravel())
        subplot_ord = ord("A")
        models = dict(self.tc_summaries())
        for model in order if order else models.keys():
            df = models[model]
            if axes is not None:
                ax = next(axes)
                kwargs["ax"] = ax
            df.plot(
                x="T_c",
                y=["TPR", "PPV", "F1"],
                xlim=xlim,
                ylim=(0, 1),
                ylabel=T_(i18n, "value"),
                xlabel="$T_c$",
                title=f"{chr(subplot_ord)}: {model}",
                **kwargs,
            )
            if axes is None:
                ax = plt.gca()
            ax.set_xlabel(T_(i18n, "T_c"))
            subplot_ord += 1

    def tc_tp_fp_ex(self):
        for s in self.subdirs:
            model = self.metadata[s]["model"]
            df = tp_fp_ex_by_tc(s)
            yield model, df

    def plot_tc_tp_fp_ex(
        self,
        axes=None,
        *,
        stack=False,
        order=None,
        min_Tc=0.1,
        i18n=None,
        xlim=None,
        **kwargs,
    ):
        """Plot counts of TP, FP and EX againts Tc."""
        if axes is not None:
            axes = iter(axes.ravel())

        if stack and "color" not in kwargs:
            kwargs["color"] = ["#00aa00", "#ff0000", "#eeaa00"]

        subplot_ord = ord("A")
        models = dict(self.tc_tp_fp_ex())
        yy = ["TP", "FP * (-1)", "TP+EX"] if stack else ["TP", "FP", "EX"]
        if xlim is None:
            xlim = (1, min_Tc)

        for model in order if order else models.keys():
            df = models[model]
            if axes is not None:
                ax = next(axes)
                kwargs["ax"] = ax

            if min_Tc > 0:
                df = df[df.T_c > min_Tc]

            if stack:
                df = df.copy()
                df["FP * (-1)"] = -df.FP
                df["TP+EX"] = df.TP + df.EX

            ax = df.plot(
                x="T_c",
                y=yy,
                xlim=xlim,
                ylabel=T_(i18n, "count"),
                title=f"{chr(subplot_ord)}: {model}",
                **kwargs,
            )

            if stack:
                ax.axhline(0, color="gray", linestyle="dotted")
            # nice, but fails with sharey=True
            # ax.semilogy()
            subplot_ord += 1


class QualitySummary(Summary):
    """Like Summary, but not for baseline directory: pick subdirs with specific quality."""

    def __init__(
        self, reval_dir: str, quality: int, *, model_is_basename=False
    ) -> None:
        self.quality = quality
        super().__init__(reval_dir, model_is_basename=model_is_basename)

    def _post_init(self):
        self.subdirs = sorted(
            glob.glob(os.path.join(self.phys_dir, f"*/*_{self.quality:03d}/"))
        )
        if not self.subdirs:
            raise ValueError(
                f"No dump directories for {self.reval_dir=} and {self.quality=}"
            )
        self.metadata = {
            subdir: load_meta(subdir, model_is_basename=self.model_is_basename)
            for subdir in self.subdirs
        }

    def get_summaries(self, t_score=0):
        return [{**self.metadata[s], **get_summary(s, t_score)} for s in self.subdirs]


class GrandSummary:
    """For top level reval directory."""

    set_q_idx = False

    def __init__(self, reval_dir: str, model_is_basename=False) -> None:
        self.reval_dir = reval_dir
        self.model_is_basename = model_is_basename
        self.phys_dir = os.path.expanduser(reval_dir)
        self.subdirs = _get_model_subdirectories(self.phys_dir)
        self._model_APs = None
        self._model_PRF1s = None
        self._model_PRF1_t_score = None

    def __repr__(self):
        return f"GrandSummary('{self.reval_dir}')"

    def get_model_APs(self, model=None):
        if self._model_APs is None:
            self._model_APs = {df["model"].iat[0]: df for df in self.ap_summaries()}
        if model is None:
            return self._model_APs
        return self._model_APs[model]

    def get_model_PRF1s(self, t_score=0, model=None):
        if self._model_PRF1s is None or self._model_PRF1_t_score != t_score:
            self._model_PRF1s = {
                df["model"].iat[0]: df for df in self.q_summaries(t_score)
            }
            self._model_PRF1_t_score = t_score
        if model is None:
            return self._model_PRF1s
        return self._model_PRF1s[model]

    def q_summaries(self, t_score=0):
        """For each model subdirectory get a df with per-Q summary() results."""
        for s in self.subdirs:
            df = subdir_summaries(s, t_score)
            if self.model_is_basename:
                df["model"] = dirbasename(s)
            if self.set_q_idx:
                df = df.set_index("quality")
            yield df

    def ap_summaries(self):
        for s in self.subdirs:
            df = subdir_meta_df(s, self.model_is_basename)
            for col in df.columns:
                if col.startswith("AP"):
                    df[col] = df[col] / 100
            if self.set_q_idx:
                df = df.set_index("quality")
            yield df

    def save_q_summaries(self, excel=True):
        for df in self.q_summaries():
            model = df["model"].iat[0]
            if excel:
                out = os.path.join(self.phys_dir, f"{model}.xlsx")
                df.to_excel(out)
            else:
                out = os.path.join(self.phys_dir, f"{model}.csv")
                df.to_csv(out)
            logger.info(f"Saved Q summaries to: {out}")

    def plot_q_summaries(self, axes=None, order=None, i18n=None, t_score=0, **kwargs):
        if axes is not None:
            axes = iter(axes.ravel())
        subplot_ord = ord("A")
        models = self.get_model_PRF1s(t_score=t_score)
        for model in order if order else models.keys():
            df = models[model]
            if axes is not None:
                ax = next(axes)
                kwargs["ax"] = ax
            df.plot(
                x="quality",
                y=["PPV", "TPR", "F1"],
                ylim=(0, 1),
                ylabel=T_(i18n, "value"),
                xlabel=T_(i18n, "quality"),
                title=f"{chr(subplot_ord)}: {model}",
                **kwargs,
            )
            subplot_ord += 1

    def plot_ap_summaries(
        self, axes=None, by_size=False, relative=False, order=None, i18n=None, **kwargs
    ):
        if axes is not None:
            axes = iter(axes.ravel())

        if by_size and "color" not in kwargs:
            kwargs["color"] = ["#777777", "#444444", "#000000"]

        if not by_size and "color" not in kwargs:
            kwargs["color"] = ["#000000", "#005500", "#00aa00"]

        subplot_ord = ord("A")
        ylim = (0, 1)
        value_caption = "value"
        models = self.get_model_APs()
        columns = ["APl", "APm", "APs"] if by_size else ["AP", "AP75", "AP50"]

        if relative:
            ylim = (0, 100)
            value_caption = "% of max value"
            for df in models.values():
                for col in columns:
                    df[col] = df[col] / df[col].max() * 100

        for model in order if order else models.keys():
            df = models[model]
            if axes is not None:
                ax = next(axes)
                kwargs["ax"] = ax
            df.plot(
                x="quality",
                y=columns,
                ylim=ylim,
                ylabel=T_(i18n, value_caption),
                xlabel=T_(i18n, "quality"),
                title=f"{chr(subplot_ord)}: {model}",
                **kwargs,
            )
            subplot_ord += 1
            if by_size:
                # https://stackoverflow.com/questions/62470743/change-line-width-of-specific-line-in-line-plot-pandas-matplotlib
                w = 3
                for line in ax.get_lines():
                    line.set_linewidth(w)
                    w -= 1

    def plot_ap_derivatives(self, axes=None, order=None, i18n=None, smooth=5, **kwargs):
        if axes is not None:
            axes = iter(axes.ravel())
        subplot_ord = ord("A")
        models = {df["model"].iat[0]: df for df in self.ap_summaries()}
        for model in order if order else models.keys():
            df = models[model]
            if axes is not None:
                ax = next(axes)
                kwargs["ax"] = ax
            der = df.AP.rename("d(AP)/dQ").diff() * 100
            if smooth:
                der = der.rolling(smooth).mean()
            der.plot(
                xlabel=T_(i18n, "quality"),
                ylabel=T_(i18n, "d(AP)/dQ (p.p.)"),
                legend=False,
                title=f"{chr(subplot_ord)}: {model}",
                **kwargs,
            )
            subplot_ord += 1

    def plot_AP(self, model, metric="AP", ax=None, i18n=None, metric_in_name=False):
        df = self.get_model_APs(model)
        if not self.set_q_idx:
            df = df.set_index("quality")
        if metric_in_name:
            name = f"{model}.{metric}"
            ylabel = "value"
        else:
            name = model
            ylabel = metric
        df[metric].rename(name).plot(ax=ax)
        if ax:
            plt.sca(ax)
        plt.xlabel(T_(i18n, "quality"))
        plt.ylabel(T_(i18n, ylabel))
        plt.legend()


# region: matplotlib
def get_figure_axes(rows=2, cols=5, h=6, w=15, **kwargs):
    """Produce a default figure with subplots for 9 models."""
    fig, axes = plt.subplots(rows, cols, **kwargs)
    fig.set_figheight(h)
    fig.set_figwidth(w)
    return fig, axes


def finish_plot(fig, axes, legend_row=1, legend_col=4, clear_col=None, suptitle=None):
    """Layout and place legend for a figure created with get_figure_axes()."""
    if clear_col is None:
        clear_col = legend_col
    for col in range(clear_col, legend_col + 1):
        axes[legend_row, col].axis("off")
    axes[legend_row, legend_col].legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="lower right",
        fontsize="x-large",
        borderpad=2,
    )
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()


def save_plot(fig, name, v=0, c=0, p=0):
    # https://stackoverflow.com/questions/10784652/png-options-to-produce-smaller-file-size-when-using-savefig
    if fig is None:
        fig = plt.gcf()
    fig.savefig(f"{name}.png")
    os.system(f"mogrify {'-verbose' if v else ''} +dither -colors 256 {name}.png")
    os.system(f"optipng {'' if v else '-quiet'} -o5 {name}.png")

    print(f"Saved {name}.png")
    if p:
        fig.savefig(f"{name}.pdf")
        print(f"  and {name}.pdf")
    if c:
        plt.close()


# endregion


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


# region: latex tables

# region: new style OO tables
class Column:
    format = "{}"
    align = "r"

    def __init__(self, key, caption, bad=False):
        self.key = key
        self.caption = caption
        self.bad = bad
        self.max1 = None
        self.max2 = None

    def _get_value(self, row):
        return row[self.key]

    def learn(self, data):
        values = sorted(set(self._get_value(row) for row in data), reverse=not self.bad)
        self.max1 = values[0]
        if len(values) > 1:
            self.max2 = values[1]

    def render(self, row):
        value = self._get_value(row)
        raw_repr = self.format.format(value)
        if value == self.max1:
            return r"\textbf{" + raw_repr + "}"
        if value == self.max2:
            return r"\underline{" + raw_repr + "}"
        return raw_repr


class RawPercent(Column):
    """Floats from 0.0 to 100.0."""

    format = "{:.1f}"
    align = "c"


class Percent(RawPercent):
    """Floats from 0.0 to 1.0, which should show as percent."""

    def _get_value(self, row):
        return row[self.key] * 100


class RowHeader(Column):
    align = "l"

    def learn(self, data):
        pass

    def _get_value(self, row):
        return row[self.key].replace("_", r"\_")


class Table:
    hdrfile = sys.stdout

    def __init__(
        self, *columns: Column, header=True, long=False, mark_best=True, groups=None
    ) -> None:
        self.columns = columns
        self.header = header
        self.long = long
        self.mark_best = mark_best
        self.groups = groups

    @property
    def tabular(self):
        return "longtable" if self.long else "tabular"

    @property
    def alignments(self):
        """Generate latex column alignment, possibly with verical lines."""
        alignments = "".join(c.align for c in self.columns)
        if not self.groups:
            return alignments
        split = []
        ia = iter(alignments)
        for g in self.groups:
            for _ in range(g):
                split.append(next(ia))
            split.append("|")
        split.extend(ia)
        return "".join(split)

    def _header(self):
        if not self.header:
            return

        print(
            "\\begin{" + self.tabular + "}{" + self.alignments + "} \\toprule",
            file=self.hdrfile,
        )
        headings = " & ".join(c.caption for c in self.columns)
        print(headings + r" \\ \midrule", file=self.hdrfile)

    def _footer(self):
        if not self.header:
            return
        print("\\bottomrule\n\\end{" + self.tabular + "}", file=self.hdrfile)

    def render(self, data, outfile: Optional[TextIO] = None):
        if self.mark_best:
            for col in self.columns:
                col.learn(data)
        outfile = outfile or sys.stdout
        hdrbackup = self.hdrfile
        self.hdrfile = outfile
        self._header()
        for row in data:
            line = " & ".join(col.render(row) for col in self.columns)
            print(line + r" \\", file=outfile)
        self._footer()
        self.hdrfile = hdrbackup


PRF_COLUMNS = [
    RowHeader("model", "Model"),
    Percent("precision", r"PPV\,\%"),
    Percent("recall", r"TPR\,\%"),
    Percent("f1", r"F1\,\%"),
    Column("tp", "TP"),
    Column("fp", "FP", bad=True),
    Column("ex", "EX"),
]


def baseline_table_prf_OO(reval_dir, header=True):
    metrics = load_rich_results(reval_dir)
    table = Table(*PRF_COLUMNS, header=header)
    table.render(metrics)


PRF_COLUMNS_QS = [
    RowHeader("model", "Model"),
    Percent("PPV", r"PPV\,\%"),
    Percent("TPR", r"TPR\,\%"),
    Percent("F1", r"F1\,\%"),
    Column("tp", "TP"),
    Column("fp", "FP", bad=True),
    Column("ex", "EX"),
]


def prf_table_for_quality_OO(
    reval_dir, quality, t_score=0.5, header=True, model_is_basename=False
):
    summary = QualitySummary(reval_dir, quality, model_is_basename=model_is_basename)
    table = Table(*PRF_COLUMNS_QS, header=header)
    table.render(summary.get_summaries(t_score))


def baseline_table_ap_OO(reval_dir, header=True):
    metrics = load_rich_results(reval_dir)
    table = Table(
        Column("quality", "Q"),
        RawPercent("AP", r"AP"),
        RawPercent("AP50", r"AP\tsub{50}"),
        RawPercent("AP75", r"AP\tsub{75}"),
        RawPercent("APl", r"AP\tsub{l}"),
        RawPercent("APm", r"AP\tsub{m}"),
        RawPercent("APs", r"AP\tsub{s}"),
        header=header,
    )
    table.render(metrics)


def by_quality_table_OO(
    model_dir,
    header=True,
    t_score=0.5,
    filename: str = None,
    skip: int = None,
    long=True,
):
    """Replacement for baseline_table() when used on a model directory (by quality)."""
    data = subdir_summaries_with_ap(
        model_dir, t_score, model_is_basename=False, raw=True
    )
    if skip:
        data = data[::skip]
    table = Table(
        Column("quality", "Q"),
        RawPercent("AP", r"AP"),
        RawPercent("AP50", r"AP\tsub{50}"),
        RawPercent("AP75", r"AP\tsub{75}"),
        RawPercent("APl", r"AP\tsub{l}"),
        RawPercent("APm", r"AP\tsub{m}"),
        RawPercent("APs", r"AP\tsub{s}"),
        Percent("PPV", r"PPV"),
        Percent("TPR", r"TPR"),
        Column("TP", "TP"),
        Column("FP", "FP", bad=True),
        Column("EX", "EX"),
        header=header,
        long=long,
        mark_best=False,
        groups=[1, 3, 3, 2],
    )
    if filename:
        with open(filename, "w") as outfile:
            table.render(data, outfile)
    else:
        table.render(data)


# endregion

# region: baseline_table() -- old style
def _table_xcol(metrics):
    """Find if this is a collection of same quality, or model.
    Return the variable attribute (X column ~ row header)."""
    if len(set(d["quality"] for d in metrics)) == 1:
        return "{model} & "
    return "{quality} & "


def _table_xhdr(metrics):
    """Find if this is a collection of same quality, or model.
    Return the row headers' column header."""
    if len(set(d["quality"] for d in metrics)) == 1:
        return "Model & "
    return "Q & "


TABLE_FORMAT = (
    "{AP:.1f} & {AP50:.1f} & {AP75:.1f} & {APl:.1f} & {APm:.1f}"
    " & {APs:.1f} & {recall:.1f} & {precision:.1f} & {tp} & {fp}"
)

TABLE_HEADINGS = (
    r"AP & AP\tsub{50} & AP\tsub{75} & AP\tsub{l} & AP\tsub{m}"
    r" & AP\tsub{s} & TPR & PPV & TP & FP"
)


def baseline_table(reval_dir, header=False):
    # previously as evaluate_coco.py side-effect
    metrics = load_rich_results(reval_dir)
    hav_ex = all("ex" in d for d in metrics)
    x_col = _table_xcol(metrics)
    fmt = x_col + TABLE_FORMAT + (r" & {ex:} \\" if hav_ex else r" \\")
    kind = "tabular" if len(metrics) <= 20 else "longtable"
    cols = "lccccccccrr" if len(metrics) <= 20 else "l|ccc|ccc|cc|rr"
    if not hav_ex:
        print("Warning: no EX column available.")
        head = "\\begin{" + kind + "}{" + cols + "} \\toprule"
        headings = TABLE_HEADINGS + r" \\ \midrule"
    else:
        head = "\\begin{" + kind + "}{" + cols + "r} \\toprule"
        headings = TABLE_HEADINGS + r" & EX \\ \midrule"

    if header:
        # \newcommand\tsub[1]{\textsubscript{#1}}
        print(head, file=sys.stderr)
        print(_table_xhdr(metrics) + headings, file=sys.stderr, flush=True)

    for row in metrics:
        row["precision"] *= 100
        row["recall"] *= 100
        row["model"] = row["model"].replace("_", r"\_")  # LaTeX excape
        print(fmt.format(**row))

    if header:
        sys.stdout.flush()
        print("\\bottomrule\n\\end{" + kind + "}", file=sys.stderr)


# endregion

# region: baseline_table_prf() -- old style
TABLE_FORMAT_PRF = "{recall:.1f} & {precision:.1f} & {f1:.1f} & {tp:} & {fp:}"


def baseline_table_prf(reval_dir, header=False):
    # previously as evaluate_coco.py side-effect
    metrics = load_rich_results(reval_dir)
    hav_ex = all("ex" in d for d in metrics)
    x_col = _table_xcol(metrics)
    fmt = x_col + TABLE_FORMAT_PRF + (r" & {ex} \\" if hav_ex else r" \\")

    if not hav_ex:
        print("Warning: no EX column available.")
        head = "\\begin{tabular}{lcccrr} \\toprule"
        headings = r"TPR\,\% & PPV\,\% & F1\,\% & TP & FP \\ \midrule"
    else:
        head = "\\begin{tabular}{lcccrrr}\n\\toprule"
        headings = r"TPR\% & PPV\% & F1\% & TP & FP & EX \\ \midrule"

    if header:
        # \newcommand\tsub[1]{\textsubscript{#1}}
        print(head, file=sys.stderr)
        print(_table_xhdr(metrics) + headings, file=sys.stderr, flush=True)

    for row in metrics:
        row["precision"] *= 100
        row["recall"] *= 100
        row["f1"] *= 100
        row["model"] = row["model"].replace("_", r"\_")  # LaTeX excape
        print(fmt.format(**row))

    if header:
        print("\\bottomrule\n\\end{tabular}", file=sys.stderr)


# endregion
# region: baseline_table_ap() -- old style
TABLE_FORMAT_AP = (
    r"{AP:.1f} & {AP50:.1f} & {AP75:.1f} & {APl:.1f} & {APm:.1f} & {APs:.1f} \\"
)


def baseline_table_ap(reval_dir, header=False):
    # previously as evaluate_coco.py side-effect
    metrics = load_rich_results(reval_dir)
    x_col = _table_xcol(metrics)
    fmt = x_col + TABLE_FORMAT_AP

    head = "\\begin{tabular}{lcccccc}\n\\toprule"
    headings = r"AP & AP\tsub{50} & AP\tsub{75} & AP\tsub{l} & AP\tsub{m} & AP\tsub{s} \\ \midrule"

    if header:
        # \newcommand\tsub[1]{\textsubscript{#1}}
        print(head, file=sys.stderr)
        print(_table_xhdr(metrics) + headings, file=sys.stderr, flush=True)

    for row in metrics:
        row["model"] = row["model"].replace("_", r"\_")  # LaTeX excape
        print(fmt.format(**row))

    if header:
        print("\\bottomrule\n\\end{tabular}", file=sys.stderr)


# endregion


def baseline_table_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "reval_dir",
        help="directory to summare with DIRECT evaluator_dump_(...) subdirectories",
    )
    parser.add_argument(
        "--header", "-g", action="store_true", help="print LaTeX table header"
    )
    parser.add_argument(
        "--prf", action="store_true", help="print only T_c dependent metrics."
    )
    parser.add_argument(
        "--ap", action="store_true", help="print only T_c independent metrics."
    )

    args = parser.parse_args()
    if args.prf:
        baseline_table_prf(args.reval_dir, args.header)
    elif args.ap:
        baseline_table_ap(args.reval_dir, args.header)
    else:
        baseline_table(args.reval_dir, args.header)


# endregion


# region: Q subdirectories
def symlink_by_quality(reval_dir: str):
    reval_dir = os.path.abspath(os.path.expanduser(reval_dir))
    subdirs = _get_model_subdirectories(reval_dir)
    dumps = [
        os.path.abspath(g)
        for subdir in subdirs
        for g in glob.glob(os.path.join(subdir, "*/"))
    ]
    key = operator.itemgetter(slice(-3, None))
    for k, v in itertools.groupby(sorted(dumps, key=key), key=key):
        by_q_dir = os.path.join(reval_dir, f"quality_{k}")
        os.makedirs(by_q_dir, exist_ok=True)
        for dump in v:
            symlink = os.path.join(by_q_dir, os.path.basename(dump))
            target = os.path.relpath(dump, by_q_dir)
            if os.path.exists(symlink):
                logger.warn(f"Exists: {symlink}")
                continue
            logger.info(f"Symlinking {symlink} -> {target}")
            os.symlink(target, symlink)


def symlink_q_main() -> None:
    parser = argparse.ArgumentParser("symlink_q")
    parser.add_argument("reval_dir")
    args = parser.parse_args()
    symlink_by_quality(args.reval_dir)


def _get_quality_subdirectories(top_dir):
    return sorted(glob.glob(os.path.join(top_dir, "quality_*/")), reverse=True)


# endregion

# region: PDF Book plots, 1 page per Q
def plot_book(reval_dir, step=1):
    from matplotlib.backends.backend_pdf import PdfPages

    reval_dir = os.path.expanduser(reval_dir)
    pdf_output = os.path.join(reval_dir, "counts_vs_Tc_by_Q.pdf")
    ylim = None
    with PdfPages(pdf_output) as pdf:
        for subdir in _get_quality_subdirectories(reval_dir)[::step]:
            s = Summary(subdir)
            fig, axes = get_figure_axes(sharey=True)
            if ylim:
                axes[0, 0].set_ylim(ylim)
            s.plot_tc_tp_fp_ex(
                axes, stack=True, min_Tc=0.15, order=DEFAULT_ORDER, legend=False
            )
            finish_plot(fig, axes, dirbasename(subdir))
            if ylim is None:
                ylim = axes[0, 0].get_ylim()
            pdf.savefig(fig)
            plt.close(fig)


def plot_PRF1_book(reval_dir, step=1):
    from matplotlib.backends.backend_pdf import PdfPages

    reval_dir = os.path.expanduser(reval_dir)
    pdf_output = os.path.join(reval_dir, "PRF1_vs_Tc_by_Q.pdf")
    with PdfPages(pdf_output) as pdf:
        for subdir in _get_quality_subdirectories(reval_dir)[::step]:
            s = Summary(subdir)
            fig, axes = get_figure_axes()
            s.plot_tc_summaries(axes, order=DEFAULT_ORDER, legend=False)
            finish_plot(fig, axes, dirbasename(subdir))
            pdf.savefig(fig)
            plt.close(fig)


def _plot_book() -> None:
    parser = argparse.ArgumentParser("plot_book")
    parser.add_argument("reval_dir")
    args = parser.parse_args()
    plot_book(args.reval_dir)


# endregion


def gt_for_single_run(subdir: str):
    dr = DetectionResults(subdir)
    meta = load_meta(subdir)
    summary = pd.DataFrame(
        [det for det in dr.detections if "gt_id" in det],
        columns="image_id score iou gt_id category".split(),
    )
    summary["category"] = summary.category.astype("category")  # what a coincidence...
    summary["model"] = meta["model"]
    summary["quality"] = meta["quality"]
    summary["crowd"] = summary.gt_id > CROWD_ID_T
    return summary


@cached_directory_data(compress=True)
def gt_id_statistics(reval_dir: str):
    try:
        from multiprocess import Pool
    except ImportError:
        from multiprocessing import Pool

    data = []

    with Pool() as p:
        for model_dir in _get_model_subdirectories(reval_dir):
            logger.info(f"{datetime.now()}: {model_dir=}")
            subdirs = sorted(glob.glob(os.path.join(model_dir, "*/")))
            chunk = p.map(gt_for_single_run, subdirs)
            data.extend(chunk)

    return pd.concat(data, ignore_index=True)


if __name__ == "__main__":
    baseline_table_main()
