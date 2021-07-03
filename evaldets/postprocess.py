import argparse
import glob
import os
import warnings

from uo.utils import load


class Summary:
    def __init__(self, reval_dir: str) -> None:
        self.reval_dir = reval_dir
        self.phys_dir = os.path.expanduser(reval_dir)
        self.subdirs = sorted(glob.glob(os.path.join(self.phys_dir, "*")))


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
