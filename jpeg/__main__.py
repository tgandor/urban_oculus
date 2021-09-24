import argparse
import collections
import glob
import pprint

from . import get_QTs, recognize_QT_quality, reorder
from .quantization.ijg_tables import Q_to_QT1, Q_to_QT2

parser = argparse.ArgumentParser("")
parser.add_argument("filenames", nargs="*")
parser.add_argument(
    "--show-qts", type=int, help="a subcommand: only print specified quality's QTs."
)
parser.add_argument("--reorder", "-r", action="store_true", help="render 2d QTs")
parser.add_argument("--stats", "-s", action="store_true")
args = parser.parse_args()

stats = collections.Counter()

for filename in (
    # good not only for windows, also big globs on Linux (cli limits).
    fn for p in args.filenames for fn in (glob.glob(p) if "*" in p else [p])
):
    q = recognize_QT_quality(filename, failsafe=True)
    stats[q] += 1
    if args.stats:
        continue
    print(filename)
    for qt in get_QTs(filename):
        if args.reorder:
            qt = pprint.pformat(reorder(qt))
        print(qt)
    print(f"Recognized Q: {q}")

if args.stats:
    for val, count in stats.most_common():
        print(f"{count:,} x Q = {val}")
    print(f"Total: {sum(stats.values()):,}")

if args.show_qts is not None:
    q1 = Q_to_QT1[args.show_qts]
    q2 = Q_to_QT2[args.show_qts]

    if args.reorder:
        q1 = pprint.pformat(reorder(q1))
        q2 = pprint.pformat(reorder(q2))

    print(q1)
    print(q2)
