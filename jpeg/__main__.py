import argparse
import pprint

from . import get_QTs, recognize_QT_quality, reorder
from .quantization.ijg_tables import Q_to_QT1, Q_to_QT2

parser = argparse.ArgumentParser("")
parser.add_argument("filenames", nargs="*")
parser.add_argument("--show-qts", type=int)
parser.add_argument("--reorder", "-r", action="store_true", help="render 2d QTs")
args = parser.parse_args()

for filename in args.filenames:
    print(filename)
    for qt in get_QTs(filename):
        if args.reorder:
            qt = pprint.pformat(reorder(qt))
        print(qt)
    print(f"Recognized Q: {recognize_QT_quality(filename)}")

if args.show_qts is not None:
    print(Q_to_QT1[args.show_qts])
    print(Q_to_QT2[args.show_qts])
