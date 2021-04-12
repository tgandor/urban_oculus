import argparse

from . import get_QTs
from .quantization.ijg_tables import Q_to_QT1, Q_to_QT2

parser = argparse.ArgumentParser()
parser.add_argument('filenames', nargs='*')
parser.add_argument('--show-qts', type=int)
args = parser.parse_args()

for filename in args.filenames:
    print(filename)
    for qt in get_QTs(filename):
        print(qt)

if args.show_qts is not None:
    print(Q_to_QT1[args.show_qts])
    print(Q_to_QT2[args.show_qts])
