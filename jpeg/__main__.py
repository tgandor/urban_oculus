import argparse

from . import get_QTs

parser = argparse.ArgumentParser()
parser.add_argument('filenames', nargs='+')
args = parser.parse_args()

for filename in args.filenames:
    print(filename)
    for qt in get_QTs(filename):
        print(qt)
