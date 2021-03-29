import argparse
import os
import pprint
import shutil

from tqdm import trange

from jpeg import get_QTs2d

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument("--minQ", type=int, help="min JPEG quality", default=1)
parser.add_argument("--maxQ", type=int, help="min JPEG quality", default=100)
args = parser.parse_args()

filename = 'temp_for_qt.jpg'

for q in trange(args.minQ, args.maxQ+1):
    shutil.copy(args.filename, filename)
    os.system(f"mogrify -verbose -quality {q} {filename}")
    qts = get_QTs2d(filename)
    pprint.pprint(qts)
