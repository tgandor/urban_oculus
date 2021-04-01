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

HEADER = """
# JPEG quantization tables

Below are the quantization tables used in JPEG compression for Q parameter = 1, ..., 100
"""

verb = '-verbose'
verb = ''
rgen = range if verb else trange

for q in rgen(args.minQ, args.maxQ+1):
    shutil.copy(args.filename, filename)
    os.system(f"mogrify {verb} -quality {q} {filename}")
    qts = get_QTs2d(filename)
    shutil.copy(args.filename, filename)
    os.system(f"mogrify {verb} -type Grayscale -quality {q} {filename}")
    qt = get_QTs2d(filename)[0]
    assert qts[0] == qt, "Mono QT should be same as color QT1"
    if qts[0] != qt:
        print('For quality', q, '(QT1)')
        pprint.pprint(qts[0])
        print('vs')
        pprint.pprint(qt)
        print('---')
