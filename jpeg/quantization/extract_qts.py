import argparse
import os
import pprint
import shutil

from tqdm import trange
import pandas as pd

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

<table>
<tr><th>Table 1 Heading 1 </th><th>Table 1 Heading 2</th></tr>
<tr><td>

|Table 1| Middle | Table 2|
|--|--|--|
|a| not b|and c |

</td><td>

|b|1|2|3|
|--|--|--|--|
|a|s|d|f|

</td></tr> </table>

"""

verb = '-verbose'
verb = ''
rgen = range if verb else trange

with open("README.md", "w") as out_md:
    out_md.write(HEADER)

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
