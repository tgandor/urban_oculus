#!/usr/bin/env python

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
NL = "\n"

HEADER = f"""\
# JPEG quantization tables

Quantization tables used in JPEG compression for Q = 1, ..., 100

## TOC

Links to specific quality:

{"|" + 10 * "             |"}
{"|" + 10 * ":-----------:|"}
{NL.join(
    "| " +
    " | ".join("%11s" % f"[{10*row+col+1}](#{10*row+col+1})" for col in range(10)) +
    " |"
    for row in range(10)
)}
"""

TEMPLATE = """
## {q}

<table>
<tr><th>QT1 (luma or mono)</th><th>QT2 (chroma)</th></tr>

<tr><td>

{qt1_table}

</td><td>

{qt2_table}

</td></tr>
</table>

[TOC](#toc)
"""

verb = '-verbose'
verb = ''
rgen = range if verb else trange


def check(q, qt, qts):
    assert qts[0] == qt, "Mono QT should be same as color QT1"
    if qts[0] != qt:
        print('For quality', q, '(QT1)')
        pprint.pprint(qts[0])
        print('vs')
        pprint.pprint(qt)
        print('---')


def qt_to_md(qt, html=False):
    df = pd.DataFrame(qt)
    if html:
        df.columns.name = 'u'
        df.index.name = 'v'
        return df.to_html()
    df.index.name = 'u&#8594; <br> v&#8595;'
    return df.to_markdown()


with open("README.md", "w") as out_md:
    out_md.write(HEADER)

    for q in rgen(args.minQ, args.maxQ+1):
        shutil.copy(args.filename, filename)
        os.system(f"mogrify {verb} -quality {q} {filename}")
        qts = get_QTs2d(filename)
        shutil.copy(args.filename, filename)
        os.system(f"mogrify {verb} -type Grayscale -quality {q} {filename}")
        qt = get_QTs2d(filename)[0]
        check(q, qt, qts)
        out_md.write(TEMPLATE.format(
            q=q,
            qt1_table=qt_to_md(qts[0]),
            qt2_table=qt_to_md(qts[1]),
        ))

        with open(f"qts_{q:03d}.txt", "w") as qtxt:
            for qt in qts:
                for row in qt:
                    print(''.join(f"{x:4d}" for x in row), file=qtxt)
