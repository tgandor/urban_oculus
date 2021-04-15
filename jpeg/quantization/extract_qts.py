#!/usr/bin/env python

from types import SimpleNamespace as ns
import argparse
import os
import pprint
import shutil

from tqdm import trange
import pandas as pd

from jpeg import get_QTs, get_QTs2d

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument("--minQ", type=int, help="min JPEG quality", default=1)
parser.add_argument("--maxQ", type=int, help="min JPEG quality", default=100)
parser.add_argument("--use-opencv", "-cv2", action="store_true")
args = parser.parse_args()

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


def opencv_degrade(orig, filename, q, grayscale=False):
    import cv2
    img = cv2.imread(
        orig,
        cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED,
    )
    cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, q])


def mogrify_degrade(orig, filename, q, grayscale=False):
    gray = "-type Grayscale" if grayscale else ""
    shutil.copy(orig, filename)
    os.system(f"mogrify {verb} {gray} -quality {q} {filename}")


def gen_qts(args):
    degrade = opencv_degrade if args.use_opencv else mogrify_degrade

    for q in rgen(args.minQ, args.maxQ+1):
        degrade(args.filename, filename, q)
        qts = get_QTs2d(filename)
        hex_qts = get_QTs(filename)

        degrade(args.filename, filename, q, True)
        qt = get_QTs2d(filename)[0]
        check(q, qt, qts)

        yield ns(
            q=q,
            qts=qts,
            hex_qts=hex_qts,
        )


def save_qts_txt(q, qts):
    with open(f"qts_{q:03d}.txt", "w") as qtxt:
        for qt in qts:
            for row in qt:
                print(''.join(f"{x:4d}" for x in row), file=qtxt)


def main():
    with open("README.md", "w") as out_md, open('ijg_tables.py', 'w') as out_py:
        out_md.write(HEADER)
        out_py.write(PY_HEADER)

        for x in gen_qts(args):
            out_md.write(TEMPLATE.format(
                q=x.q,
                qt1_table=qt_to_md(x.qts[0]),
                qt2_table=qt_to_md(x.qts[1]),
            ))

            out_py.write(PY_TEMPLATE.format(
                q=x.q,
                qt1=repr(x.hex_qts[0]),
                qt2=repr(x.hex_qts[1]),
            ))

            save_qts_txt(x.q, x.qts)


# __DATA__ ;)

PY_HEADER = """
Q_to_QT1 = {}
Q_to_QT2 = {}
QT1_to_Q = {}
# non-unique: Q = (1, 2, 3) all have 64 x "ff"
QT2_to_Q = {}
"""

PY_TEMPLATE = """

Q_to_QT1[{q}] = {qt1}
Q_to_QT2[{q}] = {qt2}
QT1_to_Q[{qt1}] = {q}
QT2_to_Q[{qt2}] = {q}
"""

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

if __name__ == '__main__':
    main()
