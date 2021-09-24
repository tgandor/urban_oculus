#!/usr/bin/env python

import os
import pathlib
import shutil

png_file = pathlib.Path("~/datasets/coco/train2017/000000320612.jpg").expanduser()

if not png_file.exists():
    print(f'Not found: {png_file}')
    exit()

with png_file.open("rb") as png:
    header = png.read(6)

if header != b'\x89PNG\r\n':
    print(f"Not a PNG: {png_file}")
    exit()

bak_dir = pathlib.Path("~/datasets/coco/png_bak").expanduser()
bak_dir.mkdir(exist_ok=True)
target = bak_dir / '000000320612.png'

shutil.move(png_file, target)
command = f"convert -verbose -quality 96 {target} {png_file}"
print(command)
os.system(command)
