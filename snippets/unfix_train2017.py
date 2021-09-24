#!/usr/bin/env python

import pathlib
import shutil

bak_dir = pathlib.Path("~/datasets/coco/png_bak").expanduser()
bak_dir.mkdir(exist_ok=True)
bak_file = bak_dir / '000000320612.png'

target = pathlib.Path("~/datasets/coco/train2017/000000320612.jpg").expanduser()

if not bak_file.exists():
    print(f'Not found: {bak_file}')
    exit()

with bak_file.open("rb") as png:
    header = png.read(6)

if header != b'\x89PNG\r\n':
    print(f"Not a PNG: {bak_file}")
    exit()

print(f"Copying: {bak_file} -> {target}")
shutil.copy(bak_file, target)
