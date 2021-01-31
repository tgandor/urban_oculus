import argparse
import os

from tqdm import tqdm

from couch import db
from jpeg import get_QTs, identify_quality

parser = argparse.ArgumentParser()
parser.add_argument('filenames', nargs='+')
parser.add_argument('--type', default='JpgImg')
parser.add_argument('--dataset')
args = parser.parse_args()

for filename in tqdm(args.filenames):
    data = {
        "type": args.type,
        "name": os.path.basename(filename),
        "quantization": get_QTs(filename),
        "quality": identify_quality(filename),
    }

    if args.dataset:
        data["dataset"] = args.dataset

    db.save(data)
