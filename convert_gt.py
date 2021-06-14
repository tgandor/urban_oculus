from argparse import ArgumentParser
from itertools import count
import json
from pathlib import Path

from evaldets.results import _load_gt

parser = ArgumentParser()
parser.add_argument("--bbox", action="store_true")
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--limit", "-l", type=int, default=0)
parser.add_argument("--output", "-o", type=Path, default="gt_objects.json")
parser.add_argument("--indent", type=int, default=2)
args = parser.parse_args()

gt = _load_gt("coco_2017_val", debug=0)

result = []

for new_id, obj in zip(count(1), gt.dataset["annotations"]):
    obj["area"] = round(obj["area"], 1)
    obj["category"] = gt.cats[obj["category_id"]]["name"]
    del obj["category_id"]
    if args.bbox:
        obj["x"], obj["y"], obj["w"], obj["h"] = obj["bbox"]
        del obj["bbox"]

    if args.verbose:
        print(obj)

    result.append(obj)

    if new_id == args.limit:
        break

with args.output.open("w") as jsf:
    json.dump(result, jsf, indent=args.indent)
    jsf.write("\n")
