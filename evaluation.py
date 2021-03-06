import argparse
import json
from pathlib import Path
import sys
import time


from evaldets.api import DetectionResults


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("detection_files", nargs="+")
    parser.add_argument("--gt-match", "-g", action="store_true")
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--no-cache", "-C", action="store_false")
    parser.add_argument("--output", "-o", default="detections.json", type=Path)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    detections = []

    for detection_file in args.detection_files:
        res = DetectionResults(
            detection_file,
            cache=args.no_cache,
            debug=int(args.verbose),
            gt_match=args.gt_match,
            iou_thresh=args.min_iou,
        )
        for d in res.detections:
            if args.verbose:
                print(d)
            detections.append(d)

    with args.output.open("w") as jsf:
        json.dump(detections, jsf, indent=2)


if __name__ == "__main__":
    start = time.time()
    try:
        _main()
    finally:
        print(f"Done: {time.time()-start:.3f} s", file=sys.stderr)
