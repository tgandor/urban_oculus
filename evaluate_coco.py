import argparse
import glob
import json
from operator import itemgetter
import os

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from imports import load

IoU_T_IDX = 0  # first IoU threshold = 0.5
MAXDET_IDX = -1  # last "maxDets"
AREARNG_IDX = 0  # 'all'

ANNOTATIONS = "datasets/coco/annotations/instances_val2017.json"
# same as:
# from detectron2.data import MetadataCatalog
# ANNOTATIONS = MetadataCatalog.get('coco_2017_val').json_file


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "detection_files",
        nargs="+",
        help="path to coco_instances_results.json[.gz|.bz2|], or folder",
    )
    parser.add_argument(
        "--min-score", "-t", type=float, help="confidence threshold for detections"
    )
    parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=3,
        help="how many best and worst classes to report",
    )
    parser.add_argument(
        "--full", "-f", action="store_true", help="perform full accumulate / summarize"
    )
    return parser.parse_args()


def load_gt():
    if not os.path.exists(ANNOTATIONS):
        print(
            "Please symlink datasets/ or unzip annotations_trainval2017.zip to datasets/coco/"
        )
        exit()

    return COCO(ANNOTATIONS, debug=0)


def load_detections_results(file_or_dir):
    if os.path.isdir(file_or_dir):
        dump_dir = file_or_dir
        det_filename = glob.glob(
            os.path.join(dump_dir, "coco_instances_results.json*")
        )[0]
    else:
        dump_dir = os.path.dirname(file_or_dir)
        det_filename = file_or_dir

    results_file = glob.glob(os.path.join(dump_dir, "results.json*"))[0]

    results = load(results_file)

    detections = load(det_filename)

    print(f'Loaded dets by {results["model"]}, count = {len(detections):,}.')

    return detections, results, dump_dir, det_filename


args = parse_cli()
gt = load_gt()


metrics = []

for df in args.detection_files:
    print(f"Processing {df}")

    detections, results, dump_dir, det_filename = load_detections_results(df)

    if args.min_score:
        detections = [d for d in detections if d["score"] > args.min_score]
        print(f"Filtered with T={args.min_score} to {len(detections)} dets.")

    min_score = min(d["score"] for d in detections)

    dt = gt.loadRes(detections)

    coco = COCOeval(gt, dt, iouType="bbox")

    if not args.full:
        # don't evalImage for 'small', 'medium', 'large'
        coco.params.areaRng = [[0.0, 1e9]]

    coco.evaluate()

    tp = 0
    fp = 0
    n_ign = 0
    n_gt = 0

    nCats = len(coco.params.catIds)
    nArea = len(coco.params.areaRng)
    nImgs = len(coco.params.imgIds)

    assert len(coco.evalImgs) == nCats * nArea * nImgs

    print(f"nArea: {nArea}, len(evalImgs): {len(coco.evalImgs)}")
    for catIx in range(nCats):
        offs = catIx * (nArea * nImgs)
        for ix, img in enumerate(coco.evalImgs[offs : offs + nImgs]):  # noqa
            if img is None:
                continue

            ign = img["dtIgnore"][IoU_T_IDX]
            mask = ~ign
            n_ignored = ign.sum()
            n_ign += n_ignored
            tp += (img["dtMatches"][IoU_T_IDX][mask] > 0).sum()
            fp += (img["dtMatches"][IoU_T_IDX][mask] == 0).sum()
            n_gt += len(img["gtIds"]) - img["gtIgnore"].astype(int).sum()

    recall = tp / n_gt
    precision = tp / (tp + fp)

    assert tp + fp + n_ign == len(
        detections
    ), f"TP/{tp}/ + FP/{fp}/ +I/{n_ign}/ == {tp+fp} != |D| /{len(detections)}/"
    f1 = 2 * precision * recall / (precision + recall)

    print(f"Total objects found: {tp:,} (of {n_gt:,} GT, {n_ign:,} ignored, {fp:,} FP)")
    print(f"precision {precision*100:.1f} recall {recall*100:.1f} f1 score: {f1*100:.1f}")

    model = results["model"].replace("_", r"\_")
    ap = results["results"]["bbox"]["AP"]
    ap50 = results["results"]["bbox"]["AP50"]
    ap75 = results["results"]["bbox"]["AP75"]
    apl = results["results"]["bbox"]["APl"]
    apm = results["results"]["bbox"]["APm"]
    aps = results["results"]["bbox"]["APs"]

    print(
        results["model"],
        {
            k: np.round(v, 1)
            for k, v in results["results"]["bbox"].items()
            if "-" not in k
        },
    )

    if args.full:
        coco.accumulate()
        coco.summarize()
        raw_rc = coco.eval["recall"][IoU_T_IDX, :, AREARNG_IDX, MAXDET_IDX]
        e_recall = np.mean(raw_rc[raw_rc > -1])
        print(f"Recall by .eval: {e_recall}")
        new_results = dict(
            zip(results["results"]["bbox"].keys(), (100 * coco.stats[:6]).round(1))
        )
        print("New results by coco.stats =", new_results)
        ap, ap50, ap75, aps, apm, apl = new_results.values()

    classes = sorted(
        [(v, k) for k, v in results["results"]["bbox"].items() if "-" in k],
        reverse=True,
    )
    print(
        f"Top {args.top} classes (orig results):\n ",
        ",\n  ".join(f"{v:4.1f} = {k}" for v, k in classes[: args.top]),
    )
    print(
        "Bottom {args.top} classes (orig results):\n ",
        ",\n  ".join(f"{v:4.1f} = {k}" for v, k in classes[-args.top :]),  # noqa
    )

    metrics.append(
        [
            model,
            ap,
            ap50,
            ap75,
            apl,
            apm,
            aps,
            100 * recall,
            100 * precision,
            tp,
            fp,
        ]
    )

    with open(".evaluate_log.txt", "a") as log:
        print(
            f"{model:10s}: TP {tp:,} (GT {n_gt:,}, FP {fp:,}), "
            f"PPV {precision*100:.1f} TPR {recall*100:.1f} F1 {f1*100:.1f} - {det_filename}",
            file=log,
        )
        if args.full:
            print("New Results: ", new_results, file=log)

    new_results_file = os.path.join(dump_dir, "rich_results.json")
    bbox = {k: v for k, v in results["results"]["bbox"].items() if "-" not in k}
    rich_results = {
        "quality": results["quality"],
        "model": results["model"],
        "elapsed": results["elapsed"],
        # TypeError: Object of type int64 is not JSON serializable...
        "tp": int(tp),
        "fp": int(fp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "min_score": min_score,
        "score_T": args.min_score,
        **bbox,
    }
    # import code; code.interact(local=locals())
    with open(new_results_file, "w") as jsf:
        json.dump(rich_results, jsf)

    print("-" * 79)

# Summary ("bestAP" table)
# Remember to define:
# \newcommand\tsub[1]{\textsubscript{#1}}
metrics.sort(key=itemgetter(0))
print(
    r"Model & AP & mAP\tsub{.5} & mAP\tsub{.75} & AP\tsub{l} & AP\tsub{m}"
    r" & AP\tsub{s} & TPR & PPV & TP & FP \\"
)
print(r"\midrule")
for row in metrics:
    model, ap, ap50, ap75, apl, apm, aps, tpr, ppv, tp, fp = row
    print(
        f"{model} & {ap:.1f} & {ap50:.1f} & {ap75:.1f} & {apl:.1f} & {apm:.1f}"
        f" & {aps:.1f} & {tpr:.1f} & {ppv:.1f} & {tp:,} & {fp:,} \\\\"
    )
