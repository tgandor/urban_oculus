import argparse
import logging
import os
import time
import json

import torch

from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultPredictor

# recommended for nice information
logging.basicConfig(level=logging.INFO)


def validate_quality(
    q, model: str, config: str, weights: str, min_score: str, verbose: bool = False
):

    if not os.path.exists("datasets"):
        print("Copy or symlink datasets/ directory here. (please)")
        exit()

    if not os.path.exists("val2017.zip"):
        print("Copy or symlink val2017.zip file here. (please)")
        exit()

    out_folder = f"evaluator_dump_{model}_{q:03d}"
    os.system(f"unzip {'' if verbose else '-q'} -o val2017.zip -d datasets/coco/")
    if 1 <= q <= 100:
        os.system(f"mogrify {'-verbose' if verbose else ''} -quality {q} datasets/coco/val2017/*")
    else:
        print("Skipping quality degradation.")
    cfg = get_cfg()
    cfg.merge_from_file(config)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min_score  # set threshold for R-CNN
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = min_score  # set threshold for RetinaNet
    cfg.MODEL.WEIGHTS = weights
    predictor = DefaultPredictor(cfg)
    data_loader = build_detection_test_loader(cfg, "coco_2017_val")
    evaluator = COCOEvaluator("coco_2017_val", cfg, False, out_folder)
    start = time.time()
    results = inference_on_dataset(predictor.model, data_loader, evaluator)
    inference_time = time.time() - start
    print(results)
    torch.save(results, out_folder + "/results.pth")
    with open(out_folder + "/results.json", "w") as jsf:
        json.dump(
            {
                "quality": q,
                "model": model,
                "results": results,
                "elapsed": inference_time,
                "device": torch.cuda.get_device_name(),
            },
            jsf,
        )


def _parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("config")
    parser.add_argument("weights")
    parser.add_argument("--device", "-d", type=int, help="select CUDA device")
    # default: single -- no degradation!
    parser.add_argument("--minQ", "-m", type=int, help="min JPEG quality", default=101)
    parser.add_argument("--maxQ", "-M", type=int, help="max JPEG quality", default=101)
    parser.add_argument("--stepQ", "-S", type=int, help="JPEG quality step", default=1)
    parser.add_argument(
        "--min-score",
        "-t",
        type=float,
        help="score threshold for objects",
        default=0.05,
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_cli()
    if args.device is not None:
        print(
            f"Current device {torch.cuda.current_device()} ({torch.cuda.get_device_name()})"
        )
        torch.cuda.set_device(args.device)
        print(
            f"New device {torch.cuda.current_device()} ({torch.cuda.get_device_name()})"
        )

    for i in range(args.minQ, args.maxQ + 1, args.stepQ):
        validate_quality(i, args.model, args.config, args.weights, args.min_score, args.verbose)


if __name__ == "__main__":
    main()
