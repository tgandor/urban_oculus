import argparse
import logging
import os
import time
import json

import torch

import detectron2.model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultPredictor

# recommended for nice information
logging.basicConfig(level=logging.INFO)

MODEL_ZOO_CONFIGS = {
    "R50_C4": "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
    "R50_DC5": "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml",
    "R50_FPN": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "R101_C4": "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml",
    "R101_DC5": "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml",
    "R101_FPN": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    "X101": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    "R50": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
    "R101": "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
}

if not os.path.exists("datasets"):
    print("Copy or symlink datasets/ directory here. (please)")
    exit()

if not os.path.exists("val2017.zip"):
    print("Copy or symlink val2017.zip file here. (please)")
    exit()


def validate_quality(q, model, min_score):
    model_config = MODEL_ZOO_CONFIGS[model]
    out_folder = f"evaluator_dump_{model}_{q:03d}"
    os.system("unzip -o val2017.zip -d datasets/coco/")
    if 1 <= q <= 100:
        os.system(f"mogrify -verbose -quality {q} datasets/coco/val2017/*")
    else:
        print('Skipping quality degradation.')
    cfg = get_cfg()
    cfg.merge_from_file(detectron2.model_zoo.get_config_file(model_config))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min_score  # set threshold for R-CNN
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = min_score  # set threshold for RetinaNet
    cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url(model_config)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--device", "-d", type=int, help="select CUDA device")
    parser.add_argument("--minQ", type=int, help="min JPEG quality", default=1)
    parser.add_argument("--maxQ", type=int, help="min JPEG quality", default=100)
    parser.add_argument("--min-score", "-t", type=float, help="score threshold for objects", default=0.5)
    args = parser.parse_args()

    if args.device is not None:
        print(
            f"Current device {torch.cuda.current_device()} ({torch.cuda.get_device_name()})"
        )
        torch.cuda.set_device(args.device)
        print(
            f"New device {torch.cuda.current_device()} ({torch.cuda.get_device_name()})"
        )

    for i in range(args.minQ, args.maxQ + 1):
        validate_quality(i, args.model, args.min_score)
