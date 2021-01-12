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

model_config = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"

if not os.path.exists('datasets'):
    print('Copy or symlink datasets/ directory here. (please)')
    exit()

if not os.path.exists('val2017.zip'):
    print('Copy or symlink val2017.zip file here. (please)')
    exit()

def validate_quality(q, model='R50'):
    out_folder = f'evaluator_dump_{model}_{q:03d}'
    os.system('unzip -o val2017.zip -d datasets/coco/')
    os.system(f'mogrify -verbose -quality {q} datasets/coco/val2017/*')
    cfg = get_cfg()
    cfg.merge_from_file(detectron2.model_zoo.get_config_file(model_config))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url(model_config)
    predictor = DefaultPredictor(cfg)
    data_loader = build_detection_test_loader(cfg, 'coco_2017_val')
    evaluator = COCOEvaluator('coco_2017_val', cfg, False, out_folder)
    start = time.time()
    results = inference_on_dataset(predictor.model, data_loader, evaluator)
    inference_time = time.time() - start
    print(results)
    torch.save(results, out_folder + '/results.pth')
    with open(out_folder + '/results.json', 'w') as jsf:
        json.dump({'quality': q, 'model': model, 'results': results, 'elapsed': inference_time}, jsf)


if __name__ == '__main__':
    for i in range(1, 101):
        validate_quality(i)
