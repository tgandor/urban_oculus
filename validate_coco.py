import argparse
import logging
import os
import time

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

cfg = get_cfg()
cfg.merge_from_file(detectron2.model_zoo.get_config_file(model_config))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url(model_config)
predictor = DefaultPredictor(cfg)
data_loader = build_detection_test_loader(cfg, 'coco_2017_val')
evaluator = COCOEvaluator('coco_2017_val', cfg, False, 'evaluator_dump')
start = time.time()
results = inference_on_dataset(predictor.model, data_loader, evaluator)
inference_time = time.time() - start
print(results)
