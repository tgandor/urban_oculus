import os
import shutil
import json
import time

import torch

import detectron2.model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultPredictor

try:
  from tqdm.notebook import tqdm
except ImportError:
  from tqdm import tqdm_notebook as tqdm


def filesizes(glob_expr):
  import glob
  return {
      os.path.basename(f): os.path.getsize(f)
      for f in glob.glob(glob_expr)
  }


def run_validation(model_config="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"):
  cfg = get_cfg()
  cfg.merge_from_file(detectron2.model_zoo.get_config_file(model_config))
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
  cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url(model_config)
  predictor = DefaultPredictor(cfg)
  data_loader = build_detection_test_loader(cfg, 'coco_2017_val')
  evaluator = COCOEvaluator('coco_2017_val', cfg, False)
  start = time.time()
  results = inference_on_dataset(predictor.model, data_loader, evaluator)
  inference_time = time.time() - start
  return results, inference_time


def save(path, output_dir=None):
    if not output_dir:
        return

    basename = os.path.basename(path)
    target = os.path.join(output_dir,  basename)
    if os.path.abspath(path) == os.path.abspath(target):
      print('Not copying:', path)
      return

    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(path, output_dir)
    print(f'Saving {path} to {output_dir}')


def save_filesizes(name, quality, elapsed, output_dir=None):
  with open(name, 'w') as f:
    f.write(json.dumps({
        'quality': quality,
        'filesizes': filesizes('datasets/coco/val2017/*.jpg'),
        'elapsed': 0,
    }))
  save(name, output_dir)


def result_name(model_config, quality):
    """Produce the result JSON name.

    >>> result_name("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml", 10)
    'val2017_mask_rcnn_R_101_FPN_3x_q_10.json'
    """
    model = os.path.splitext(os.path.basename(model_config))[0]
    return  f'val2017_{model}_q_{quality}.json'


def save_results(model_config, quality, results, inference_time, output_dir=None):
  result_file = result_name(model_config, quality)
  with open(result_file, 'w') as f:
    f.write(json.dumps({
        'quality': quality,
        'bbox': results['bbox'],
        'elapsed': inference_time,
        'model_config': model_config,
        'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': torch.cuda.get_device_name(),
    }))
  save(result_file, output_dir)


def result_path(model_config, quality, output_dir):
  result_file = result_name(model_config, quality)
  return os.path.join(output_dir, result_file)


'''
COCO Object Detection Baselines

Faster R-CNN:
Name     lr_sched train_time_(s/iter) inference_time_(s/im) train_mem_(GB) box_AP model_id   model_idx
R50-C4    3x        0.543                 0.104                   4.8       38.4   137849393  0
R50-DC5   3x        0.378                 0.070                   5.0       39.0   137849425  1
R50-FPN   3x        0.209                 0.038                   3.0       40.2   137849458  2
R101-C4   3x        0.619                 0.139                   5.9       41.1   138204752  3
R101-DC5  3x        0.452                 0.086                   6.1       40.6   138204841  4
R101-FPN  3x        0.286                 0.051                   4.1       42.0   137851257  5
X101-FPN  3x        0.638                 0.098                   6.7       43.0   139173657  6

TODO: add RetinaNet, consided RPN & Fast R-CNN
'''

# for object detection, BTW.

MODEL_ZOO_CONFIGS = [
  "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
  "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml",
  "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
  "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml",
  "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml",
  "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
  "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
  "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
]
