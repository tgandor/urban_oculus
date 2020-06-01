import os
import shutil
import json
import time

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


def run_validation(model_config="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
  cfg = get_cfg()
  cfg.merge_from_file(detectron2.model_zoo.get_config_file(model_config)))
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
    if output_dir:
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


def save_results(result_file, quality, results, inference_time, output_dir=None):
  with open(result_file, 'w') as f:
    f.write(json.dumps({
        'quality': quality,
        'bbox': results['bbox'],
        'elapsed': inference_time,
    }))
  save(result_file, output_dir)

