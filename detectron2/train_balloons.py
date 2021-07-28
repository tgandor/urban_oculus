# This is taken from:
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
# i.e.
# Detectron2 Tutorial.ipynb

from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, Metadata
import random
import time

import cv2

# download, decompress the data
# !wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
# !unzip balloon_dataset.zip > /dev/null

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

import os
import numpy as np
import json
from detectron2.structures import BoxMode


def get_balloon_dicts(img_dir: str) -> list[dict]:
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_dataset() -> Metadata:
    if not os.path.isdir("balloon/"):
        print("download and decompress the data first:")
        print(
            "https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip"
        )
        exit()

    # registering the dataset in detectron:

    print("registering dataset...")

    for d in ["train", "val"]:
        DatasetCatalog.register(
            "balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d)
        )
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])

    return MetadataCatalog.get("balloon_train")


# training:


def train():
    print("preparing training...")

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # 300 iterations seems good enough for this toy dataset;
    # you may need to train longer for a practical dataset
    cfg.SOLVER.MAX_ITER = 300
    # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    print("this is a good moment to launch Tensorboard..., namely:")
    print(f"tensorboard --logdir {cfg.OUTPUT_DIR}")
    print("starting in 5 s ...")
    time.sleep(5)

    trainer.train()


# testing:


def cv2_imshow(image):
    cv2.imshow("result", image)
    res = cv2.waitKey(0)
    if res & 0xFF == ord("q"):
        exit()


def testing():
    print("testing...")
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    balloon_metadata = MetadataCatalog.get("balloon_train")

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # now needed again...
    # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.DATASETS.TEST = ("balloon_val",)
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_balloon_dicts("balloon/val")
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=balloon_metadata,
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        try:
            cv2_imshow(v.get_image()[:, :, ::-1])
        except:  # noqa
            print(d["file_name"])
            print(outputs["instances"].to("cpu"))


def main():
    register_dataset()
    train()
    testing()


if __name__ == "__main__":
    main()
