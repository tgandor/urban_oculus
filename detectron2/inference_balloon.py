import argparse
import os

import cv2

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from train_balloons import register_dataset


def _parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    )
    parser.add_argument("--weights", default="output/model_final.pth")
    parser.add_argument("images", nargs=argparse.REMAINDER)
    return parser.parse_args()


def cv2_imshow(image):
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", image)
    res = cv2.waitKey(0)
    if res & 0xFF == ord("q"):
        exit()


def main():
    args = _parse_cli()
    if not os.path.exists(args.weights):
        print(f"Missing weights file {args.weights}. Please run train_balloons.py first...")
        exit()

    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights

    # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # optional
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # critical, or weights will be ignored
    predictor = DefaultPredictor(cfg)
    balloon_metadata = register_dataset()
    names = balloon_metadata.get("thing_classes", None)

    for image in args.images:
        print(image)
        im = cv2.imread(image)
        outputs = predictor(im)

        v = Visualizer(
            im[:, :, ::-1],
            metadata=balloon_metadata,
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels
        )

        predictions = outputs["instances"].to("cpu")

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = (
            predictions.pred_classes.tolist()
            if predictions.has("pred_classes")
            else None
        )

        for box, score, class_id in zip(boxes, scores, classes):
            x0, y0, x1, y1 = box
            print(
                {
                    "bbox": [float(b) for b in box],
                    "x": float(x0),
                    "y": float(y0),
                    "w": float(x1 - x0),
                    "h": float(y1 - y0),
                    "category_id": class_id,
                    "category": names[class_id],
                    "score": float(score),
                }
            )

        v = v.draw_instance_predictions(predictions)
        cv2_imshow(v.get_image()[..., ::-1])


if __name__ == "__main__":
    main()
