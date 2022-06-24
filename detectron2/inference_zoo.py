import argparse

import cv2

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import get_checkpoint_url, get_config_file
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

ALIASES = {
    "R50_C4": "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
    "R50_DC5": "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml",
    "R50_FPN": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "R101_C4": "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml",
    "R101_DC5": "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml",
    "R101_FPN": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    "X101": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    "R50": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
    "R101": "COCO-Detection/retinanet_R_101_FPN_3x.yaml",
    "R50_C4x1": "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml",
    "R50_DC5x1": "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml",
    "R50_FPNx1": "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
    "R50x1": "COCO-Detection/retinanet_R_50_FPN_1x.yaml",
}


def _parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_alias")
    parser.add_argument("--threshold", "-t", type=float)  # D2 default: 0.05
    parser.add_argument("--delay", "-d", type=int, default=0)
    parser.add_argument("--category", "-c")
    parser.add_argument("--mono", "-m", action="store_true", help="desaturate img")
    parser.add_argument("images", nargs="+")
    return parser.parse_args()


def cv2_imshow(image, delay=0):
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", image)
    res = cv2.waitKey(delay)
    if res & 0xFF == ord("q"):
        exit()


def main():
    args = _parse_cli()
    config = ALIASES.get(args.config_alias)
    if config is None:
        print("Wrong config alias, choose one of:")
        print("   ", ", ".join(ALIASES.keys()))
        exit()

    cfg = get_cfg()
    cfg.merge_from_file(get_config_file(config))
    cfg.MODEL.WEIGHTS = get_checkpoint_url(config)
    if args.threshold:
        # for R-CNN
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold
        # for RetinaNet
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.threshold

    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get("coco_2017_val")
    names = metadata.get("thing_classes", None)

    for img_idx, image in enumerate(args.images):
        print(img_idx + 1, image)
        im = cv2.imread(image)
        outputs = predictor(im)

        v = Visualizer(
            im[:, :, ::-1],
            metadata=metadata,
            scale=0.8,
            # remove the colors of unsegmented pixels
            instance_mode=ColorMode.IMAGE_BW if args.mono else ColorMode.IMAGE,
        )

        predictions = outputs["instances"].to("cpu")

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = (
            predictions.pred_classes.tolist()
            if predictions.has("pred_classes")
            else None
        )

        found_category = args.category is None

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
            if names[class_id] == args.category:
                found_category = True

        if not found_category:
            print(f"No {args.category} found on {image} (not showing)")
            continue

        v = v.draw_instance_predictions(predictions)
        cv2_imshow(v.get_image()[..., ::-1], args.delay)


if __name__ == "__main__":
    main()
