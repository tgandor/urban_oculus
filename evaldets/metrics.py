from operator import itemgetter
import numpy as np


def interpolated_PPV(ppv: np.array):
    """Set precision to max(current, max(following)), so called p_interp."""
    ppvl = ppv.tolist()
    for k in range(len(ppvl)-1, 0, -1):
        ppvl[k-1] = max(ppvl[k-1], ppvl[k])
    return np.array(ppvl)


def resample_pr_curve(tpr, ppvi, thresholds=None):
    """Find precision values for recall thresholds. see COCOeval.accumulate()"""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.00, 101, endpoint=True)

    precision = np.zeros_like(thresholds)
    inds = np.searchsorted(tpr, thresholds, side="left")

    for ri, pi in enumerate(inds):
        if pi >= len(ppvi):
            break
        precision[ri] = ppvi[pi]

    return thresholds, precision


def compute_iou(x, gt):
    """A little deprecated (gt_bbox - not used)."""
    def segment_overlap(x1, x2, y1, y2):
        return max(0, min(x2, y2) - max(x1, y1))
    left1, top1, w1, h1 = itemgetter("x", "y", "w", "h")(x)
    left2, top2, w2, h2 = gt["bbox"]
    intersection = (
        segment_overlap(left1, left1+w1, left2, left2+w2)
        * segment_overlap(top1, top1+h1, top2, top2+h2)
    )
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union
