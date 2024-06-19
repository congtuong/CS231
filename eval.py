from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
from pathlib import Path

pred = "/mlcv1/WorkingSpace/Personal/tuongbck/project_231/test_new_output/predictions_coco_vgg_lower.json"
gt = "/mlcv1/WorkingSpace/Personal/tuongbck/project_231/test_output/gt.json"

def evaluate(
    pred: str,
    gt: str,
):  
    cocoGt = COCO(gt)
    cocoDt = cocoGt.loadRes(pred)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print(cocoEval.stats)
    
evaluate(pred, gt)