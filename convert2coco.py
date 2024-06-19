from pathlib import Path
import json

predictions = "/mlcv1/WorkingSpace/Personal/tuongbck/project_231/test_new_output/predictions_eff_lower_non_nms.json"
output = "/mlcv1/WorkingSpace/Personal/tuongbck/project_231/test_new_output/predictions_coco_eff_lower_non_nms.json"
gt = "/mlcv1/WorkingSpace/Personal/tuongbck/project_231/test_output/gt.json"

def get_id_from_filename(filename,gt):
    for item in gt["images"]:
        if item["file_name"] == filename:
            return item["id"]
    return None

def convert2coco(
    predictions: str,
    gt: str,
    output: str,
):
    with open(predictions, "r") as f:
        data = json.load(f)
    with open(gt, "r") as f:
        gt = json.load(f)
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 0,
            "name": "No",
            "supercategory": "No"
        }, 
        {
            "id": 1,
            "name": "Violate",
            "supercategory": "Violate"
        }],
        "info": {},
        "licenses": [],
    }
    pred = []
    # data is a list of dictionary with file_name: [x,y,w,h, label, score]
    for i, item in enumerate(data):
        image_name = list(item.keys())[0]
        image_stem = image_name.split("/")[-1]
        image_id = get_id_from_filename(image_stem,gt)
        # print(image_name)
        # input()
        image_width = 640
        image_height = 320
        image = {
            "id": image_id,
            "file_name": image_stem,
            "width": image_width,
            "height": image_height
        }
        coco_data["images"].append(image)
        for bbox in item[image_name]:
            x, y, w, h, label, score = bbox
            # recalculate the box
            x = 640*x / 416
            y = 320*y / 416
            w = 640*w / 416
            h = 320*h / 416 
            annotation = {
                "id": len(coco_data["annotations"]),
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
            pred.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x, y, w, h],
                "score": score,
            })
            coco_data["annotations"].append(annotation)
    with open(output, "w") as f:
        json.dump(pred, f)
    
convert2coco(predictions,gt, output)