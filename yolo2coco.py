import json

from pathlib import Path

gt = "/mlcv1/WorkingSpace/Personal/tuongbck/project_231/dataset/test/labels"
output = "/mlcv1/WorkingSpace/Personal/tuongbck/project_231/test_output/gt.json"

def yolo2coco(
    gt: str,
    output: str,
):
    data = []
    for file in Path(gt).glob("*"):
        # print(file)
        file = str(file)
        with open(file, "r") as f:
            lines = f.readlines()
        annotations = []
        for line in lines:
            line = line.strip().split(" ")
            label = int(line[0])
            x, y, w, h = map(float, line[1:])
            x, y, w, h = x - w/2, y - h/2, w, h
            annotations.append([x, y, w, h, label, 1.0])
        # print(file)
        data.append({
            file.split("/")[-1].replace("txt", "jpg"): annotations,
        })
            
    convert2coco(data, output)
        
def convert2coco(
    data: str,
    output: str,
):

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
    # data is a list of dictionary with file_name: [x,y,w,h, label, score]
    for i, item in enumerate(data):
        image_id = i
        # print(item)
        image_name = list(item.keys())[0]
        image_width = 640
        image_height = 320
        image = {
            "id": image_id,
            "file_name": image_name,
            "width": image_width,
            "height": image_height
        }
        coco_data["images"].append(image)
        for bbox in item[image_name]:
            x, y, w, h, label, score = bbox
            # recalculate the box
            x = 640*x
            y = 320*y
            w = 640*w
            h = 320*h
            annotation = {
                "id": len(coco_data["annotations"]),
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
            coco_data["annotations"].append(annotation)
    with open(output, "w") as f:
        json.dump(coco_data, f)
    
yolo2coco(gt, output)