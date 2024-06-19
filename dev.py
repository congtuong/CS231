import json

gt = "/mlcv1/WorkingSpace/Personal/tuongbck/project_231/test_output/gt.json"

def get_id_from_filename(filename,gt):
    for item in gt["images"]:
        if item["file_name"] == filename:
            return item["id"]
    return None

with open(gt, "r") as f:
    gt = json.load(f)
    file_name = "Bago_highway_10_03_jpg.rf.5782c5f4c44036cd5123f07fdc48b5f9.jpg"
    print(get_id_from_filename(file_name,gt))