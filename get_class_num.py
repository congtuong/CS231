from pathlib import Path
from glob import glob
from tqdm import tqdm

label_path = "/mlcv1/WorkingSpace/Personal/tuongbck/project_231/dataset/train/labels"

def get_class_num(data_dir):
    class_0 = 0
    class_1 = 0
    for file in tqdm(glob(data_dir + "/*.txt")):
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                class_num = line.split(" ")[0]
                if class_num == "0":
                    class_0 += 1
                elif class_num == "1":
                    class_1 += 1
    return class_0, class_1

print(get_class_num(label_path))