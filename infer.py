from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import json
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import torchvision
from torchvision.models import vgg16_bn
from sklearn.cluster import KMeans
from utils import *
dataset_path = Path("./dataset")

train_path = dataset_path / "train"
val_path = dataset_path / "valid"
test_path = dataset_path / "test"

class HelmetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def _load_img_tensor(self,img_path):
        img = cv2.imread(img_path).astype(np.float32)/255.0
        img = cv2.resize(img, (416, 416))
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img

    def boxes_to_tensor(self, boxes, labels):
        """
        Convert list of boxes (and labels) to tensor format
        Output:
            boxes_tensor: shape = (Batchsize, S, S, Box_nb, (4+1+CLS))
        """
        boxes_tensor = torch.zeros((13, 13, 5, 7))
        cell_w, cell_h = 416/13,416/13
        temp = torch.zeros(len(labels), 2)
        temp.scatter_(1, labels.unsqueeze(1).type(torch.int64), 1)
        labels = temp
        for i, box in enumerate(boxes):
            x,y,w,h = box
            x,y,w,h = x*416,y*416,w*416,h*416
            # normalize xywh with cell_size
            x,y,w,h = x/cell_w, y/cell_h, w/cell_w, h/cell_h
            center_x, center_y = x+w/2, y+h/2
            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))

            if grid_x < 13 and grid_y < 13:
                boxes_tensor[grid_y, grid_x, :, 0:4] = torch.tensor(5 * [[center_x-grid_x,center_y-grid_y,w,h]])
                boxes_tensor[grid_y, grid_x, :, 4]  = torch.tensor(5 * [1.])
                boxes_tensor[grid_y, grid_x, :, 5:]  = torch.tensor(5*[labels[i].numpy()])
        return boxes_tensor

    def __getitem__(self, idx):
        img, bboxes, classes = self.dataset[idx]
        target = self.boxes_to_tensor(bboxes, classes)
        return img,self._load_img_tensor(img), target

def collate_fn(batch):
    return tuple(zip(*batch))

def convert_yolo_to_xyxy(yolo_bbox, img_width, img_height):
    x, y, w, h = yolo_bbox
    x1 = (x - w/2) * img_width
    x2 = (x + w/2) * img_width
    y1 = (y - h/2) * img_height
    y2 = (y + h/2) * img_height
    return [x1, y1, x2, y2]

def read_yolo_annotation(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    bboxes = []
    classes = []
    height, width = cv2.imread(str(file_path).replace("labels", "images").replace("txt", "jpg")).shape[:2]
    for line in lines:
        class_idx, x, y, w, h = map(float, line.strip().split())
        bboxes.append([x-w/2, y-h/2, w, h])
        classes.append(int(class_idx))
    return np.array(bboxes), np.array(classes)

def load_img_tensor(img_path):
    img = cv2.imread(img_path).astype(np.float32)/255.0
    img = cv2.resize(img, (416, 416))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img

def find_anchor_boxes(dataset):
    anchor_boxes = []
    kmean = KMeans(n_clusters=5,random_state=42)
    x,y,w,h = [],[],[],[]
    for img, bboxes, classes in dataset:
        for bbox in bboxes:
            x.append(bbox[0]*640)
            y.append(bbox[1]*320)
            w.append(bbox[2]*640)
            h.append(bbox[3]*320)
    X = np.asarray([np.asarray(w), np.asarray(h)])
    X = X.transpose()
    kmean.fit(X)
    y_kmeans = kmean.predict(X)
    centers = kmean.cluster_centers_
    yolo_anchor_avg = []
    for i in range(5):
        yolo_anchor_avg.append(np.mean(X[y_kmeans==i],axis=0))

    yolo_anchor_avg = np.array(yolo_anchor_avg)
    yolo_anchors = yolo_anchor_avg
    yolo_anchors[:, 0] =yolo_anchor_avg[:, 0]/640 *416
    yolo_anchors[:, 1] =yolo_anchor_avg[:, 1]/320 *416
    yolo_anchors = np.rint(yolo_anchors)
    # normalize anchors by cell size
    return yolo_anchors / 32

def dataset_tensor_to_boxes(boxes_tensor, thresh = 0.0):
    cell_w, cell_h = 416/13, 416/13
    boxes = []
    probs = []

    for temp in boxes_tensor:
        temp_boxes = []
        for i in range(13):
            for j in range(13):
                for b in range(5):
                    data = temp[i,j,b]
                    xy = data[:2]
                    wh = data[2:4]
                    obj_prob = (data[4:5]+1e-5)
                    cls_prob =torch.nn.Softmax(dim=-1)(data[5:])
                    # print(torch.nn.Softmax(dim=-1)(torch.tensor([0.99,0.01])))
                    # values,indices  =torch.max(cls_prob,dim=1)
                    # print(temp,values)
                    # print(obj_prob)
                    combine_prob = obj_prob*max(cls_prob)
                    # best_box = torch.argmax(combine_prob)

                    if combine_prob > thresh:
                        x_center, y_center, w, h = xy[0], xy[1], wh[0], wh[1]
                        x, y = x_center+j-w/2, y_center+i-h/2
                        x,y,w,h = x*cell_w, y*cell_h, w*cell_w, h*cell_h
                        box = [x,y,w,h, combine_prob]
                        temp_boxes.append(box)
        boxes.append(temp_boxes)
    return boxes

def load_dataset(dataset_path):
    dataset = []
    for img_path in tqdm(list(dataset_path.glob("images/*.jpg"))):
        label_path = img_path.as_posix().replace("images", "labels").replace(".jpg", ".txt")
        bboxes, classes = read_yolo_annotation(label_path)
        img = img_path.as_posix()
        # resize to VGG16 input size
        # img = cv2.imread(img_path.as_posix()).astype(np.float32)/255.0
        # img = cv2.resize(img, (416, 416))
        # img = torch.from_numpy(img).permute(2, 0, 1).float()
        bboxes = torch.from_numpy(bboxes)
        classes = torch.from_numpy(classes)
        dataset.append((img, bboxes, classes))
    anchors = find_anchor_boxes(dataset)
    return HelmetDataset(dataset), anchors[np.argsort(anchors[:, 0])]


print("Loading dataset")
# train,anchors = load_dataset(train_path)
test,_ = load_dataset(test_path)
# val,_ = load_dataset(val_path)
# anchors = torch.from_numpy(anchors)

anchors = torch.tensor([[ 19.,  50.],
        [ 33.,  78.],
        [ 49., 128.],
        [ 81.,  77.],
        [ 90., 162.]])

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn = True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.drop = nn.Dropout(p=0.2, inplace=False)
        self.bn_flag = bn
    def forward(self, x):
        x = self.conv(x)
        if self.bn_flag:
            x = self.bn(x)
        x = self.drop(self.relu(x))
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, repeat = 1, use_residual = True):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(repeat):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels//2, kernel_size=1),
                    CNNBlock(channels//2, channels, kernel_size=3, padding=1)
                )
            ]
        self.use_residual = use_residual
        self.repeat = repeat
    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x

def test_res_block():
    x = torch.randn(2, 3, 256, 256)
    res_block = ResidualBlock(3, 8)
    print(res_block(x).shape)

class Yolov3VGG(nn.Module):
    def __init__(self,anchors, num_classes=2, vgg = vgg16_bn(pretrained=True),device='cuda'):
        super(Yolov3VGG, self).__init__()
        self.num_anchors = 5
        self.anchors = anchors
        self.num_classes = num_classes
        self.vgg_layers = nn.Sequential(*list(vgg.features.children())[:-1])
        self.device = device
        # fit through the first 23 layers of vgg then add extra layers with 3 convs and a maxpool
        self.extra = nn.Sequential(
            CNNBlock(512, 512, kernel_size=3, padding=1, stride=1),
            CNNBlock(512, 512, kernel_size=3, padding=1, stride=1),
            CNNBlock(512, 512, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # skip
        self.skip_module = nn.Sequential(
            CNNBlock(512, 64, kernel_size=1, padding=0, stride=1),
        )

        self.final = nn.Sequential(
            CNNBlock(256 + 512, 1024, kernel_size=3, padding=1,stride=1),
            CNNBlock(1024,256, kernel_size=3, padding=1,stride = 1),
            nn.Conv2d(256, self.num_anchors*(5 + self.num_classes),1),
        )
        self.init_values()

    def init_values(self):
        for c in self.final.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, 0, 0.01)
                nn.init.constant_(c.bias, 0)
        for c in self.extra.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, 0, 0.01)
                nn.init.constant_(c.bias, 0)
        for c in self.skip_module.children():
            if isinstance(c, nn.Conv2d):
                nn.init.normal_(c.weight, 0, 0.01)
                nn.init.constant_(c.bias, 0)

    def forward(self, x):
        # x is a batch of images
        x = torch.stack(x)
        # print(x.shape)
        output_size = x[0].shape[-1]
        output_size /= 32
        o_size = int(output_size)

        x = x.to(self.device)

        x = self.vgg_layers(x)

        skip_x = self.skip_module(x).to(self.device)

        skip_x = skip_x.view(-1, 64, o_size, 2, o_size, 2).contiguous()
        skip_x = skip_x.permute(0, 3, 5, 1, 2, 4).contiguous()
        skip_x = skip_x.view(-1, 256, o_size, o_size)

        x = self.extra(x)
        # print(x.shape)
        # print(skip_x.shape)
        x = torch.cat((x, skip_x), dim=1)
        x = self.final(x)
        return x

def test_yolov3_vgg():
    x = torch.randn(2, 3, 256, 256)
    model = Yolov3VGG()
    print(model(x).shape)

def Yolo_loss(pred_targets, target_tensor,anchors, num_classes=2):
    # anchors = torch.tensor([[1.08,1.19],
    #            [3.42,4.41],
    #            [6.63,11.38],
    #            [9.42,5.11],
    #            [16.62,10.52]])
    num_anchors = 5
    pred_targets = pred_targets.permute(0, 2, 3, 1).to('cpu')
    out_size = pred_targets.size(1)
    pred_targets = pred_targets.view(-1, out_size, out_size, 5, 5 + num_classes)

    pred_xy = pred_targets[..., :2].sigmoid()
    pred_wh = pred_targets[..., 2:4].exp() * anchors.view(1, 1, 1, num_anchors, 2)
    # pred_bbox = torch.cat((pred_xy, pred_wh), dim=-1)
    # pred_bbox = pred_bbox.view(-1, 13 * 13 * 5, 4) / 13.
    pred_conf = (pred_targets[..., 4:5]+1e-5).sigmoid()
    pred_cls = pred_targets[..., 5:]
    pred_cls = torch.nn.Softmax(dim=-1)(pred_cls)

    pred_x1y1 = pred_xy - pred_wh/2
    pred_x2y2 = pred_xy + pred_wh/2
    pred_area = pred_wh[..., 0]*pred_wh[..., 1]

    target_tensor = torch.stack(target_tensor)
    gt_xy = target_tensor[..., :2]
    gt_wh = target_tensor[..., 2:4]
    gt_x1y1 = gt_xy - gt_wh/2
    gt_x2y2 = gt_xy + gt_wh/2
    gt_conf = target_tensor[..., 4:5]
    gt_cls = target_tensor[..., 5:]


    gt_area = (gt_x2y2[..., 0] - gt_x1y1[..., 0])*(gt_x2y2[..., 1] - gt_x1y1[..., 1])

    intersect_x1y1 = torch.max(pred_x1y1, gt_x1y1)
    intersect_x2y2 = torch.min(pred_x2y2, gt_x2y2)
    intersect_wh = intersect_x2y2 - intersect_x1y1
    intersect_area = intersect_wh[..., 0]*intersect_wh[..., 1]

    iou = intersect_area/(pred_area + gt_area - intersect_area + 1e-7)
    max_iou = torch.max(iou, dim=3, keepdim=True)[0]
    best_box_idx = torch.unsqueeze(torch.eq(iou, max_iou).float(), dim=-1)
    true_box_conf = best_box_idx*gt_cls
    # print(true_box_conf.shape)
    batch = pred_targets.shape[0]
    xy_loss = (((pred_xy - gt_x1y1)**2)*true_box_conf*5.0).sum() / batch
    wh_loss = (((pred_wh - gt_wh)**2)*true_box_conf*5.0).sum() / batch
    obj_loss = (((pred_conf - gt_conf)**2)*(true_box_conf + 0.5*(1-true_box_conf))).sum() / batch
    cls_loss = (((pred_cls -gt_cls)**2)*true_box_conf*1.0).sum() / batch

    return (xy_loss + wh_loss + obj_loss + cls_loss), (xy_loss + wh_loss + obj_loss + cls_loss)


import time
def train_fn(model, train_dataset, val_dataset,save_path, epochs = 10, batch_size = 32, collate_fn = None):
    print("Training")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5,weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=1000, eta_min=0)

    for epoch in tqdm(range(epochs)):
        model.train()
        for i, (imgs, target_tensor) in enumerate(train_loader):
            optimizer.zero_grad()
            start = time.time()
            output = model(imgs)
            # print("Prediction Time: ", time.time() - start)
            # print(output.shape)
            # return output
            start = time.time()
            loss,losses = Yolo_loss(output, target_tensor,model.anchors, model.num_classes)
            # print("Loss Time: ", time.time() - start)
            loss.backward()
            optimizer.step()
            if i%100 == 0:
                print(f"Epoch {epoch} Iter {i} Loss: {loss.item()} , lr = {optimizer.param_groups[0]['lr']}")
        lr_scheduler.step()
        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_path + f"model_yololoss_cuda_{epoch}.pt")
        model.eval()
        with torch.no_grad():
            for i, (imgs, target_tensor) in enumerate(val_loader):
                output = model(imgs)
                loss,losses = Yolo_loss(output, target_tensor,model.anchors, model.num_classes)
                print(f"Epoch {epoch} Iter {i} Val Loss: {loss.item()}")
                
                
                
weights = "/mlcv1/WorkingSpace/Personal/tuongbck/project_231/weights_new_lower/model_yololoss_cuda_40.pt"

# # save

model = Yolov3VGG(anchors = anchors,vgg=vgg16_bn(pretrained=True),device='cpu').to('cpu')
model.load_state_dict(torch.load(weights,map_location=torch.device('cpu')))

# !nvidia-smi --gpu-reset -i "0"
# !fuser -v /dev/nvidia*
# !kill -9 "7732"


def output_tensor_to_boxes(boxes_tensor,anchors, thresh = 0.0):
    cell_w, cell_h = 416/13, 416/13
    boxes = []
    probs = []

    for temp in boxes_tensor:
        temp_boxes = []
        for i in range(13):
            for j in range(13):
                for b in range(5):
                    anchor_wh =anchors[b]
                    data = temp[i,j,b]
                    xy = data[:2].sigmoid()
                    wh = data[2:4].exp()*anchor_wh
                    obj_prob = (data[4:5]+1e-5).sigmoid()
                    cls_prob =torch.nn.Softmax(dim=-1)(data[5:])
                    # print(cls_prob)
                    # values,indices  =torch.max(cls_prob,dim=1)
                    # print(temp,values)
                    # print(obj_prob)
                    combine_prob = obj_prob*max(cls_prob)
                    # best_box = torch.argmax(combine_prob)

                    if combine_prob > thresh:
                        x_center, y_center, w, h = xy[0], xy[1], wh[0], wh[1]
                        x, y = x_center+j-w/2, y_center+i-h/2
                        x,y,w,h = x*cell_w, y*cell_h, w*cell_w, h*cell_h
                        box = [float(x),float(y),float(w),float(h), float(combine_prob[0])]
                        temp_boxes.append(box)
        boxes.append(temp_boxes)
    return boxes

def _get_prediction(pred_targets,anchors, num_classes=2):

    num_anchors = 5
    pred_targets = pred_targets.permute(0, 2, 3, 1)
    out_size = pred_targets.size(1)
    pred_targets = pred_targets.view(-1, out_size, out_size, 5, 5 + num_classes)

    boxes = output_tensor_to_boxes(pred_targets,anchors,thresh=0.0)

    pred_xy = pred_targets[..., :2].sigmoid()
    pred_wh = pred_targets[..., 2:4].exp() * anchors.view(1, 1, 1, num_anchors, 2)
    pred_bbox = torch.cat((pred_xy, pred_wh), dim=-1)
    # pred_bbox = pred_bbox.view(-1, 13 * 13 * 5, 4) / 13.
    pred_conf = (pred_targets[..., 4:5]+1e-5).sigmoid()
    pred_cls = pred_targets[..., 5:]
    pred_cls = torch.nn.Softmax(dim=-1)(pred_cls)

    return boxes, pred_conf, pred_cls

def visualize_prediction(bbox,cls,conf, img,threshold=0.02):
    img = img.permute(1, 2, 0).numpy()
    img = cv2.resize(img, (416, 416))
    img = (img * 255).astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)sssssssssss
    # idxs = torch.where(torch.max(conf)>=threshold)
    # print(bbox.shape)
    map_color = {
        0: (0,255,0),
        1: (255,0,0)
    }
    thresholded_list = []
    for i in range(len(bbox)):
        box = bbox[i]
        x,y,w,h,conf= box
        x_center, y_center, w, h = x+w/2, y+h/2, w, h
        # print(x,y,w,h)
        x1, y1, x2, y2 = int(x),int(y),int((x+w)),int((y+h))
        if x1 > 0 and y1 > 0 and x2 < 416 and y2 < 416 and float(conf) >= threshold:
            thresholded_list.append([x1, y1, int(w),int(h), int(torch.argmax(cls[i])), float(conf)])
            cv2.putText(img, str(float(conf))[:4], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(img, (
                x1, 
                y1,
                ), (
                    x2, 
                    y2,
                    ), map_color[int(torch.argmax(cls[i]))], 2)
    return img, thresholded_list
    # cv2.imwrite("test.jpg",img)
    # plt.imshow(img)
    # plt.show()
    # fix floating point image
    # plt.imsave("test.jpg",img)

def visualize_prediction_to_video(test_dataset, model,anchors,device, save_path, batch_size = 32, collate_fn = None,threshold = 0.5):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.to(device)
    anchors = anchors.to(device)# print(imgs)(device)
    model.eval()
    video_list = []
    output_list = []
    with torch.no_grad():
        for i, (path,imgs, _) in tqdm(enumerate(test_loader)):
            # print(imgs)
            output = model(imgs)
            pred_bbox, pred_conf, pred_cls = _get_prediction(output,anchors)
            for idx in range(len(pred_bbox)):
                # _,_,indices = nms(np.array(pred_bbox[idx]), np.array(pred_bbox[idx])[:,4], 0.5)
                # print(np.array(pred_bbox[idx])[indices].shape)
                # print(np.array(pred_cls[idx].cpu().view((845,2))[indices].shape))
                # print(np.array(pred_conf[idx].cpu().reshape((845,1)))[indices].shape)
                
                # img, thresholded_list = visualize_prediction(np.array(pred_bbox[idx])[indices],pred_cls[idx].cpu().view((845,2))[indices],np.array(pred_conf[idx].cpu().reshape((845,1)))[indices], imgs[idx],threshold)
                img, thresholded_list = visualize_prediction(pred_bbox[idx],pred_cls[idx].cpu().view((845,2)),pred_conf[idx].cpu().reshape((845,1)), imgs[idx],threshold)
                output_list.append({path[idx]:thresholded_list})
                video_list.append(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = Image.fromarray(img)
                #img.save(save_path + f"/images_nms/img_{i*batch_size+idx}.jpg")
    video = cv2.VideoWriter(save_path + "output_vgg_lower_non_nms.avi", cv2.VideoWriter_fourcc(*'DIVX'), 1, (416,416))
    for img in video_list:
        video.write(img)
    video.release()
    with open(save_path + "predictions_vgg_lower_non_nms.json", "w") as f:
        json.dump(output_list, f)
    # video.save(save_path + "output.avi")
    
                # cv2.imwrite(save_path + f"img_{i*batch_size+idx}.jpg",img)
                # plt.imshow(img)
                # plt.show()
                # input()
                # print(f"Iter {i} Predictions: {pred_bbox}, {pred_conf}, {pred_cls[idx]}")
                # visualize_prediction(pred_bbox[0], pred_cls[0],pred_cls[0], imgs[0],threshold)

# def predict(model, test_dataset, batch_size = 32, collate_fn = None,threshold = 0.5):
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
#     model.eval()
#     with torch.no_grad():
#         for i, (imgs, bboxes, classes) in enumerate(test_loader):
#             output = model(imgs)
#             pred_bbox, pred_conf, pred_cls = _get_prediction(output)
#             for idx in range(len(pred_bbox)):
#                 indices = nms(np.array(pred_bbox[idx]), np.array(pred_bbox[idx])[:,4], 0.2)
                
                # print(f"Iter {i} Predictions: {pred_bbox}, {pred_conf}, {pred_cls[idx]}")
                # visualize_prediction(pred_bbox[0], pred_cls[0],pred_cls[0], imgs[0],threshold)

# predict(model, train, batch_size = 2, collate_fn = collate_fn,threshold = 0.5)
model.eval()
# img, target_tensor = test[250]
import pickle
with open("/mlcv1/WorkingSpace/Personal/tuongbck/project_231/anchors.pkl", 'rb') as f:
    anchors = torch.tensor(pickle.load(f))
print(anchors)

visualize_prediction_to_video(test, model,anchors,'cpu', "/mlcv1/WorkingSpace/Personal/tuongbck/project_231/test_new_output/", batch_size = 32, collate_fn = collate_fn,threshold = 0.5)
# with torch.no_grad():
#     output = model([img])
# pred_bbox, pred_conf, pred_cls = _get_prediction(output,anchors)
# # print(pred_bbox[0])
# print(pred_conf[0].shape)
# # input()
# picked_boxes, picked_score, indices = nms(np.array(pred_bbox[0]), np.array(pred_bbox[0])[:,4], 0.2)

# print(np.array(pred_bbox[0]).shape)
# print(indices)
# # input()
# print(pred_cls.shape)
# visualize_prediction(np.array(pred_bbox[0])[indices], pred_cls[0].view((845,2))[indices],np.array(pred_conf[0]), img,threshold=0.3)
