import numpy as np
import torch
import cv2

from PIL import Image

def iou(box1, box2):
    """
    Calculate intersection over union between two boxes
    box1: [x1, y1, x2, y2]
    box2
    """
    # Calculate intersection
    # print(box1, box2)
    x1, y1, w1, h1,_ = box1
    x1_, y1_, w1_, h1_,_ = box2
    x2, y2 = x1 + w1, y1 + h1
    x2_, y2_ = x1_ + w1_, y1_ + h1_
    xA = max(x1, x1_)
    yA = max(y1, y1_)
    xB = min(x2, x2_)
    yB = min(y2, y2_)
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Calculate union
    box1Area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2Area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)
    unionArea = box1Area + box2Area - interArea

    # Calculate IoU
    iou = interArea / unionArea
    return iou

def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)


    # coordinates of bounding boxes
    l_x = boxes[:, 0]
    l_y = boxes[:, 1]
    r_x = boxes[:, 2] + l_x
    r_y = boxes[:, 3] + l_y
    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (r_x - l_x + 1) * (r_y - l_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)
    indices = []

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]
        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        indices.append(index)

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(l_x[index], l_x[order[:-1]])
        x2 = np.minimum(r_x[index], r_x[order[:-1]])
        y1 = np.maximum(l_y[index], l_y[order[:-1]])
        y2 = np.minimum(r_y[index], r_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]
    # x_center, y_center = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2
    # x_center = x_center[indices]
    # y_center = y_center[indices]
    # for i in range(len(indices)):
    #     for j in range(len(indices)):
    #         if i == j:
    #             continue
    #         if abs(y_center[i] - y_center[j]) <=1 and iou(bounding_boxes[indices[i]], bounding_boxes[indices[j]]) > 0:
    #             if confidence_score[i] > confidence_score[j]:
    #                 indices[j] = -1
    #             else:
    #                 indices[i] = -1
    # new_indices = []
    # for i in range(len(indices)):
    #     if indices[i] != -1:
    #         new_indices.append(indices[i])
    return picked_boxes, picked_score, indices

def load_img_tensor(img_path):
    img = Image.open(img_path)
    img.save("test.png")
    # img = np.array(img).astype(np.float32)/255.0
    print(img_path)
    img = cv2.imread("test.png").astype(np.float32)/255.0
    img = cv2.resize(img, (416, 416))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img

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