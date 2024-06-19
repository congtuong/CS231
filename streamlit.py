import streamlit as st
import torch
import pickle
import cv2
from models import Yolov3VGG
from utils import (
    load_img_tensor,
    _get_prediction,
    visualize_prediction,
)

from utils import *

weights = "/mlcv1/WorkingSpace/Personal/tuongbck/project_231/weights_new_lower/model_yololoss_cuda_40.pt"
anchors = "/mlcv1/WorkingSpace/Personal/tuongbck/project_231/anchors.pkl"

@st.cache_resource()
def load_anchors(_anchors):
    with open(_anchors, "rb") as f:
        anchors = torch.tensor(pickle.load(f))
    return anchors

@st.cache_resource()
def load_model(_weights,_anchors):
    model = Yolov3VGG(_anchors,device='cpu')
    model.load_state_dict(torch.load(_weights))
    print("Model loaded")
    return model

anchors = load_anchors(anchors)
model = load_model(weights,anchors)
print(anchors)
uploaded_file = st.file_uploader("Choose an image...")
nms_thres = st.slider('NMS Threshold', 0.0, 1.0, 0.1)
iou_thres = st.slider('Conf Threshold', 0.0, 1.0, 0.1)
model.eval()
if uploaded_file is not None:
    img = load_img_tensor(uploaded_file)
    with torch.no_grad():
        
        out = model([img])
        pred_bbox, pred_conf,pred_cls = _get_prediction(out, anchors)
        _,_,indices = nms(np.array(pred_bbox[0]), np.array(pred_bbox[0])[:,4], nms_thres)
        result_img, thresholded_list = visualize_prediction(np.array(pred_bbox[0])[indices],pred_cls[0].cpu().view((845,2))[indices],np.array(pred_conf[0].cpu().reshape((845,1)))[indices], img,iou_thres)
        # result_img, thresholded_list = visualize_prediction(np.array(pred_bbox[0]),pred_cls[0].cpu().view((845,2)),np.array(pred_conf[0].cpu().reshape((845,1))), img,0.1)
        # img, thresholded_list = visualize_prediction(np.array(pred_bbox[idx])[indices],pred_cls[idx].cpu().view((845,2))[indices],np.array(pred_conf[idx].cpu().reshape((845,1)))[indices], imgs[idx],threshold)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        result_img = Image.fromarray(result_img)
        st.image(np.array(result_img), caption='Processed Image', use_column_width=True)



