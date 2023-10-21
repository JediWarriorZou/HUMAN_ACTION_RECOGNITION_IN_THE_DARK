import cv2
import random
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from LIME import LIME
from torch import Tensor as ts
from torchvision.models import resnet50, ResNet50_Weights


def frame_sampling(type, path, sp_rate):
    v_capture = cv2.VideoCapture(path)
    frame_list = []
    if type =='uniform':
        c = 1
        while (True):
            ret, frame = v_capture.read()
            if ret:
                if c % sp_rate == 0:
                   frame_list.append(cv2.resize(frame,(256,256)))
                c += 1
            else:
                break
        v_capture.release()
        
    if  type =='random':
        idx_list = []
        frame_num = int(v_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        select_num = math.floor(frame_num / sp_rate)
        while True:
            frame_index = random.randint(1, frame_num + 1)
            v_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = v_capture.read()
            
            if frame is not None:
            # Check if the frame is already selected to avoid duplicates   
                if frame_index not in idx_list:
                    idx_list.append(frame_index)
            else:
                continue
            # Break the loop when you've selected the desired number of frames
            if len(idx_list) >= int(select_num):
                break
        for i in range(select_num):
            v_capture.set(cv2.CAP_PROP_POS_FRAMES, idx_list[i])
            ret, frame = v_capture.read()
            frame_list.append(cv2.resize(frame,(256,256)))
        v_capture.release()
    return frame_list 

def frame_sampling_enhancement(type, path, sp_rate):
    v_capture = cv2.VideoCapture(path)
    frame_list = []
    if type =='uniform':
        c = 1
        while (True):
            ret, frame = v_capture.read()
            if ret:
                if c % sp_rate == 0:
                   enhancor = LIME(img = frame / 255)
                   _ = enhancor.optimizeIllumMap()
                   frame_list.append(cv2.resize(enhancor.enhance(),(256,256)))
                c += 1
            else:
                break
        v_capture.release()
        
    if  type =='random':
        idx_list = []
        frame_num = int(v_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        select_num = math.floor(frame_num / sp_rate)
        while True:
            frame_index = random.randint(1, frame_num + 1)
            v_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = v_capture.read()
            
            if frame is not None:
            # Check if the frame is already selected to avoid duplicates   
                if frame_index not in idx_list:
                    idx_list.append(frame_index)
            else:
                continue
            # Break the loop when you've selected the desired number of frames
            if len(idx_list) >= int(select_num):
                break
        for i in range(select_num):
            v_capture.set(cv2.CAP_PROP_POS_FRAMES, idx_list[i])
            ret, frame = v_capture.read()
            enhancor = LIME(img = frame / 255)
            _ = enhancor.optimizeIllumMap()
            frame_list.append(cv2.resize(enhancor.enhance(),(256,256)))
        v_capture.release()
    return frame_list 

def late_fusion(frame_list,**kargs):
    feature_list = []
    preprocess = transforms.Compose([
    transforms.ToTensor(),    #0-255 to 0-1
    transforms.Normalize(mean = kargs['mean'], std = kargs['std'])
    ])
    model = resnet50(weights = ResNet50_Weights.DEFAULT)
    modules = list(model.children())[:-1]      # delete the last fc layer.
    model = nn.Sequential(*modules)
    model.eval()
    
    for i in range(len(frame_list)):
        frame = frame_list[i]
        frame = preprocess(frame)
        frame = frame.unsqueeze(0)
        with torch.no_grad():
            feature = model(frame)
        feature_list.append(feature)
    fusion = torch.sum(torch.stack(feature_list), dim=0)/len(feature_list)
    
    return fusion