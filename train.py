import torchvision
import cv2
import random
import math
import torch
import torchvision.transforms as transforms
import numpy as np

from torch import Tensor as ts
from torchvision.models import vgg16, VGG16_Weights
from video_dataset import Video_Dataset
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#frame_num=varied
#fps=30
#width=320
#height=240

def frame_sampling(type, path, sp_rate):
    v_capture = cv2.VideoCapture(path)
    frame_list = []
    if type =='uniform':
        c = 1
        while (True):
            ret, frame = v_capture.read()
            if ret:
                if c % sp_rate == 0:
                   frame_list.append(cv2.resize(frame,(224,224)))
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
            
            # Check if the frame is already selected to avoid duplicates
            if frame_index not in idx_list:
                idx_list.append(frame_index)
            
            # Break the loop when you've selected the desired number of frames
            if len(idx_list) >= int(select_num):
                break
        for i in range(select_num):
            v_capture.set(cv2.CAP_PROP_POS_FRAMES, idx_list[i])
            ret, frame = v_capture.read()
            frame_list.append(cv2.resize(frame,(224,224)))
        v_capture.release()
        
    return frame_list 

def late_fusion(frame_list):
    feature_list = []
    preprocess = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize(mean = [0.07, 0.07, 0.07], std = [0.1, 0.09, 0.08])
    ])
    model = vgg16(weights = VGG16_Weights.IMAGENET1K_V1)
    model.eval()
    
    for i in range(len(frame_list)):
        frame = frame_list[i]
        frame = preprocess(frame)
        frame = frame.unsqueeze(0)
        with torch.no_grad():
            feature = model.features(frame)
        feature_list.append(feature)
    fusion = torch.sum(torch.stack(feature_list), dim=0)/len(feature_list)
    
    return fusion
             
if __name__ == '__main__':
    print('-----Training Begin-----')
    train_set = Video_Dataset('data/train','data/train.txt')
    train_feature_set = []
    print('-----Feature Extraction-----')
    for i in range(len(train_set)):
        path, _ = train_set[i]
        frame_list = frame_sampling('uniform', path, 10)
        feature = late_fusion(frame_list)
        train_feature_set.append((feature.view(-1)).numpy())
    
    train_feature_set = np.array(train_feature_set)
    
    print('-----Training SVM Begin-----')
    clf = SVC(kernel='linear')
    clf.fit(train_feature_set, train_set.label_list)
    
    print('-----Testinging Begin-----')
    test_set = Video_Dataset('data/validate','data/validate.txt')
    test_feature_set = []
    print('-----Feature Extraction-----')
    for i in range(len(test_set)):
        path, _ = test_set[i]
        frame_list = frame_sampling('uniform', path, 10)
        feature = late_fusion(frame_list)
        test_feature_set.append((feature.view(-1)).numpy())
    
    test_feature_set = np.array(test_feature_set)
    print('-----Inference-----')
    y_pred = clf.predict(test_feature_set)
    
    accuracy = accuracy_score(test_set.label_list, y_pred)
    confusion = confusion_matrix(test_set.label_list, y_pred)
    report = classification_report(test_set.label_list, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{confusion}')
    print(f'Classification Report:\n{report}')
    
    
    
    



