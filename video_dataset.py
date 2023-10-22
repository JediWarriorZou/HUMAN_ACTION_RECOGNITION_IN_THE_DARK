import os
from torch.utils import data
import torch
import cv2
import random
import math
torch




from tqdm import tqdm


class Video_Dataset:
    def __init__(self,video_path, label_path):
        text_list = []
        self.video_list = []
        self.label_list = []
        with open(label_path,"r") as f:
            text_list = f.readlines()
        f.close()
        for i in range(len(text_list)):
            self.video_list.append(os.path.join(video_path, text_list[i].split()[2]))
            self.label_list.append(int(text_list[i].split()[1]))
            
    def __getitem__(self,index):
        label = self.label_list[index]
        video = self.video_list[index]
        return video, label

    def __len__(self):
        return len(self.label_list)
    
# for 3DCNN
class Dataset_3DCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, video_path, label_path, sampling_rate, resize, transform=None):
        "Initialization"
        text_list = []
        self.sp_rate = sampling_rate
        self.transfrom =transform
        self.video_list = []
        self.label_list = []
        self.resize = resize
        with open(label_path,"r") as f:
            text_list = f.readlines()
        f.close()
        for i in range(len(text_list)):
            self.video_list.append(os.path.join(video_path, text_list[i].split()[2]))
            self.label_list.append(int(text_list[i].split()[1]))
        

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.label_list)

    def frame_sampling(self, path, sp_rate, resize):
        v_capture = cv2.VideoCapture(path)
        frame_list = []
        idx_list = []
        frame_num = int(v_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        select_num = sp_rate
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
        idx_list.sort()
        for i in range(select_num):
            v_capture.set(cv2.CAP_PROP_POS_FRAMES, idx_list[i])
            ret, frame = v_capture.read()
            gray_frame = cv2.cvtColor(cv2.resize(frame, resize), cv2.COLOR_BGR2GRAY) #RGB to GRAYSCALE
            frame_list.append(gray_frame)
        v_capture.release()
        return frame_list 
    

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        video_path = self.video_list[index]
        frames = self.frame_sampling(video_path, self.sp_rate, self.resize)
        # Load data
        p_frames = []
        for f in frames:
            frame = self.transfrom(f)
            p_frames.append(frame.squeeze_(0))
            
        X = torch.stack(p_frames, dim=0).unsqueeze_(0)
        y = torch.LongTensor([self.label_list[index]])                             # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y