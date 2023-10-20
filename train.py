import torchvision
import cv2
import random
import math
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
                   frame_list.append(frame)
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
            frame_list.append(frame)
        v_capture.release()
        
    return frame_list
        
             
               
           

video_path = 'data/train/Jump/Jump_8_7.mp4'



