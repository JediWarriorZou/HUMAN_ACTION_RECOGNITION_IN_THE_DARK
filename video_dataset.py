import os

class Video_Dataset:
    def __init__(self,video_path, label_path):
        video_path = video_path
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