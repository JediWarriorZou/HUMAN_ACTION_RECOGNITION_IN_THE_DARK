import argparse
import numpy as np

from utils import frame_sampling, late_fusion, frame_sampling_enhancement
from video_dataset import Video_Dataset
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#frame_num=varied
#fps=30
#width=320
#height=240

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enhancement",
        type=bool,
        nargs="?",
        default= False,
        help="image enhancement"
    )
    parser.add_argument(
        "--sample",
        type=str,
        nargs="?",
        default= 'uniform',
        help="sampling method"
    )
    
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_args()
    print('-----Training Begin-----')
    train_set = Video_Dataset('data/train','data/train.txt')
    train_feature_set = []
    print('-----Feature Extraction-----')
    for i in range(len(train_set)):
        path, _ = train_set[i]
        if opt.enhancement:
            frame_list = frame_sampling_enhancement(opt.sample, path, 10, (256,256))
            feature = late_fusion(frame_list, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        else: 
            frame_list = frame_sampling(opt.sample, path, 10, (256,256))
            feature = late_fusion(frame_list, mean = [0.07, 0.07, 0.07], std = [0.1, 0.09, 0.08])
        train_feature_set.append((feature.view(-1)).numpy())
    
    train_feature_set = np.array(train_feature_set)
    
    print('-----Training SVM Begin-----')
    clf = SVC(gamma='scale', C = 1.0, decision_function_shape='ovr',kernel = 'rbf')
    clf.fit(train_feature_set, train_set.label_list)
    
    print('-----Testinging Begin-----')
    test_set = Video_Dataset('data/validate','data/validate.txt')
    test_feature_set = []
    print('-----Feature Extraction-----')
    for i in range(len(test_set)):
        path, _ = test_set[i]
        if opt.enhancement:
            frame_list = frame_sampling_enhancement(opt.sample, path, 10, (256,256))
            feature = late_fusion(frame_list, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        else: 
            frame_list = frame_sampling(opt.sample, path, 10, (256,256))
            feature = late_fusion(frame_list, mean = [0.07, 0.07, 0.07], std = [0.1, 0.09, 0.08])
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
    
    with open('results.txt','+a') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Confusion Matrix:\n{confusion}\n')
        f.write(f'Classification Report:\n{report}')
    f.close()
      
    
    
    



