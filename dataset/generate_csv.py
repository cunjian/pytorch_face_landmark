import cv2
import os
import random
anno_root = '/mnt/lvmhdd1/dataset/face_keypoints/annos/'
img_root = '/mnt/lvmhdd1/dataset/face_keypoints/images/'

items = []
for anno_path in os.listdir(anno_root):
    if 'pts' in anno_path:
        with open(os.path.join(anno_root,anno_path)) as anno_file:
            landmarks = anno_file.readline().strip().split(' ')
            if(len(landmarks) == 152):
                items.append(anno_path.split('.')[0]+'.jpg,'+','.join(landmarks)+'\n')
            else:
                print anno_path
random.shuffle(items)
train_items = items[:30000]
val_items = items[30000:]
with open('face_landmark_train.csv','w') as trainfile:
    for item in train_items:
        trainfile.write(item)
with open('face_landmark_val.csv','w') as valfile:
    for item in val_items:
        valfile.write(item)
