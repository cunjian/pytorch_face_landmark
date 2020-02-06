import sys
import os
raw_data_root = '/mnt/lvmhdd1/dataset/face_keypoints/data/'
for data_path in os.listdir(raw_data_root):
    data_path = os.path.join(raw_data_root+data_path)
    cmd = 'cp '+data_path+'/*.jpg /mnt/lvmhdd1/dataset/face_keypoints/images'
    print cmd
    os.system(cmd)
    cmd = 'cp '+data_path+'/*.pts /mnt/lvmhdd1/dataset/face_keypoints/annos'
    print cmd
    os.system(cmd)
