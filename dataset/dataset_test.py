from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from FaceLandmarksDataset import FaceLandmarksDataset
from FaceLandmarksDataset import SmartRandomCrop
from FaceLandmarksDataset import Rescale

# Ignore warnings
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
landmarks_frame = pd.read_csv('face_landmarks.csv')

n = 65
img_name = landmarks_frame.ix[n, 0]
landmarks = landmarks_frame.ix[n, 1:].as_matrix().astype('float')
landmarks = landmarks.reshape(-1, 2)
max_xy = np.max(landmarks,axis=0)
min_xy = np.min(landmarks,axis=0)
print(max_xy)
print(min_xy)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))
face_dataset = FaceLandmarksDataset(csv_file='face_landmarks.csv',
                                    root_dir='data/image/')

fig = plt.figure()
crop = SmartRandomCrop()
scale = Rescale((256,256))
composed = transforms.Compose([SmartRandomCrop(),])

for i in range(len(face_dataset)):
    sample = face_dataset[i]
    sample = crop(sample)
    sample = scale(sample)

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break

