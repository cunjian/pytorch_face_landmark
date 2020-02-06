# Pytorch Face Landmark Detection
Implementation of face landmark detector with PyTorch. It utilizes the MTCNN as the face detector. The model was trained on 68-point landmark annotation of 300W dataset only and it generalizes well to unseen datasets. 

## Test on a sample folder
python3 test_batch_mtcnn.py

## Test on the camera
python3 test_camera_mtcnn.py

# Visualization Results

![alt text](https://github.com/cunjian/pytorch_face_landmark/blob/master/results/12_Group_Group_12_Group_Group_12_24.jpg "Logo Title Text 1")

# Benchmark Results

| Algorithms | Common | Challenge | Full Set |
|:-:|:-:|:-:|:-:|
| ResNet18 | 3.73 | 7.14 | 4.39 |
| MobileNetV2 | 3.97 | 8.54 | 4.85 |


# Training:

The following repository has been modified for training. 

https://github.com/lzx1413/pytorch_face_landmark

