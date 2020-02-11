# Pytorch Face Landmark Detection
Implementation of face landmark detector with PyTorch. It utilizes MTCNN as a face detector. The model was trained on 68-point landmark annotation from 300W dataset. 

## Inference
Test on a sample folder and save the landmark detection results.
> python3 -W ignore test_batch_mtcnn.py

Optimize with ONNX and test on a camera. Here, the pytorch model has been converted to ONNX.
> python3 -W ignore test_camera_mtcnn_onnx.py


## Visualization Results

![alt text](https://github.com/cunjian/pytorch_face_landmark/blob/master/results/12_Group_Group_12_Group_Group_12_24.jpg "Logo Title Text 1")

## Benchmark Results on 300W

| Algorithms | Common | Challenge | Full Set |
|:-:|:-:|:-:|:-:|
| ResNet18 (224×224) | 3.73 | 7.14 | 4.39 |
| [MobileNetV2 (224×224)](https://drive.google.com/file/d/1w424ZxfBsv7NFwoqynRPNxe43FHABeJV/view?usp=sharing )   | 3.70 | 7.27 | 4.39 |
| [MobileNetV2 (56×56)](https://drive.google.com/file/d/10DyP9GqAATXFj64MmXlet84Ewb4ryP1K/view?usp=sharing) | 4.50 | 8.50 | 5.27 |

## Training:

The following repository has been modified for training. 

https://github.com/lzx1413/pytorch_face_landmark

