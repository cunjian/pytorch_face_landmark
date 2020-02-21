# Pytorch Face Landmark Detection
Implementation of face landmark detection with PyTorch. The model was trained using coordinate-based regression methods. 

* Support 68-point and 39-point landmark inference.
* Support different backbone networks.
* Support ONNX inference. 

## Inference
Test on a sample folder and save the landmark detection results.
> python3 -W ignore test_batch_mtcnn.py

Optimize with ONNX and test on a camera. The pytorch model has been converted to ONNX for fast inference.
> python3 -W ignore test_camera_mtcnn_onnx.py

## Benchmark Results on 300W

* Inter-ocular Normalization (ION)

| Algorithms | Common | Challenge | Full Set | CPU Inference (s)
|:-:|:-:|:-:|:-:|:-:|
| ResNet18 (224×224) | 3.73 | 7.14 | 4.39 | / |
| [MobileNetV2 (224×224)](https://drive.google.com/file/d/1w424ZxfBsv7NFwoqynRPNxe43FHABeJV/view?usp=sharing )   | 3.70 | 7.27 | 4.39 | 1.2 |
| [MobileNetV2 (56×56)](https://drive.google.com/file/d/10DyP9GqAATXFj64MmXlet84Ewb4ryP1K/view?usp=sharing) | 4.50 | 8.50 | 5.27 | 0.01 (onnx) |
| [MobileNetV2_External (224×224)](https://drive.google.com/file/d/1Le5UdpMkKOTRr1sTp4lwkw8263sbgdSe/view?usp=sharing)  | 3.48 | 6.0 | 3.96 | 1.2 |


## Visualization Results
* Face alignment on 300W dataset
![img1](https://github.com/cunjian/pytorch_face_landmark/blob/master/imgs/300w.png)

* Semi-frontal face alignment on Menpo dataset
![img1](https://github.com/cunjian/pytorch_face_landmark/blob/master/imgs/menpo_semi_frontal.png)

* Profile face alignment on Menpo dataset
![img1](https://github.com/cunjian/pytorch_face_landmark/blob/master/imgs/menpo_profile.png)

* With MTCNN as a detector
![img1](https://github.com/cunjian/pytorch_face_landmark/blob/master/results/12_Group_Team_Organized_Group_12_Group_Team_Organized_Group_12_50.jpg)

## TODO
The following features will be added soon. 
- Still to come:
  * [x] Support for the 39-point detection
  * [ ] Support for the 106 point detection
  * [ ] Support for heatmap-based inferences
 


## References:

* https://github.com/lzx1413/pytorch_face_landmark
* https://github.com/polarisZhao/PFLD-pytorch


