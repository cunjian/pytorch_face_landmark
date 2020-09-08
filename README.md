# Pytorch Face Landmark Detection
Implementation of face landmark detection with PyTorch. The models were trained using coordinate-based or heatmap-based regression methods. Different face detetors were supported. A [video demo](https://lnkd.in/eH27JcP) and [image detection](https://github.com/cunjian/pytorch_face_landmark/tree/master/results) results were displayed here. 

* Support 68-point and 39-point landmark inference.
* Support different backbone networks and face detectors.
* Support ONNX inference. 
* Support heatmap-based inference.

## Inference
Test on a sample folder and save the landmark detection results. Support different backbones and face detectors.
> python3 test_batch_detections.py --backbone MobileFaceNet --detector Retinaface
* backbone: MobileNet/PFLD/MobileFaceNet; detector: MTCNN/FaceBoxes/Retinaface

Optimize with ONNX and test on a camera with MTCNN as a face detector. 
> python3 test_camera_mtcnn_onnx.py

Optimize with ONNX and test on a camera with a lightweight face detector. It can achieve **real-time speed on CPU**.
> python3 test_camera_light_onnx.py

## Benchmark Results on 300W

* Inter-ocular Normalization (ION)

| Algorithms | Common | Challenge | Full Set | Param # (M) | 
|:-:|:-:|:-:|:-:|:-:|
| ResNet18 (224×224)  |3.73 | 7.14 | 4.39 | 11.76 | 
| Res2Net50_v1b (224×224)  |3.43 | 6.77 | 4.07 | 26.00 | 
| Res2Net50_v1b_SE (224×224)  |3.37 | 6.67 | 4.01| 27.05 |
| Res2Net50_v1b_ExternalData (224×224)  |3.30 | 5.92 | 3.81 | 26.00 | 
| HRNet_w18_small_v2 (224×224)  | 3.57 | 6.85 | 4.20 | 13.83 | 

* Inter-ocular Normalization (ION) with Lightweight Models

| Algorithms | Common | Challenge | Full Set | Param # (M) | CPU Inference (s) |
|:-:|:-:|:-:|:-:|:-:|:-:|
| [MobileNetV2 (224×224)](https://drive.google.com/file/d/1w424ZxfBsv7NFwoqynRPNxe43FHABeJV/view?usp=sharing )    |3.70 | 7.27 | 4.39 | 3.74 | 1.2|
| MobileNetV2_SE (224×224)  | 3.63 | 7.01 | 4.28 | 4.15 | /|
| [MobileNetV2_SE_RE (224×224)](https://drive.google.com/file/d/18ADLfuucnNhJyNIA3p0WJLR8J3-An_OG/view?usp=sharing)  | 3.63 | 6.66 | 4.21 | 4.15 | /|
| [MobileNetV2_ExternalData (224×224)](https://drive.google.com/file/d/1Le5UdpMkKOTRr1sTp4lwkw8263sbgdSe/view?usp=sharing)   |3.48 | 6.0 | 3.96 | 3.74 | 1.2|
| [MobileNetV2 (56×56)](https://drive.google.com/file/d/10DyP9GqAATXFj64MmXlet84Ewb4ryP1K/view?usp=sharing)  |4.50 | 8.50 | 5.27 | 3.74 | 0.01 ([onnx](https://drive.google.com/file/d/1UkJfsY1Y00IhxuGS-mEZkfKC3ekfFI3G/view?usp=sharing))|
| [MobileNetV2_SE_ExternalData (56×56)](https://drive.google.com/file/d/1BcfUVGPHlILLlWN4h6E9lbwtz85PUbuW/view?usp=sharing)  |4.10 | 6.89 | 4.64 | 4.10 | 0.01 ([onnx](https://drive.google.com/file/d/1Kw-OKKAzoPxg1hVMvdtCbnCw2GWNC85q/view?usp=sharing))|
| [PFLD_ExternalData (112×112)](https://drive.google.com/file/d/1gjgtm6qaBQJ_EY7lQfQj3EuMJCVg9lVu/view?usp=sharing)  | 3.49 | 6.01 | 3.97 | 0.73 | 0.01 ([onnx](https://drive.google.com/file/d/1me3-AC6rVcvVyyxNP7FxqdAN5SoDTj95/view?usp=sharing))|
| [MobileFaceNet_ExternalData (112×112)](https://drive.google.com/file/d/1T8J73UTcB25BEJ_ObAJczCkyGKW5VaeY/view?usp=sharing)  | 3.30 | 5.69 | 3.76 | 1.01 | / |

> Note: SE (squeeze-and-excitation module), RE (random erasing module).

* Heatmap Inference (still under test)

| Algorithms | Common | Challenge | Full Set | Param # (M) | 
|:-:|:-:|:-:|:-:|:-:|
| Hourglass2  | 3.06 | 5.54 | 3.55 | 8.73 | 

## Visualization Results
* Face alignment on 300W dataset

![img1](https://github.com/cunjian/pytorch_face_landmark/blob/master/imgs/300w.png)

* Semi-frontal face alignment on Menpo dataset

![img1](https://github.com/cunjian/pytorch_face_landmark/blob/master/imgs/menpo_semi_frontal.png)

* Profile face alignment on Menpo dataset

![img1](https://github.com/cunjian/pytorch_face_landmark/blob/master/imgs/menpo_profile.png)


## TODO
The following features will be added soon. 
- Still to come:
  * [x] Support for the 39-point detection
  * [ ] Support for the 106 point detection
  * [ ] Support for heatmap-based inferences
 
## Public Datasets:

* 68-point: [300W](https://ibug.doc.ic.ac.uk/resources/300-W/), [Menpo](https://ibug.doc.ic.ac.uk/resources/2nd-facial-landmark-tracking-competition-menpo-ben/), [300-VW](https://ibug.doc.ic.ac.uk/resources/300-VW/), [300W-LP](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm), [300W-Style](https://github.com/D-X-Y/landmark-detection/tree/master/SAN)
* 98-point: [WFLW](https://wywu.github.io/projects/LAB/WFLW.html)
* 106-point: [JD](https://facial-landmarks-localization-challenge.github.io/)


## References:
* https://github.com/biubug6/Pytorch_Retinaface
* https://github.com/cleardusk/3DDFA_V2
* https://github.com/lzx1413/pytorch_face_landmark
* https://github.com/polarisZhao/PFLD-pytorch


