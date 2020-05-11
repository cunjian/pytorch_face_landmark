# Pytorch Face Landmark Detection
Implementation of face landmark detection with PyTorch. The model was trained using coordinate-based regression methods. A [video demo](https://lnkd.in/eH27JcP) was displayed here. 

* Support 68-point and 39-point landmark inference.
* Support different backbone networks.
* Support ONNX inference. 

## Inference
Test on a sample folder and save the landmark detection results.
> python3 -W ignore test_batch_mtcnn.py

Optimize with ONNX and test on a camera with MTCNN as a face detector. 
> python3 -W ignore test_camera_mtcnn_onnx.py

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
| PFLD_ExternalData (112×112)  | 3.57 | 6.23 | 4.08 | 1.25 | 0.16 ([onnx](https://drive.google.com/file/d/1zLC2mPtv4vfQZjkzj21F4OJw16vdyk_k/view?usp=sharing))|
> Note: SE (squeeze-and-excitation module), RE (random erasing module).

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
 
## Datasets:

* [300W (68-point)](https://ibug.doc.ic.ac.uk/resources/300-W/), [Menpo (68-point)](https://ibug.doc.ic.ac.uk/resources/2nd-facial-landmark-tracking-competition-menpo-ben/), [300-VW (68-point)](https://ibug.doc.ic.ac.uk/resources/300-VW/)
* [WFLW (98-point)](https://wywu.github.io/projects/LAB/WFLW.html)
* [JD (106-point)](https://facial-landmarks-localization-challenge.github.io/)


## References:
* https://github.com/rwightman/pytorch-image-models
* https://github.com/Res2Net/Res2Net-PretrainedModels
* https://github.com/HRNet/HRNet-Image-Classification
* https://github.com/lzx1413/pytorch_face_landmark
* https://github.com/polarisZhao/PFLD-pytorch


