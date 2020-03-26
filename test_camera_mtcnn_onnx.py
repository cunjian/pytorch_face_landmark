# Face alignment demo
# Cunjian Chen (cunjian@msu.edu)
from __future__ import division
import argparse
import torch
import torch.onnx
import torchvision.transforms as transforms

import os
import cv2
import numpy as np
#import dlib
from common.utils import BBox,drawLandmark,drawLandmark_multiple
from models.basenet import MobileNet_GDConv_56
import matplotlib.pyplot as plt
from src import detect_faces
from PIL import Image
import time

parser = argparse.ArgumentParser(description='PyTorch face landmark')
# Datasets
parser.add_argument('-img', '--image', default='face76', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu_id', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('-c', '--checkpoint', default='checkpoint/mobilenet_56_model_best_gdconv.pth.tar', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

args = parser.parse_args()
mean = np.asarray([ 0.485, 0.456, 0.406 ])
std = np.asarray([ 0.229, 0.224, 0.225 ])
resize = transforms.Resize([56, 56])
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

def load_model():
    model = MobileNet_GDConv_56(136)
    checkpoint = torch.load(args.checkpoint, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__ == '__main__':
    import onnx

    onnx_model = onnx.load("landmark_detection_56_se_external.onnx")
    onnx.checker.check_model(onnx_model)

    import onnxruntime

    ort_session = onnxruntime.InferenceSession("landmark_detection_56_se_external.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    out_size = 56
    #model = load_model()
    #model = model.eval()
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    while success:
        success, img = cap.read()
        height,width,_=img.shape
        # perform face detection using MTCNN
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        faces, landmarks = detect_faces(image)

        ratio=0
        if len(faces)==0:
            print('NO face is detected!')
            continue
        for k, face in enumerate(faces): 
            x1=face[0]
            y1=face[1]
            x2=face[2]
            y2=face[3]
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            size = int(max([w, h])*1.1)
            cx = x1 + w//2
            cy = y1 + h//2
            x1 = cx - size//2
            x2 = x1 + size
            y1 = cy - size//2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)
            new_bbox = list(map(int, [x1, x2, y1, y2]))
            new_bbox = BBox(new_bbox)
            cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)            
            cropped_face = cv2.resize(cropped, (out_size, out_size))

            if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
                continue
            #test_face = cv2.resize(cropped_face,(out_size,out_size))
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            cropped_face = Image.fromarray(cropped_face)
            test_face = resize(cropped_face)
            test_face = to_tensor(test_face)
            test_face = normalize(test_face)
            test_face.unsqueeze_(0)

            start = time.time()             
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_face)}
            ort_outs = ort_session.run(None, ort_inputs)
            end = time.time()
            print('Time: {:.6f}s.'.format(end - start))
            landmark = ort_outs[0]
            #print(landmark)

            landmark = landmark.reshape(-1,2)
            landmark = new_bbox.reprojectLandmark(landmark)
            img = drawLandmark_multiple(img, new_bbox, landmark)
        cv2.imshow('Face Alignment Demo', img)
        cv2.waitKey(30)
        if cv2.waitKey(10) == 27: # Esc key to stop
            break
    cap.release()
    cv2.destroyAllWindows()

