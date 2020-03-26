"""
This code uses the onnx model to detect faces from live video or cameras.
Use a much faster face detector: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
Date: 3/26/2020 by Cunjian Chen (ccunjian@gmail.com)
"""
import time
import cv2
import numpy as np
import onnx
import vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend

# onnx runtime
import onnxruntime as ort

# import libraries for landmark
from common.utils import BBox,drawLandmark,drawLandmark_multiple
from PIL import Image
import torchvision.transforms as transforms

# setup the parameters
resize = transforms.Resize([56, 56])
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
# import the landmark detection models
import onnx
import onnxruntime
onnx_model_landmark = onnx.load("landmark_detection_56_se_external.onnx")
onnx.checker.check_model(onnx_model_landmark)
ort_session_landmark = onnxruntime.InferenceSession("landmark_detection_56_se_external.onnx")
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# face detection setting
def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


label_path = "models/voc-model-labels.txt"

onnx_path = "models/onnx/version-RFB-320.onnx"
class_names = [name.strip() for name in open(label_path).readlines()]

predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
onnx.helper.printable_graph(predictor.graph)
predictor = backend.prepare(predictor, device="CPU")  # default CPU

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

# perform face detection and alignment from camera
cap = cv2.VideoCapture(0)  # capture from camera
threshold = 0.7

sum = 0
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        print("no img")
        break
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    # image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    # confidences, boxes = predictor.run(image)
    time_time = time.time()
    confidences, boxes = ort_session.run(None, {input_name: image})
    print("cost time:{}".format(time.time() - time_time))
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"

        #cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        # perform landmark detection
        out_size = 56
        img=orig_image.copy()
        height,width,_=img.shape
        x1=box[0]
        y1=box[1]
        x2=box[2]
        y2=box[3]
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
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)    
        cropped_face = Image.fromarray(cropped_face)
        test_face = resize(cropped_face)
        test_face = to_tensor(test_face)
        test_face = normalize(test_face)
        test_face.unsqueeze_(0)

        start = time.time()             
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_face)}
        ort_outs = ort_session_landmark.run(None, ort_inputs)
        end = time.time()
        print('Time: {:.6f}s.'.format(end - start))
        landmark = ort_outs[0]
        landmark = landmark.reshape(-1,2)
        landmark = new_bbox.reprojectLandmark(landmark)
        orig_image = drawLandmark_multiple(orig_image, new_bbox, landmark)

    sum += boxes.shape[0]
    orig_image = cv2.resize(orig_image, (0, 0), fx=0.7, fy=0.7)
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("sum:{}".format(sum))
