# coding: utf-8

import cv2
import numpy as np

class BBox(object):
    # bbox is a list of [left, right, top, bottom]
    def __init__(self, bbox):
        self.left = bbox[0]
        self.right = bbox[1]
        self.top = bbox[2]
        self.bottom = bbox[3]
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]
        self.h = bbox[3] - bbox[2]

    # scale to [0,1]
    def projectLandmark(self, landmark):
        landmark_= np.asarray(np.zeros(landmark.shape))     
        for i, point in enumerate(landmark):
            landmark_[i] = ((point[0]-self.x)/self.w, (point[1]-self.y)/self.h)
        return landmark_

    # landmark of (5L, 2L) from [0,1] to real range
    def reprojectLandmark(self, landmark):
        landmark_= np.asarray(np.zeros(landmark.shape)) 
        for i, point in enumerate(landmark):
            x = point[0] * self.w + self.x
            y = point[1] * self.h + self.y
            landmark_[i] = (x, y)
        return landmark_

def drawLandmark(img, bbox, landmark):
    '''
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    '''
    img_ = img.copy()
    cv2.rectangle(img_, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img_, (int(x), int(y)), 3, (0,255,0), -1)
    return img_

def drawLandmark_multiple(img, bbox, landmark):
    '''
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    '''
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
    return img

def drawLandmark_Attribute(img, bbox, landmark,gender,age):
    '''
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    '''
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 3, (0,255,0), -1)
        if gender.argmax()==0:
                # -1->female, 1->male; -1->old, 1->young
                cv2.putText(img, 'female', (int(bbox.left), int(bbox.top)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        else:
                cv2.putText(img, 'male', (int(bbox.left), int(bbox.top)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),3)
        if age.argmax()==0:
                cv2.putText(img, 'old', (int(bbox.right), int(bbox.bottom)),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), 3)
        else:
                cv2.putText(img, 'young', (int(bbox.right), int(bbox.bottom)),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), 3)
    return img


def drawLandmark_only(img, landmark):
    '''
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    '''
    img_=img.copy()
    #cv2.rectangle(img_, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img_, (int(x), int(y)), 3, (0,255,0), -1)
    return img_


def processImage(imgs):
    '''
    Subtract mean and normalize, imgs [N, 1, W, H]
    '''
    imgs = imgs.astype(np.float32)
    for i, img in enumerate(imgs):
        m = img.mean()
        s = img.std()
        imgs[i] = (img-m)/s
    return imgs

def flip(face, landmark):
    '''
    flip a face and its landmark
    '''
    face_ = cv2.flip(face, 1) # 1 means flip horizontal
    landmark_flip = np.asarray(np.zeros(landmark.shape))
    for i, point in enumerate(landmark):
        landmark_flip[i] = (1-point[0], point[1])
    # for 5-point flip
        landmark_flip[[0,1]] = landmark_flip[[1,0]]
    landmark_flip[[3,4]] = landmark_flip[[4,3]]
    # for 19-point flip
        #landmark_flip[[0,9]] = landmark_flip[[9,0]]
    #landmark_flip[[1,8]] = landmark_flip[[8,1]]
    #landmark_flip[[2,7]] = landmark_flip[[7,2]]
    #landmark_flip[[3,6]] = landmark_flip[[6,3]]
    #landmark_flip[[4,11]] = landmark_flip[[11,4]]
    #landmark_flip[[5,10]] = landmark_flip[[10,5]]
    #landmark_flip[[12,14]] = landmark_flip[[14,12]]
    #landmark_flip[[15,17]] = landmark_flip[[17,15]]
    return (face_, landmark_flip)

def scale(landmark):
    '''
    scale the landmark from [0,1] to [-1,1]
    '''
    landmark_ = np.asarray(np.zeros(landmark.shape))
    lanmark_=(landmark-0.5)*2
    return landmark_

def check_bbox(img, bbox):
    '''
    Check whether bbox is out of the range of the image
    '''
    img_w, img_h = img.shape
    if bbox.x > 0 and bbox.y > 0 and bbox.right < img_w and bbox.bottom < img_h:
        return True
    else:
        return False

def rotate(img, bbox, landmark, alpha):
    """
        given a face with bbox and landmark, rotate with alpha
        and return rotated face with bbox, landmark (absolute position)
    """
    center = ((bbox.left+bbox.right)/2, (bbox.top+bbox.bottom)/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, img.shape)
    landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
    face = img_rotated_by_alpha[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
    return (face, landmark_)
