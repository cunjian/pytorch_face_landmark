import cv2
import os
import dlib
detector = dlib.get_frontal_face_detector()
img_dir = 'images/'
anno_dir = 'annos/'
data_path = 'data/sort031001_032000/'
def get_rect(pts,w,h):
    x_pts = []
    y_pts = []
    for i in xrange(len(pts)/2):
        x_pts.append(float(pts[2*i]))
        y_pts.append(float(pts[2*i+1]))
    max_x = max(x_pts)
    min_x = min(x_pts)
    max_y = max(y_pts)
    min_y = min(y_pts)
    rec_w = max_x  - min_x
    rec_h = max_y  - min_y
    min_x = min_x - rec_w/2 if min_x - rec_w/2 >0 else 0
    min_y = min_y - rec_h/2 if min_y - rec_h/2 >0 else 0
    max_x = max_x + rec_w/2 if max_x + rec_w/2 <w else w
    max_y = max_y + rec_h/2 if max_y + rec_h/2 <h else h

    return (int(min_x),int(min_y)),(int(max_x),int(max_y))

for img_path in os.listdir(data_path):
    if 'jpg' in img_path:
        img = cv2.imread(os.path.join(data_path,img_path))
        with open(data_path+img_path.split('.')[0]+'.pts') as ptsfile:
            pts = ptsfile.readline().strip().split(' ')
            print len(pts)
        for i in xrange(76):
            cv2.circle(img,(int(float(pts[2*i])),int(float(pts[2*i+1]))),1,(255,0,0))
        rect_1,rect_2 = get_rect(pts,img.shape[1],img.shape[0])
        cv2.rectangle(img,rect_1,rect_2,(0,255,0))
        cv2.imshow('data',img)
        cv2.waitKey()
