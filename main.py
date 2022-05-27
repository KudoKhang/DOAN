import time
import cv2
import numpy as np
import torch
import os
import sys
os.sys.path
sys.path.insert(0, "yolov5")
from yolov5.models.experimental import attempt_load
from yolov5.yolo_utils import non_max_suppression, scale_coords, resize_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load("./Sources/weights/detect_orange.pt", map_location=device)

size_convert = 640  
conf_thres = 0.4
iou_thres = 0.4

def check_type(img_path):
    if type(img_path) == str:
        if img_path.endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(img_path)
        else:
            raise Exception("Please input a image file")
    elif type(img_path) == np.ndarray:
        img = img_path
    return img

def classify(bboxs, thresh):
    area = (bboxs[2] - bboxs[0]) * (bboxs[3] - bboxs[1])
    return 1 if area > thresh else 2

def draw(image, bboxs):
    x1, y1, x2, y2 = bboxs
    lv = classify(bboxs, 35000)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(image, 'ORANGE_' + str(lv), (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 255), 2)


def inference(image):
    image = check_type(image)
    img = resize_image(image.copy(), size_convert).to(device)
    with torch.no_grad():
        pred = model(img[None, :])[0]
        det = non_max_suppression(pred, conf_thres, iou_thres)[0] # x1,y1,x2,y2, score, class
        bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], image.shape[:-1]).round().cpu().numpy())# x1, y1, x2, y2    
    return bboxs

def webcam(mode=0):
    # 0: Camera or Path Video
    cap = cv2.VideoCapture(mode)
    while True:
        _, frame = cap.read()
        start = time.time()
        bboxs = inference(frame)
        if len(bboxs) > 0:
            for bb in bboxs:
                draw(frame, bb)

        # FPS
        fps = round(1 / (time.time() - start), 2)
        cv2.putText(frame, "FPS: " + str(fps), (20, 30), cv2.FONT_HERSHEY_COMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, 'TOTAL ORANGE 1: ', (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 0, 255), 2)
        cv2.putText(frame, 'TOTAL ORANGE 2: ', (20, 110), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imshow('RESULT', frame)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('s'):
            cv2.imwrite(str(time.time()) + '.jpg', frame)
        if k == ord('q'):
            break

#TODO
"""
    - Count orange
    - Processing in ROI
""" 

def image():
    root = './Sources/final_data/yolo_orange/test/images/'
    path_name = [name for name in os.listdir(root) if name.endswith('jpg')]
    for name in path_name[:2]:
        img = cv2.imread(root + name)
        bboxs = inference(img)
        draw(img, bboxs)
        cv2.imshow('RESULT', img)
        cv2.waitKey(0)

if __name__ == '__main__':
    webcam('./Sources/Video/3.mp4')