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

size_convert = 640  # setup size de day qua model
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

def draw(image, bboxs):
    cv2.rectangle(image, (bboxs[0], bboxs[1]), (bboxs[2], bboxs[3]), (255,255,255), 2)

def inference(image):
    image = check_type(image)
    img = resize_image(image.copy(), size_convert).to(device)

    with torch.no_grad():
        pred = model(img[None, :])[0]
        det = non_max_suppression(pred, conf_thres, iou_thres)[0] # x1,y1,x2,y2, score, class
        bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], image.shape[:-1]).round().cpu().numpy())# x1, y1, x2, y2    
    return bboxs

def webcam():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        start = time.time()
        bboxs = inference(frame)
        if len(bboxs) > 0:
            for bb in bboxs:
                draw(frame, bb)

        # FPS
        fps = round(1 / (time.time() - start), 2)
        cv2.putText(frame, "FPS: " + str(fps), (100, 50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow('RESULT', frame)

        k = cv2.waitKey(20) & 0xFF
        if k == ord('s'):
            cv2.imwrite(str(time.time()) + '.jpg', frame)
        if k == ord('q'):
            break


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
    webcam()