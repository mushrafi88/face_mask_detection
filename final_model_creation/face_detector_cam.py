import cv2
import numpy as np
import pandas as pd
import time 
import math
import argparse
import os
import glob
import random
import darknet
import time


detector = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
random.seed(3)
network, class_names, class_colors = darknet.load_network('cfg/yolov4-obj.cfg','data/obj.data','yolov4-obj_best.weights')

def mask_detection(frame, network, class_names, class_colors, thresh):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    image = frame.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    return detections

def process_frames_from_video(frame):
    base_img = frame.copy()
    original_size = frame.shape
    target_size = (300,300)
    frame = cv2.resize(frame, target_size)
    aspect_ratio_x = (original_size[1] / target_size[1])
    aspect_ratio_y = (original_size[0] / target_size[0])
    imageBlob = cv2.dnn.blobFromImage(frame)
    detector.setInput(imageBlob)
    detections = detector.forward()
    detections_df = pd.DataFrame(detections[0][0], columns = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"])
    detections_df = detections_df[detections_df['is_face'] == 1] #0: background, 1: face
    detections_df = detections_df[detections_df['confidence'] >= 0.50]

    
    for i, instance in detections_df.iterrows():
        confidence_score = str(round(100*instance["confidence"], 2))+" %"
        left = int(instance["left"] * 300 )
        bottom = int(instance["bottom"] * 300 )
        right = int(instance["right"] * 300  )
        top = int(instance["top"] * 300 )
        detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y*1.08), int(left*aspect_ratio_x- left*aspect_ratio_x*0.08):int(right*aspect_ratio_x*1.08)]
        detections = mask_detection(detected_face, network, class_names, class_colors, 0.91)

        if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
            if len(detections) == 1:
                cv2.putText(base_img, 'mask detected', (int(left*aspect_ratio_x), int(top*aspect_ratio_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.rectangle(base_img, (int(left*aspect_ratio_x), int(top*aspect_ratio_y)), (int(right*aspect_ratio_x), int(bottom*aspect_ratio_y)), (0, 255, 0), 3) #draw rectangle to main image
            elif len(detections) == 0:
                cv2.putText(base_img, 'not wearing mask', (int(left*aspect_ratio_x), int(top*aspect_ratio_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.rectangle(base_img, (int(left*aspect_ratio_x), int(top*aspect_ratio_y)), (int(right*aspect_ratio_x), int(bottom*aspect_ratio_y)), (0, 0, 255), 3) #draw rectangle to main image
    return base_img

def face_detector(filename):
    cap = cv2.VideoCapture(filename)
    while True:
        ret, next_frame = cap.read() 
        
        if ret == False: break
        next_frame = process_frames_from_video(next_frame)
        
        cv2.imshow('frame',next_frame)
        key = cv2.waitKey(50)
        
        if key == 27: # Hit ESC key to stop
            break
    
    cap.release()
    cv2.destroyAllWindows()
        
face_detector(1)
