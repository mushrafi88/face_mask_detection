import cv2
import numpy as np
import pandas as pd
import time 
import math

detector = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

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
        left = int(instance["left"] * 300)
        bottom = int(instance["bottom"] * 300)
        right = int(instance["right"] * 300)
        top = int(instance["top"] * 300)
        
        detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]
    
        if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
            cv2.putText(base_img, confidence_score, (int(left*aspect_ratio_x), int(top*aspect_ratio_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.rectangle(base_img, (int(left*aspect_ratio_x), int(top*aspect_ratio_y)), (int(right*aspect_ratio_x), int(bottom*aspect_ratio_y)), (255, 255, 255), 3) #draw rectangle to main image
        
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
