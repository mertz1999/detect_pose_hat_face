from inc.process import MainProcess
import numpy as np
import cv2
import os
import torch

# Process class instance
main_process = MainProcess(hat_model='./models/hat_detection.pt', 
                           face_model='./models/face_detection.pt', 
                           pose_model='./models/pose_detection.pt',
                           )

# Main loop
video_path = './inputs/2.mp4'
cap        = cv2.VideoCapture(video_path)
width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps        = int(cap.get(cv2.CAP_PROP_FPS))

fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
output_path = './result.mp4'
out         = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

count = -1
while cap.isOpened():
    count += 1
    ret, img = cap.read()
    if not ret:
        break
    img_copy = img.copy()

    face_box    = main_process.face_model(img_copy)   
    face_box    = face_box[0].boxes.xyxy.detach().cpu().numpy()

    persons_box = main_process.pose_model(img_copy)
    persons_box = persons_box[0].boxes.xyxy.detach().cpu().numpy()
    pose_lands  = persons_box[0].keypoints.xy.detach().cpu().numpy()

    img_copy = main_process.draw_person(img_copy, persons_box, face_box)
    img_copy = main_process.draw_face(img_copy, face_box)
    img_copy = main_process.draw_pose(img_copy, pose_lands)

    break