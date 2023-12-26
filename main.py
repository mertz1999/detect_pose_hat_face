from inc.process import MainProcess
from parser_args import parser_
import tensorflow as tf
import numpy as np
import torch
import time
import cv2
import os

'''
webcam setting : 
    python main.py --webcam
video with skip: 
    python main.py --video './inputs/3.mp4' --skip 3
if you want to show the results:
    python main.py --webcam --show

'''


args = parser_.parse_args()
hat_model  = args.hat
face_model = args.face
pose_model = args.pose
show       = args.show
skip       = args.skip
webcam     = args.webcam
video_path = 0 if webcam else args.video    

# Process class instance
main_process = MainProcess(hat_model  =hat_model, 
                           face_model =face_model, 
                           pose_model =pose_model,
                           )

# Main loop
cap        = cv2.VideoCapture(video_path)
width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps        = int(cap.get(cv2.CAP_PROP_FPS))

fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
output_path = './outputs/result.mp4'
out         = cv2.VideoWriter(output_path, fourcc, fps//skip, (width, height))

count = -1
with tf.device(tf.DeviceSpec(device_type="CPU", device_index=0)):
    while cap.isOpened():
        # if count > 30*5:
        #     break
        count += 1
        ret, img = cap.read()
        if not ret:
            break
        img_copy = img.copy()

        if count % skip == 0:
            if show:
                cv2.imshow('frame', img_copy)

            start_time = time.time()
            face_result    = main_process.face_model(img_copy)   
            face_box    = face_result[0].boxes.xyxy.detach().cpu().numpy()

            pose_result = main_process.pose_model(img_copy)
            persons_box = pose_result[0].boxes.xyxy.detach().cpu().numpy()
            pose_lands  = pose_result[0].keypoints.xy.detach().cpu().numpy()

            img_copy = main_process.draw_person(img_copy, persons_box, face_box)
            img_copy = main_process.draw_face(img_copy, face_box)
            img_copy = main_process.draw_pose(img_copy, pose_lands)

            print('\nTotal secounds : ', time.time() - start_time,'\n')

            cv2.imwrite('./temp.png', img_copy)

            if show:
                cv2.imshow('frame', img_copy)  
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break
            # break
            print('Frame count : ',count)
            out.write(img_copy)


out.release()