import argparse

parser_ = argparse.ArgumentParser()

parser_.add_argument('--hat', type=str, default='./models/hat_detection.pt',
                    help='Location of hat yolov8 model')

parser_.add_argument('--face', type=str, default='./models/face_detection.pt',
                    help='Location of face yolov8 model')

parser_.add_argument('--pose', type=str, default='./models/pose_detection.pt',
                    help='Location of pose yolov8 model')

parser_.add_argument('--webcam',  action='store_true')

parser_.add_argument('--show',  action='store_true')

parser_.add_argument('--video', type=str, default='',
                    help='Path to your video')

parser_.add_argument('--skip', type=int, default=1)

