from insightface.data import get_image as ins_get_image
from insightface.app import FaceAnalysis
from deepface import DeepFace
from ultralytics import YOLO
import numpy as np
import cv2

def cutting(img,y1,y2,x1,x2):
  H,W,_ = img.shape
  if y1 < 0: y1 = 0
  if x1 < 0: x1 = 0
  if y2 > H: y2 = H-1
  if x2 > W: x2 = W-1

  return img[y1:y2, x1:x2, :], (y1,y2,x1,x2)

class MainProcess():
    def __init__(self, hat_model='./models/hat_detection.pt', face_model='./models/face_detection.pt', pose_model='./models/pose_detection.pt') -> None:
        self.hat_model       = YOLO(hat_model)
        self.face_model      = YOLO(face_model)
        self.pose_model      = YOLO(pose_model)
        self.landmark_model  = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
        self.landmark_model.prepare(ctx_id=0, det_size=(640, 640))

        self.pair_points = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                         [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    def draw_face(self, img, bboxs, color=(0,255,0)):
      for bbox in bboxs:
        try:
          dist = int(bbox[3]-bbox[1])
          selected, cut_box = cutting(img, int(bbox[1]-dist), int(bbox[3]+dist), int(bbox[0]-dist), int(bbox[2]+dist))
          faces = self.landmark_model.get(selected)
          for face in faces:
              lmk = face.landmark_2d_106
              lmk = np.round(lmk).astype(np.int32)
              for i in range(lmk.shape[0]):
                  p = tuple(lmk[i])
                  cv2.circle(selected, p, 1, (255,255,255), 1, cv2.LINE_AA)
          img[cut_box[0]:cut_box[1],cut_box[2]:cut_box[3],:] = selected
        except:
          pass
      
        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 1)
      return img


    def draw_pose(self, img, points_list):
      for points in points_list:
        # draw circle on each point
        for point in points:
          cv2.circle(img, (int(point[0]),int(point[1])), 5, (0,0,0), -1)
    
        from random import randrange
        rand_color = (randrange(255), randrange(255), randrange(255))
    
        for pair in self.pair_points:
          try:
            if points[pair[0]-1][0] != 0 and points[pair[0]-1][1] != 0 and points[pair[1]-1][0] != 0 and points[pair[1]-1][1] != 0:
              cv2.line(img, (int(points[pair[0]-1][0]),int(points[pair[0]-1][1])), (int(points[pair[1]-1][0]),int(points[pair[1]-1][1])), rand_color, 2)
          except:
            pass
        
      return img

    # Draw person with hat detection
    def draw_person(self, img, bboxs, face_boxes,color=(0,255,0)):
      for idx, bbox in enumerate(bboxs):
        # Hat detection
        dist = 30
        try:
          selected, cut_box = cutting(img, int(bbox[1]-dist), int(bbox[3]+dist), int(bbox[0]-dist), int(bbox[2]+dist))
          hat_result = self.hat_model(selected)
          if hat_result[0].boxes.cls.cpu().detach().numpy()[0] == 0:
            hat_text = 'HAT'
          else:
            hat_text = 'Non-HAT'
        except:
          hat_text = 'Non-HAT'

        # Detect age, gender, emotion
        try:
          face_box = face_boxes[idx]
          dist = int(face_box[3]-face_box[1])
          selected, cut_box = cutting(img, int(face_box[1]-dist), int(face_box[3]+dist), int(face_box[0]-dist), int(face_box[2]+dist))

          objs   = DeepFace.analyze(img_path = selected, detector_backend='mtcnn' ,actions = ['age', 'gender', 'emotion'])
          age    = 'Age : '+str(objs[0]['age'])
          gender = 'woman' if objs[0]['gender']['Woman'] > objs[0]['gender']['Man'] else 'Man';gender = 'Gender : ' + gender

          best = 0
          emotion_name = ''
          for emotion in objs[0]['emotion'].keys():
              if objs[0]['emotion'][emotion] > best:
                best = objs[0]['emotion'][emotion]
                emotion_name = emotion
          emotion = 'Emotion : ' + emotion_name

          cv2.putText(img, age, (int(bbox[2])+2,int(bbox[1])+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
          cv2.putText(img, gender, (int(bbox[2])+2,int(bbox[1])+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
          cv2.putText(img, emotion, (int(bbox[2])+2,int(bbox[1])+70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        except:
          pass

        cv2.putText(img, hat_text, (int(bbox[2])+2,int(bbox[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 1)


      return img
