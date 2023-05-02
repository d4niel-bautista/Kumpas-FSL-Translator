from base_recog_class import BaseRecognitionClass
from models.recog_models import FacialExpressionRecognition
import mediapipe as mp
import csv
import copy
import itertools
import cv2 as cv

class FaceRecog(BaseRecognitionClass):
    def __init__(self):
        self.recog_model = FacialExpressionRecognition()
        self.labels = self.get_labels()
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
        self.draw_landmarks = True
        self.collect_data = False
    
    def get_labels(self):
        with open('data/face/face_expre_labels.csv',
            encoding='utf-8-sig') as f:
            face_expre_labels = csv.reader(f)
            face_expre_labels = [row[0] for row in face_expre_labels]
            return face_expre_labels
    
    def calc_landmarks(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point
    
    def preprocess_landmarks(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list
    
    def draw_connections(self, image, landmark_point):
        for point in landmark_point:
            cv.circle(image,tuple(point),1,(255, 255, 255),1)
        return image
    
    def draw_text(self, image, brect, facial_text):
        if facial_text != "":
            info_text = facial_text
        (w, h), _ = cv.getTextSize(info_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        if brect[0] + w > brect[2]:
            brect[2] = brect[0] + w
        cv.rectangle(image, (brect[0], brect[1]), (brect[2] + 10, brect[1] - 22),
                    (255, 255, 255), -1)
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA)

        return image
        