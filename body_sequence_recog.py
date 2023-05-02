from base_recog_class import BaseRecognitionClass
from models.recog_models import BodySequenceRecognition
import mediapipe as mp
import copy
import itertools
import cv2 as cv
import os

class BodySequenceRecog(BaseRecognitionClass):
    def __init__(self):
        self.recog_model = BodySequenceRecognition()
        self.labels = self.get_labels()
        self.mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
        self.pose_detection_threshold = 0.5
        self.zeros = [0.0] * 99
        self.draw_landmarks = True
        self.collect_data = False
    
    def get_labels(self):
        return os.listdir('data/body_sequence')
    
    def calc_landmarks(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            if landmark.visibility > self.pose_detection_threshold:
                landmark_x = min(int(landmark.x * image_width), image_width - 1)
                landmark_y = min(int(landmark.y * image_height), image_height - 1)
                landmark_vis = landmark.visibility
                landmark_point.append([landmark_x, landmark_y, landmark_vis])
            else:
                landmark_point.append([0, 0, 0])
        return landmark_point
    
    def preprocess_landmarks(self, pose_res_list):
        temp_pose_res_list = copy.deepcopy(pose_res_list)

        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_pose_res_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            
            if landmark_point[2] != 0:
                temp_pose_res_list[index][0] = temp_pose_res_list[index][0] - base_x
                temp_pose_res_list[index][1] = temp_pose_res_list[index][1] - base_y

        temp_pose_res_list = list(
            itertools.chain.from_iterable(temp_pose_res_list))

        max_value = max(list(map(abs, temp_pose_res_list)))

        def normalize_(n):
            return n / max_value

        temp_pose_res_list = list(map(normalize_, temp_pose_res_list))

        return temp_pose_res_list
    
    def draw_connections(self, image, landmark_point):
        pose_connections = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                            (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                            (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                            (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                            (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                            (29, 31), (30, 32), (27, 31), (28, 32)]

        for i in pose_connections:
            if landmark_point[i[0]][2] > self.pose_detection_threshold and landmark_point[i[1]][2] > self.pose_detection_threshold:
                cv.line(image, tuple(landmark_point[i[0]][0:2]), tuple(landmark_point[i[1]][0:2]),(255, 255, 255), 1)
                cv.circle(image,tuple(landmark_point[i[0]][0:2]),2,(255, 240, 196),2)
                cv.circle(image,tuple(landmark_point[i[1]][0:2]),2,(255, 240, 196),2)