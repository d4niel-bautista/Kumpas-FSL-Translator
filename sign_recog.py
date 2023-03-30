import mediapipe as mp
from models.recog_models import FacialExpressionRecognition, HandPoseRecognition, BodySequenceRecognition
import cv2 as cv
import csv
import os
import copy
import itertools
import numpy as np

class SignTextTranslator():
    def __init__(self):
        self.face_expre_recog = FacialExpressionRecognition()
        self.hand_pose_recog = HandPoseRecognition()
        self.body_sequence_recog = BodySequenceRecognition()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
        
        self.pose_zeros = [0.0] * 99
        self.hand_zeros = [0.0] * 42
        with open('data/face/face_expre_labels.csv',
            encoding='utf-8-sig') as f:
            self.face_expre_labels = csv.reader(f)
            self.face_expre_labels = [
                row[0] for row in self.face_expre_labels
            ]

        with open('data/hand/hand_pose_labels.csv',
                    encoding='utf-8-sig') as g:
            self.hand_pose_labels = csv.reader(g)
            self.hand_pose_labels = [
                row[0] for row in self.hand_pose_labels
            ]
        
        self.body_sequence_labels = os.listdir('data/body_sequence')

    def calc_face_landmarks(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point
    
    def calc_hand_landmarks(self, image, landmarks, handedness):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []
        if handedness == "Left":
            for _, landmark in enumerate(landmarks.landmark):
                landmark_x = min(int(landmark.x * image_width), image_width - 1)
                landmark_y = min(int(landmark.y * image_height), image_height - 1)

                landmark_point.append([landmark_x, landmark_y])
        # Keypoint
        else:
            for _, landmark in enumerate(landmarks.landmark):
                landmark_x = min(int(landmark.x * image_width), image_width - 1)
                landmark_y = min(int(landmark.y * image_height), image_height - 1)

                landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def calc_pose_res(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            if landmark.visibility > 0.5:
                # print(landmark.visibility)
                landmark_x = min(int(landmark.x * image_width), image_width - 1)
                landmark_y = min(int(landmark.y * image_height), image_height - 1)
                landmark_vis = landmark.visibility
                landmark_point.append([landmark_x, landmark_y, landmark_vis])
            else:
                landmark_point.append([0, 0, 0])
            # landmark_z = landmark.z
        return landmark_point
    
    def preprocess_face_landmarks(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def preprocess_hand_landmarks(self, landmark_list, handedness):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            if handedness == "Left":
                temp_landmark_list[index][0] = -(temp_landmark_list[index][0] - base_x)
                temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
            else:
                temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
                temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list
    
    def preprocess_pose_res(self, pose_res_list):
        temp_pose_res_list = copy.deepcopy(pose_res_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_pose_res_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            
            if landmark_point[2] != 0:
                temp_pose_res_list[index][0] = temp_pose_res_list[index][0] - base_x
                temp_pose_res_list[index][1] = temp_pose_res_list[index][1] - base_y
        # if temp_pose_res_list[32][2] != 0:
        #     exit()
        # Convert to a one-dimensional list
        temp_pose_res_list = list(
            itertools.chain.from_iterable(temp_pose_res_list))
        # print(temp_pose_res_list)
        # Normalization
        max_value = max(list(map(abs, temp_pose_res_list)))
        # print(max_value)
        def normalize_(n):
            return n / max_value

        temp_pose_res_list = list(map(normalize_, temp_pose_res_list))
        # print(temp_pose_res_list)

        return temp_pose_res_list
        
    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]
    
    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                        (255, 255, 255), 1)

        return image
    
    def draw_face_landmarks(self, image, landmark_point):
        for point in landmark_point:
            cv.circle(image,tuple(landmark_point[point[1]][0:2]),2,(255, 255, 255),2)

    def draw_pose_connections(self, image, landmark_point):
        pose_connections = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                                (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                                (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                                (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                                (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                                (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                                (29, 31), (30, 32), (27, 31), (28, 32)]

        for i in pose_connections:
            if landmark_point[i[0]][2] > 0.5 and landmark_point[i[1]][2] > 0.5:
                cv.line(image, tuple(landmark_point[i[0]][0:2]), tuple(landmark_point[i[1]][0:2]),(255, 255, 255), 1)
                cv.circle(image,tuple(landmark_point[i[0]][0:2]),2,(255, 240, 196),2)
                cv.circle(image,tuple(landmark_point[i[1]][0:2]),2,(255, 240, 196),2)

    def draw_hand_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            # cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (255,255,255), 2)
            # cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (255,255,255), 2)

            # Index finger
            # cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (255,255,255), 2)
            # cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (255,255,255), 2)
            # cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (255,255,255), 2)

            # Middle finger
            # cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (255,255,255), 2)
            # cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (255,255,255), 2)
            # cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (255,255,255), 2)

            # Ring finger
            # cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (255,255,255), 2)
            # cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (255,255,255), 2)
            # cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (255,255,255), 2)

            # Little finger
            # cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (255,255,255), 2)
            # cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (255,255,255), 2)
            # cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (255,255,255), 2)

            # Palm
            # cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (255,255,255), 2)
            # cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (255,255,255), 2)
            # cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (255,255,255), 2)
            # cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (255,255,255), 2)
            # cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (255,255,255), 2)
            # cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (255,255,255), 2)
            # cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
            #         (255, 234, 214), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (255,255,255), 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:  # 手首1
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 1:  # 手首2
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 2:  # 親指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 3:  # 親指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 4:  # 親指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255,255,255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 234, 214), 1)
            if index == 5:  # 人差指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 6:  # 人差指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 7:  # 人差指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 8:  # 人差指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255,255,255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 234, 214), 1)
            if index == 9:  # 中指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 10:  # 中指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 11:  # 中指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 12:  # 中指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255,255,255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 234, 214), 1)
            if index == 13:  # 薬指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 14:  # 薬指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 15:  # 薬指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 16:  # 薬指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255,255,255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 234, 214), 1)
            if index == 17:  # 小指：付け根
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 18:  # 小指：第2関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 19:  # 小指：第1関節
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 234, 214), 1)
            if index == 20:  # 小指：指先
                cv.circle(image, (landmark[0], landmark[1]), 8, (255,255,255),
                        -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 234, 214), 1)

        return image
    
    def draw_hand_text(self, image, brect, handedness, hand_sign_text):
        if hand_sign_text == 'Error':
            return image
        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text.upper() + ': ' + hand_sign_text
        (w, h), _ = cv.getTextSize(info_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        if brect[0] + w > brect[2]:
            brect[2] = brect[0] + w
        cv.rectangle(image, (brect[0], brect[1]), (brect[2] + 10, brect[1] - 22),
                    (255, 255, 255), -1)
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA)

        return image

    def draw_face_text(self, image, brect, facial_text):
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

