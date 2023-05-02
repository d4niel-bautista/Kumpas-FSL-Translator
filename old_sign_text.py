import mediapipe as mp
from models.recog_models import FacialExpressionRecognition, HandPoseRecognition, BodySequenceRecognition
import cv2 as cv
import csv
import os
import copy
import itertools
import numpy as np
from collections import deque, Counter
import screeninfo

face_expre_recog = FacialExpressionRecognition()
hand_pose_recog = HandPoseRecognition()
body_sequence_recog = BodySequenceRecognition()
pose_detection_threshold = 0.5
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

pose_zeros = [0.0] * 99
hand_zeros = [0.0] * 42

screen = screeninfo.get_monitors()[0]
width, height = screen.width//2, screen.height//2

with open('data/face/face_expre_labels.csv',
    encoding='utf-8-sig') as f:
    face_expre_labels = csv.reader(f)
    face_expre_labels = [
        row[0] for row in face_expre_labels
    ]

with open('data/hand/hand_pose_labels.csv',
            encoding='utf-8-sig') as g:
    hand_pose_labels = csv.reader(g)
    hand_pose_labels = [
        row[0] for row in hand_pose_labels
    ]

body_sequence_labels = os.listdir('data/body_sequence')

def calc_landmark_coordinates(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def calc_pose_res(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        if landmark.visibility > pose_detection_threshold:
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_vis = landmark.visibility
            landmark_point.append([landmark_x, landmark_y, landmark_vis])
        else:
            landmark_point.append([0, 0, 0])
    return landmark_point

def preprocess_face_landmarks(landmark_list):
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

def preprocess_hand_landmarks(landmark_list, handedness):
    temp_landmark_list = copy.deepcopy(landmark_list)

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

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def preprocess_pose_res(pose_res_list):
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
    
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                    (255, 255, 255), 1)

    return image

def draw_face_landmarks(image, landmark_point):
    for point in landmark_point:
        cv.circle(image,tuple(point),1,(255, 255, 255),1)
    return image

def draw_pose_connections(image, landmark_point):
    pose_connections = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                            (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                            (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                            (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                            (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                            (29, 31), (30, 32), (27, 31), (28, 32)]

    for i in pose_connections:
        if landmark_point[i[0]][2] > pose_detection_threshold and landmark_point[i[1]][2] > pose_detection_threshold:
            cv.line(image, tuple(landmark_point[i[0]][0:2]), tuple(landmark_point[i[1]][0:2]),(255, 255, 255), 1)
            cv.circle(image,tuple(landmark_point[i[0]][0:2]),2,(255, 240, 196),2)
            cv.circle(image,tuple(landmark_point[i[1]][0:2]),2,(255, 240, 196),2)

def draw_hand_landmarks(image, landmark_point):
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

def draw_hand_text(image, brect, handedness, hand_sign_text):
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

def draw_face_text(image, brect, facial_text):
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

def draw_output(image, output_list):
    output = " ".join(output_list)
    cv.putText(image, output, (10,470),
            cv.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 9, cv.LINE_AA)
    cv.putText(image, output, (10,470),
        cv.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
    return image


cap_width = 640
cap_height = 480
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

max_frames = 20
pose_sequence = deque(maxlen=max_frames)
block_hand_recog = False
block_pos_recog = 0
use_brect = True

hand_pose_pred = deque(maxlen=30)
predictions_list = deque(maxlen=10)
output_list = deque([''], maxlen=5)

cv.namedWindow('KUMPAS FSL TRANSLATOR', cv.WINDOW_NORMAL)
cv.moveWindow('KUMPAS FSL TRANSLATOR', width - (cap_width//2), height - (cap_height//2))
while True:
    pose_one_frame = deque(maxlen=1)
    lh_one_frame = deque(maxlen=1)
    rh_one_frame = deque(maxlen=1)

    ret, image = cap.read()

    if not ret:
        break
    
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)
    
    # Detection implementation
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    face_mesh_res = face_mesh.process(image)
    hand_pose_res = hands.process(image)
    pose_res = pose.process(image)
    image.flags.writeable = True

    if pose_res.pose_landmarks:
        pose_res_list = calc_pose_res(debug_image, pose_res.pose_landmarks)
        preprocessed_pose_res_list = preprocess_pose_res(
                pose_res_list)
        draw_pose_connections(debug_image, pose_res_list)
        pose_one_frame.append(preprocessed_pose_res_list)
    else:
        if len(pose_zeros) > 99:
            pose_zeros = [0.0] * 99
            pose_one_frame.append(pose_zeros)
        else:
            pose_one_frame.append(pose_zeros)

    if block_pos_recog == 0:
        if len(pose_sequence) == max_frames:
            sign_id, lstm_proba = body_sequence_recog(pose_sequence)
            if lstm_proba >= 80:
                if body_sequence_labels[sign_id] != 'error':
                    block_hand_recog = True
                    print(body_sequence_labels[sign_id], lstm_proba)
                    predictions_list.append(body_sequence_labels[sign_id])
            else:
                block_hand_recog = False
    
    if face_mesh_res.multi_face_landmarks is not None:
        for face_landmarks in face_mesh_res.multi_face_landmarks:
            face_landmark_list = calc_landmark_coordinates(debug_image, face_landmarks)

            brect = calc_bounding_rect(debug_image, face_landmarks)
            preprocessed_face_landmarks = preprocess_face_landmarks(
                face_landmark_list)
            facial_emotion_id = face_expre_recog(preprocessed_face_landmarks)
            debug_image = draw_face_landmarks(debug_image, face_landmark_list)
            if face_expre_labels[facial_emotion_id] != "Neutral":
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_face_text(
                        debug_image,
                        brect,
                        face_expre_labels[facial_emotion_id])
                
    if hand_pose_res.multi_hand_landmarks is not None:
        if len(hand_pose_res.multi_handedness) == 1:
            for hand_landmarks, handedness in zip(hand_pose_res.multi_hand_landmarks,
                                                hand_pose_res.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                hand_landmark_list = calc_landmark_coordinates(debug_image, hand_landmarks)
                preprocessed_hand_landmarks = preprocess_hand_landmarks(
                    hand_landmark_list, handedness.classification[0].label)
                debug_image = draw_hand_landmarks(debug_image, hand_landmark_list)
                if not block_hand_recog:
                    hand_sign_id = hand_pose_recog(preprocessed_hand_landmarks)
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_hand_text(
                        debug_image,
                        brect,
                        handedness,
                        hand_pose_labels[hand_sign_id])
                    hand_pose_pred.append(hand_pose_labels[hand_sign_id])
                if handedness.classification[0].label == "Left":
                    lh_one_frame.append(preprocessed_hand_landmarks)
                    rh_one_frame.append(hand_zeros)
                else:
                    rh_one_frame.append(preprocessed_hand_landmarks)
                    lh_one_frame.append(hand_zeros)
                if hand_pose_labels[hand_sign_id] != 'Error':
                    block_pos_recog = 4
                if hand_pose_labels[hand_sign_id] != 'Error' and hand_pose_labels[hand_sign_id] != 'Finger heart':
                    hand_pose_pred.append(hand_pose_labels[hand_sign_id])
        elif len(hand_pose_res.multi_handedness) == 2:
            for hand_landmarks, handedness in zip(hand_pose_res.multi_hand_landmarks,
                                                hand_pose_res.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                hand_landmark_list = calc_landmark_coordinates(debug_image, hand_landmarks)
                preprocessed_hand_landmarks = preprocess_hand_landmarks(
                    hand_landmark_list, handedness.classification[0].label)
                debug_image = draw_hand_landmarks(debug_image, hand_landmark_list)
                if not block_hand_recog:
                    hand_sign_id = hand_pose_recog(preprocessed_hand_landmarks)
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_hand_text(
                        debug_image,
                        brect,
                        handedness,
                        hand_pose_labels[hand_sign_id])
                    if hand_pose_labels[hand_sign_id] != 'Error' and hand_pose_labels[hand_sign_id] != 'Finger heart':
                        hand_pose_pred.append(hand_pose_labels[hand_sign_id])
                if handedness.classification[0].label == "Left":
                    lh_one_frame.append(preprocessed_hand_landmarks)
                else:
                    rh_one_frame.append(preprocessed_hand_landmarks)
                if hand_pose_labels[hand_sign_id] != 'Error':
                    block_pos_recog = 4
    else:
        rh_one_frame.append(hand_zeros)
        lh_one_frame.append(hand_zeros)
        hand_pose_pred.clear()

    key = cv.waitKey(1)
    if key == 27:  # ESC
        # frame  = 0
        break
    cv.rectangle(debug_image, (0, 450), (640,480), (255,255,255), -1)
    if len(predictions_list) == 10:
        most_common_lstm = Counter(predictions_list).most_common()
        if len(output_list) != 0:
            if most_common_lstm[0][0] == output_list[-1]:
                pass
            else:
                output_list.append(most_common_lstm[0][0])
                predictions_list.clear()
                hand_pose_pred.clear()
    if len(hand_pose_pred) == 30:
        most_common_hand_pose = Counter(hand_pose_pred).most_common()
        if len(output_list) != 0:
            if most_common_hand_pose[0][0] == 'Error' or most_common_hand_pose[0][0] == 'Finger heart':
                hand_pose_pred.clear()
                block_hand_recog = True
            elif most_common_hand_pose[0][0] == output_list[-1]:
                pass
            else:
                output_list.append(most_common_hand_pose[0][0])
                predictions_list.clear()
                hand_pose_pred.clear()
    draw_output(debug_image, output_list)
    # print(output_list)
    cv.imshow('KUMPAS FSL TRANSLATOR', debug_image)
    if len(pose_one_frame) == 0:
        pose_one_frame.append(pose_zeros)
    if len(lh_one_frame) == 0:
        lh_one_frame.append(hand_zeros)
    if len(rh_one_frame) == 0:
        rh_one_frame.append(hand_zeros)
    pose_one_frame[0].extend(lh_one_frame[0])
    pose_one_frame[0].extend(rh_one_frame[0])
    pose_sequence.append(pose_one_frame[0])

    if block_pos_recog > 0:
        block_pos_recog -= 1

cap.release()
cv.destroyAllWindows()

