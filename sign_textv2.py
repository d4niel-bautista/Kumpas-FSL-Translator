import cv2 as cv
import copy
import numpy as np
from collections import deque, Counter
import screeninfo
from body_sequence_recog import BodySequenceRecog
from face_recog import FaceRecog
from hand_pose_recog import HandPoseRecog
import threading
from gui_sign_to_text import SignToTextGUI
import shutil
import csv
import os

class SignText():
    def __init__(self):
        self.hand_pose = HandPoseRecog()
        self.body_seq = BodySequenceRecog()
        self.face_expre = FaceRecog()

        self.screen = screeninfo.get_monitors()[0]
        self.width, self.height = self.screen.width//2, self.screen.height//2
        self.cap_width = 640
        self.cap_height = 480
        self.to_add_data_idx = None

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

    def draw_bounding_rect(self, image, brect):
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                    (255, 255, 255), 1)

        return image
    
    def draw_output_list(self, image, output_list):
        output = " ".join(output_list)
        cv.putText(image, output, (10,470),
                cv.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 9, cv.LINE_AA)
        cv.putText(image, output, (10,470),
            cv.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
        return image
    
    def start_thread(self):
        gui_thread = threading.Thread(target=self.start_gui, args=(self.width + 8 + (self.cap_width//2), self.height - (self.cap_height//2), self), daemon=True)
        gui_thread.start()

    def start_gui(self, x, y, main_app):
        self.gui = SignToTextGUI(x=x, y=y, main_app=main_app)
        self.gui.mainloop()

    def create_temp(self):
        if os.path.exists('temp'):
            shutil.rmtree('temp')
        for i in os.listdir('data'):
            if i == 'body_sequence':
                os.makedirs(os.path.join('temp', i))
                continue
            shutil.copytree(os.path.join('data', i), os.path.join('temp', i))
    
    def clear_output(self):
        self.output_list.clear()
        self.output_list.append('')
        self.pose_sequence.clear()
        self.hand_pose_pred.clear()
        self.body_seq_pred.clear()
    
    def main(self):
        self.create_temp()
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)

        max_frames = 20
        self.pose_sequence = deque(maxlen=max_frames)
        block_hand_recog = 0
        block_pos_recog = 0

        self.hand_pose_pred = deque(maxlen=25)
        self.body_seq_pred = deque(maxlen=6)
        self.output_list = deque([''], maxlen=5)
        window_name = 'KUMPAS FSL TRANSLATOR'

        cv.namedWindow(window_name)
        cv.moveWindow(window_name, self.width - (self.cap_width//2), self.height - (self.cap_height//2))
        cv.setWindowProperty(window_name, cv.WND_PROP_TOPMOST, 1)
        self.frames = 0
        self.start_thread()
        while True:
            pose_one_frame = deque(maxlen=1)
            lh_one_frame = deque(maxlen=1)
            rh_one_frame = deque(maxlen=1)

            while self.body_seq.collect_data:
                path = os.path.join('temp', 'body_sequence', self.gui.body_seq_labels[self.to_add_data_idx])
                if not os.path.exists(path):
                    os.makedirs(path)

                add_to_existing = False
                
                if self.gui.body_seq_labels[self.to_add_data_idx] in self.gui.to_delete_body_seq and self.gui.body_seq_labels[self.to_add_data_idx] in os.listdir(os.path.join('data', 'body_sequence')):
                    reps = len(os.listdir(path))
                    add_to_existing = False
                elif self.gui.body_seq_labels[self.to_add_data_idx] in os.listdir(os.path.join('data', 'body_sequence')) and self.gui.body_seq_labels[self.to_add_data_idx] not in self.gui.to_delete_body_seq and os.path.exists(path):
                    reps = len(os.listdir(path)) + len(os.listdir(os.path.join('data', 'body_sequence', self.gui.body_seq_labels[self.to_add_data_idx])))
                    add_to_existing = True
                elif self.gui.body_seq_labels[self.to_add_data_idx] not in self.gui.to_delete_body_seq and self.gui.body_seq_labels[self.to_add_data_idx] in os.listdir(os.path.join('data', 'body_sequence')):
                    reps = len(os.listdir(os.path.join('data', 'body_sequence', self.gui.body_seq_labels[self.to_add_data_idx])))
                    add_to_existing = True
                else:
                    reps = len(os.listdir(path))
                    add_to_existing = False
                    
                while True:
                    if add_to_existing:
                        if os.path.exists(os.path.join('data', 'body_sequence', self.gui.body_seq_labels[self.to_add_data_idx], str(reps))):
                            reps += 1
                            continue
                        else:
                            os.makedirs(os.path.join(path, str(reps)))
                            break
                    else:
                        if os.path.exists(os.path.join(path, str(reps))):
                            reps += 1
                            continue
                        else:
                            os.makedirs(os.path.join(path, str(reps)))
                            break
                
                while self.frames < max_frames:
                    ret, image = cap.read()

                    if not ret:
                        break
                    
                    image = cv.flip(image, 1)  # Mirror display
                    debug_image = copy.deepcopy(image)
                    
                    # Detection implementation
                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                    image.flags.writeable = False
                    face_mesh_res = self.face_expre.mp_face_mesh.process(image)
                    hand_pose_res = self.hand_pose.mp_hands.process(image)
                    pose_res = self.body_seq.mp_pose.process(image)
                    image.flags.writeable = True

                    if pose_res.pose_landmarks:
                        pose_res_list = self.body_seq.calc_landmarks(debug_image, pose_res.pose_landmarks)
                        preprocessed_pose_res_list = self.body_seq.preprocess_landmarks(
                                pose_res_list)
                        self.body_seq.draw_connections(debug_image, pose_res_list)
                        pose_one_frame.append(preprocessed_pose_res_list)
                    else:
                        if len(self.body_seq.zeros) > 99:
                            self.body_seq.zeros = [0.0] * 99
                            pose_one_frame.append(self.body_seq.zeros)
                        else:
                            pose_one_frame.append(self.body_seq.zeros)

                    if block_pos_recog == 0:
                        if len(self.pose_sequence) == max_frames:
                            sign_id, lstm_proba = self.body_seq.recog_model(self.pose_sequence)
                            if lstm_proba >= 80:
                                if self.body_seq.labels[sign_id] != 'error':
                                    block_hand_recog = 8
                                    print(self.body_seq.labels[sign_id], lstm_proba)
                                    self.body_seq_pred.append(self.body_seq.labels[sign_id])
                                    self.hand_pose_pred.clear()
                            else:
                                print(self.body_seq.labels[sign_id], lstm_proba)
                                # block_hand_recog = 0
                    
                    if face_mesh_res.multi_face_landmarks is not None:
                        for face_landmarks in face_mesh_res.multi_face_landmarks:
                            face_landmark_list = self.face_expre.calc_landmarks(debug_image, face_landmarks)
                            
                            preprocessed_face_landmarks = self.face_expre.preprocess_landmarks(face_landmark_list)

                            # if self.face_expre.collect_data:
                            #     with open("temp/face/face_expre_data.csv", 'a', newline="") as f:
                            #         writer = csv.writer(f)
                            #         writer.writerow([self.to_add_data_idx, *preprocessed_face_landmarks])
                            
                            facial_emotion_id = self.face_expre.recog_model(preprocessed_face_landmarks)
                            debug_image = self.face_expre.draw_connections(debug_image, face_landmark_list)
                            if self.face_expre.labels[facial_emotion_id] != "Neutral":
                                face_rect = self.calc_bounding_rect(debug_image, face_landmarks)
                                debug_image = self.draw_bounding_rect(debug_image, face_rect)
                                debug_image = self.face_expre.draw_text(
                                        debug_image,
                                        face_rect,
                                        self.face_expre.labels[facial_emotion_id])
                                
                    if hand_pose_res.multi_hand_landmarks is not None:
                        if len(hand_pose_res.multi_handedness) == 1:
                            for hand_landmarks, handedness in zip(hand_pose_res.multi_hand_landmarks,
                                                                hand_pose_res.multi_handedness):
                                
                                hand_landmark_list = self.hand_pose.calc_landmarks(debug_image, hand_landmarks)
                                preprocessed_hand_landmarks = self.hand_pose.preprocess_landmarks(
                                    hand_landmark_list, handedness.classification[0].label)
                                
                                # if self.hand_pose.collect_data:
                                #     with open("temp/hand/hand_pose_data.csv", 'a', newline="") as f:
                                #         writer = csv.writer(f)
                                #         writer.writerow([self.to_add_data_idx, *preprocessed_hand_landmarks])
                                
                                debug_image = self.hand_pose.draw_connections(debug_image, hand_landmark_list)
                                if block_hand_recog == 0:
                                    hand_sign_id = self.hand_pose.recog_model(preprocessed_hand_landmarks)

                                    hand_rect = self.calc_bounding_rect(debug_image, hand_landmarks)
                                    debug_image = self.draw_bounding_rect(debug_image, hand_rect)

                                    debug_image = self.hand_pose.draw_text(
                                        debug_image,
                                        hand_rect,
                                        handedness,
                                        self.hand_pose.labels[hand_sign_id])
                                    self.hand_pose_pred.append(self.hand_pose.labels[hand_sign_id])
                                if handedness.classification[0].label == "Left":
                                    lh_one_frame.append(preprocessed_hand_landmarks)
                                    rh_one_frame.append(self.hand_pose.zeros)
                                else:
                                    rh_one_frame.append(preprocessed_hand_landmarks)
                                    lh_one_frame.append(self.hand_pose.zeros)
                                if self.hand_pose.labels[hand_sign_id] != 'Error' and self.hand_pose.labels[hand_sign_id] != 'Finger heart':
                                    block_pos_recog = 6
                                    self.hand_pose_pred.append(self.hand_pose.labels[hand_sign_id])
                                else:
                                    if block_hand_recog == 0:
                                        block_hand_recog = 8
                                    self.hand_pose_pred.clear()
                        elif len(hand_pose_res.multi_handedness) == 2:
                            for hand_landmarks, handedness in zip(hand_pose_res.multi_hand_landmarks,
                                                                hand_pose_res.multi_handedness):
                                
                                hand_landmark_list = self.hand_pose.calc_landmarks(debug_image, hand_landmarks)
                                preprocessed_hand_landmarks = self.hand_pose.preprocess_landmarks(
                                    hand_landmark_list, handedness.classification[0].label)
                                
                                # if self.hand_pose.collect_data:
                                #     with open("temp/hand/hand_pose_data.csv", 'a', newline="") as f:
                                #         writer = csv.writer(f)
                                #         writer.writerow([self.to_add_data_idx, *preprocessed_hand_landmarks])

                                debug_image = self.hand_pose.draw_connections(debug_image, hand_landmark_list)
                                if block_hand_recog == 0:
                                    hand_sign_id = self.hand_pose.recog_model(preprocessed_hand_landmarks)

                                    hand_rect = self.calc_bounding_rect(debug_image, hand_landmarks)
                                    debug_image = self.draw_bounding_rect(debug_image, hand_rect)

                                    debug_image = self.hand_pose.draw_text(
                                        debug_image,
                                        hand_rect,
                                        handedness,
                                        self.hand_pose.labels[hand_sign_id])
                                if handedness.classification[0].label == "Left":
                                    lh_one_frame.append(preprocessed_hand_landmarks)
                                else:
                                    rh_one_frame.append(preprocessed_hand_landmarks)
                                if self.hand_pose.labels[hand_sign_id] != 'Error' and self.hand_pose.labels[hand_sign_id] != 'Finger heart':
                                    block_pos_recog = 6
                                    self.hand_pose_pred.append(self.hand_pose.labels[hand_sign_id])
                                else:
                                    if block_hand_recog == 0:
                                        block_hand_recog = 8
                                    self.hand_pose_pred.clear()
                    else:
                        rh_one_frame.append(self.hand_pose.zeros)
                        lh_one_frame.append(self.hand_pose.zeros)
                        self.hand_pose_pred.clear()

                    key = cv.waitKey(1)
                    if key == 27:  # ESC
                        # frame  = 0
                        break

                    if key == 32:
                        self.clear_output()
                    
                    if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
                        print('exited')
                        break

                    cv.rectangle(debug_image, (0, 450), (640,480), (255,255,255), -1)
                    if len(self.body_seq_pred) == self.body_seq_pred.maxlen:
                        most_common_lstm = Counter(self.body_seq_pred).most_common()
                        if len(self.output_list) != 0:
                            if most_common_lstm[0][0] == self.output_list[-1]:
                                pass
                            else:
                                self.output_list.append(most_common_lstm[0][0])
                                self.body_seq_pred.clear()
                                self.pose_sequence.clear()
                                self.hand_pose_pred.clear()

                    if len(self.hand_pose_pred) == self.hand_pose_pred.maxlen:
                        most_common_hand_pose = Counter(self.hand_pose_pred).most_common()
                        if len(self.output_list) != 0:
                            if most_common_hand_pose[0][0] == 'Error' or most_common_hand_pose[0][0] == 'Finger heart':
                                self.hand_pose_pred.clear()
                                if block_hand_recog == 0:
                                    block_hand_recog = 8
                                pass
                            elif most_common_hand_pose[0][0] == self.output_list[-1]:
                                pass
                            else:
                                self.output_list.append(most_common_hand_pose[0][0])
                                self.body_seq_pred.clear()
                                block_pos_recog = 6
                                self.hand_pose_pred.clear()

                    debug_image = self.draw_output_list(debug_image, self.output_list)
                    cv.rectangle(debug_image, (0, 0), (160,60), (255,255,255), -1)
                    cv.putText(debug_image, 'frame ' + str(self.frames), (2,25),
                        cv.FONT_HERSHEY_DUPLEX, 0.9, (1, 1, 1), 2, cv.LINE_AA)
                    cv.putText(debug_image, self.gui.body_seq_labels[self.to_add_data_idx], (2,52),
                        cv.FONT_HERSHEY_DUPLEX, 0.9, (1, 1, 1), 2, cv.LINE_AA)
                    cv.imshow(window_name, debug_image)
                    
                    # print(block_pos_recog, block_hand_recog, self.hand_pose_pred)

                    if len(pose_one_frame) == 0:
                        pose_one_frame.append(self.body_seq.zeros)
                    if len(lh_one_frame) == 0:
                        lh_one_frame.append(self.hand_pose.zeros)
                    if len(rh_one_frame) == 0:
                        rh_one_frame.append(self.hand_pose.zeros)
                    pose_one_frame[0].extend(lh_one_frame[0])
                    pose_one_frame[0].extend(rh_one_frame[0])
                    self.pose_sequence.append(pose_one_frame[0])
                    if block_pos_recog > 0:
                        block_pos_recog -= 1
                    if block_hand_recog > 0:
                        block_hand_recog -= 1
                    
                    npy = np.array(pose_one_frame[0])
                    np.save(os.path.join(path, str(reps), str(self.frames)), npy)

                    if self.frames < max_frames:
                        self.frames += 1
                    if self.frames == max_frames:
                        self.frames = 0
                        cv.waitKey(500)
                        break
            else:

                ret, image = cap.read()

                if not ret:
                    break
                
                image = cv.flip(image, 1)  # Mirror display
                debug_image = copy.deepcopy(image)
                
                # Detection implementation
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                image.flags.writeable = False
                face_mesh_res = self.face_expre.mp_face_mesh.process(image)
                hand_pose_res = self.hand_pose.mp_hands.process(image)
                pose_res = self.body_seq.mp_pose.process(image)
                image.flags.writeable = True

                if pose_res.pose_landmarks:
                    pose_res_list = self.body_seq.calc_landmarks(debug_image, pose_res.pose_landmarks)
                    preprocessed_pose_res_list = self.body_seq.preprocess_landmarks(
                            pose_res_list)
                    self.body_seq.draw_connections(debug_image, pose_res_list)
                    pose_one_frame.append(preprocessed_pose_res_list)
                else:
                    if len(self.body_seq.zeros) > 99:
                        self.body_seq.zeros = [0.0] * 99
                        pose_one_frame.append(self.body_seq.zeros)
                    else:
                        pose_one_frame.append(self.body_seq.zeros)

                if block_pos_recog == 0:
                    if len(self.pose_sequence) == max_frames:
                        sign_id, lstm_proba = self.body_seq.recog_model(self.pose_sequence)
                        if lstm_proba >= 80:
                            if self.body_seq.labels[sign_id] != 'error':
                                block_hand_recog = 8
                                print(self.body_seq.labels[sign_id], lstm_proba)
                                self.body_seq_pred.append(self.body_seq.labels[sign_id])
                                self.hand_pose_pred.clear()
                        else:
                            print(self.body_seq.labels[sign_id], lstm_proba)
                            # block_hand_recog = 0
                
                if face_mesh_res.multi_face_landmarks is not None:
                    for face_landmarks in face_mesh_res.multi_face_landmarks:
                        face_landmark_list = self.face_expre.calc_landmarks(debug_image, face_landmarks)
                        
                        preprocessed_face_landmarks = self.face_expre.preprocess_landmarks(face_landmark_list)

                        if self.face_expre.collect_data:
                            with open("temp/face/face_expre_data.csv", 'a', newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([self.to_add_data_idx, *preprocessed_face_landmarks])
                        
                        facial_emotion_id = self.face_expre.recog_model(preprocessed_face_landmarks)
                        debug_image = self.face_expre.draw_connections(debug_image, face_landmark_list)
                        if self.face_expre.labels[facial_emotion_id] != "Neutral":
                            face_rect = self.calc_bounding_rect(debug_image, face_landmarks)
                            debug_image = self.draw_bounding_rect(debug_image, face_rect)
                            debug_image = self.face_expre.draw_text(
                                    debug_image,
                                    face_rect,
                                    self.face_expre.labels[facial_emotion_id])
                            
                if hand_pose_res.multi_hand_landmarks is not None:
                    if len(hand_pose_res.multi_handedness) == 1:
                        for hand_landmarks, handedness in zip(hand_pose_res.multi_hand_landmarks,
                                                            hand_pose_res.multi_handedness):
                            
                            hand_landmark_list = self.hand_pose.calc_landmarks(debug_image, hand_landmarks)
                            preprocessed_hand_landmarks = self.hand_pose.preprocess_landmarks(
                                hand_landmark_list, handedness.classification[0].label)
                            
                            if self.hand_pose.collect_data:
                                with open("temp/hand/hand_pose_data.csv", 'a', newline="") as f:
                                    writer = csv.writer(f)
                                    writer.writerow([self.to_add_data_idx, *preprocessed_hand_landmarks])
                            
                            debug_image = self.hand_pose.draw_connections(debug_image, hand_landmark_list)
                            if block_hand_recog == 0:
                                hand_sign_id = self.hand_pose.recog_model(preprocessed_hand_landmarks)

                                hand_rect = self.calc_bounding_rect(debug_image, hand_landmarks)
                                debug_image = self.draw_bounding_rect(debug_image, hand_rect)

                                debug_image = self.hand_pose.draw_text(
                                    debug_image,
                                    hand_rect,
                                    handedness,
                                    self.hand_pose.labels[hand_sign_id])
                                self.hand_pose_pred.append(self.hand_pose.labels[hand_sign_id])
                            if handedness.classification[0].label == "Left":
                                lh_one_frame.append(preprocessed_hand_landmarks)
                                rh_one_frame.append(self.hand_pose.zeros)
                            else:
                                rh_one_frame.append(preprocessed_hand_landmarks)
                                lh_one_frame.append(self.hand_pose.zeros)
                            if self.hand_pose.labels[hand_sign_id] != 'Error' and self.hand_pose.labels[hand_sign_id] != 'Finger heart':
                                block_pos_recog = 6
                                self.hand_pose_pred.append(self.hand_pose.labels[hand_sign_id])
                            else:
                                if block_hand_recog == 0:
                                    block_hand_recog = 8
                                self.hand_pose_pred.clear()
                    elif len(hand_pose_res.multi_handedness) == 2:
                        for hand_landmarks, handedness in zip(hand_pose_res.multi_hand_landmarks,
                                                            hand_pose_res.multi_handedness):
                            
                            hand_landmark_list = self.hand_pose.calc_landmarks(debug_image, hand_landmarks)
                            preprocessed_hand_landmarks = self.hand_pose.preprocess_landmarks(
                                hand_landmark_list, handedness.classification[0].label)
                            
                            if self.hand_pose.collect_data:
                                with open("temp/hand/hand_pose_data.csv", 'a', newline="") as f:
                                    writer = csv.writer(f)
                                    writer.writerow([self.to_add_data_idx, *preprocessed_hand_landmarks])

                            debug_image = self.hand_pose.draw_connections(debug_image, hand_landmark_list)
                            if block_hand_recog == 0:
                                hand_sign_id = self.hand_pose.recog_model(preprocessed_hand_landmarks)

                                hand_rect = self.calc_bounding_rect(debug_image, hand_landmarks)
                                debug_image = self.draw_bounding_rect(debug_image, hand_rect)

                                debug_image = self.hand_pose.draw_text(
                                    debug_image,
                                    hand_rect,
                                    handedness,
                                    self.hand_pose.labels[hand_sign_id])
                            if handedness.classification[0].label == "Left":
                                lh_one_frame.append(preprocessed_hand_landmarks)
                            else:
                                rh_one_frame.append(preprocessed_hand_landmarks)
                            if self.hand_pose.labels[hand_sign_id] != 'Error' and self.hand_pose.labels[hand_sign_id] != 'Finger heart':
                                block_pos_recog = 6
                                self.hand_pose_pred.append(self.hand_pose.labels[hand_sign_id])
                            else:
                                if block_hand_recog == 0:
                                    block_hand_recog = 8
                                self.hand_pose_pred.clear()
                else:
                    rh_one_frame.append(self.hand_pose.zeros)
                    lh_one_frame.append(self.hand_pose.zeros)
                    self.hand_pose_pred.clear()

                key = cv.waitKey(1)
                if key == 27:  # ESC
                    # frame  = 0
                    break

                if key == 32:
                    self.clear_output()
                
                if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
                    print('exited')
                    break

                cv.rectangle(debug_image, (0, 450), (640,480), (255,255,255), -1)
                if len(self.body_seq_pred) == self.body_seq_pred.maxlen:
                    most_common_lstm = Counter(self.body_seq_pred).most_common()
                    if len(self.output_list) != 0:
                        if most_common_lstm[0][0] == self.output_list[-1]:
                            pass
                        else:
                            self.output_list.append(most_common_lstm[0][0])
                            self.body_seq_pred.clear()
                            self.pose_sequence.clear()
                            self.hand_pose_pred.clear()

                if len(self.hand_pose_pred) == self.hand_pose_pred.maxlen:
                    most_common_hand_pose = Counter(self.hand_pose_pred).most_common()
                    if len(self.output_list) != 0:
                        if most_common_hand_pose[0][0] == 'Error' or most_common_hand_pose[0][0] == 'Finger heart':
                            self.hand_pose_pred.clear()
                            if block_hand_recog == 0:
                                block_hand_recog = 8
                            pass
                        elif most_common_hand_pose[0][0] == self.output_list[-1]:
                            pass
                        else:
                            self.output_list.append(most_common_hand_pose[0][0])
                            self.body_seq_pred.clear()
                            block_pos_recog = 6
                            self.hand_pose_pred.clear()

                debug_image = self.draw_output_list(debug_image, self.output_list)
                cv.imshow(window_name, debug_image)
                
                # print(block_pos_recog, block_hand_recog, self.hand_pose_pred)

                if len(pose_one_frame) == 0:
                    pose_one_frame.append(self.body_seq.zeros)
                if len(lh_one_frame) == 0:
                    lh_one_frame.append(self.hand_pose.zeros)
                if len(rh_one_frame) == 0:
                    rh_one_frame.append(self.hand_pose.zeros)
                pose_one_frame[0].extend(lh_one_frame[0])
                pose_one_frame[0].extend(rh_one_frame[0])
                self.pose_sequence.append(pose_one_frame[0])
                if block_pos_recog > 0:
                    block_pos_recog -= 1
                if block_hand_recog > 0:
                    block_hand_recog -= 1
            
            
            
            

        cv.destroyAllWindows()
        cap.release()
        return

if __name__ == "__main__":    
    sign_text = SignText()
    sign_text.main()