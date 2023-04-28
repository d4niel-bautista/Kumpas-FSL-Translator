import cv2 as cv
import copy
import numpy as np
from collections import deque, Counter
import screeninfo
from body_sequence_recog import BodySequenceRecog
from face_recog import FaceRecog
from hand_pose_recog import HandPoseRecog
import threading
from gui_sign_to_text import SignToText

class SignText():
    def __init__(self):
        self.hand_pose = HandPoseRecog()
        self.body_seq = BodySequenceRecog()
        self.face_expre = FaceRecog()

        self.screen = screeninfo.get_monitors()[0]
        self.width, self.height = self.screen.width//2, self.screen.height//2

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
    
    def start_gui(self, x, y, object):
        gui = SignToText(x=x, y=y, object=object)
        gui.mainloop()
    
    def main(self):
        cap_width = 640
        cap_height = 480
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        max_frames = 20
        pose_sequence = deque(maxlen=max_frames)
        block_hand_recog = False
        block_pos_recog = 0

        hand_pose_pred = deque(maxlen=10)
        predictions_list = deque(maxlen=5)
        output_list = deque([''], maxlen=5)

        cv.namedWindow('KUMPAS FSL TRANSLATOR', cv.WINDOW_NORMAL)
        cv.moveWindow('KUMPAS FSL TRANSLATOR', self.width - (cap_width//2), self.height - (cap_height//2))
        gui_thread = threading.Thread(target=self.start_gui, args=(self.width + (cap_width//2) + 1, self.height - (cap_height//2), self), daemon=True)
        gui_thread.start()
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
                if len(pose_sequence) == max_frames:
                    sign_id, lstm_proba = self.body_seq.recog_model(pose_sequence)
                    if lstm_proba >= 80:
                        if self.body_seq.labels[sign_id] != 'error':
                            block_hand_recog = True
                            print(self.body_seq.labels[sign_id], lstm_proba)
                            predictions_list.append(self.body_seq.labels[sign_id])
                    else:
                        block_hand_recog = False
            
            if face_mesh_res.multi_face_landmarks is not None:
                for face_landmarks in face_mesh_res.multi_face_landmarks:
                    face_landmark_list = self.face_expre.calc_landmarks(debug_image, face_landmarks)

                    
                    preprocessed_face_landmarks = self.face_expre.preprocess_landmarks(face_landmark_list)
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
                        debug_image = self.hand_pose.draw_connections(debug_image, hand_landmark_list)
                        if not block_hand_recog:
                            hand_sign_id = self.hand_pose.recog_model(preprocessed_hand_landmarks)

                            hand_rect = self.calc_bounding_rect(debug_image, hand_landmarks)
                            debug_image = self.draw_bounding_rect(debug_image, hand_rect)

                            debug_image = self.hand_pose.draw_text(
                                debug_image,
                                hand_rect,
                                handedness,
                                self.hand_pose.labels[hand_sign_id])
                            hand_pose_pred.append(self.hand_pose.labels[hand_sign_id])
                        if handedness.classification[0].label == "Left":
                            lh_one_frame.append(preprocessed_hand_landmarks)
                            rh_one_frame.append(self.hand_pose.zeros)
                        else:
                            rh_one_frame.append(preprocessed_hand_landmarks)
                            lh_one_frame.append(self.hand_pose.zeros)
                        if self.hand_pose.labels[hand_sign_id] != 'Error':
                            block_pos_recog = 4
                        if self.hand_pose.labels[hand_sign_id] != 'Error' and self.hand_pose.labels[hand_sign_id] != 'Finger heart':
                            hand_pose_pred.append(self.hand_pose.labels[hand_sign_id])
                elif len(hand_pose_res.multi_handedness) == 2:
                    for hand_landmarks, handedness in zip(hand_pose_res.multi_hand_landmarks,
                                                        hand_pose_res.multi_handedness):
                        
                        hand_landmark_list = self.hand_pose.calc_landmarks(debug_image, hand_landmarks)
                        preprocessed_hand_landmarks = self.hand_pose.preprocess_landmarks(
                            hand_landmark_list, handedness.classification[0].label)
                        debug_image = self.hand_pose.draw_connections(debug_image, hand_landmark_list)
                        if not block_hand_recog:
                            hand_sign_id = self.hand_pose.recog_model(preprocessed_hand_landmarks)

                            hand_rect = self.calc_bounding_rect(debug_image, hand_landmarks)
                            debug_image = self.draw_bounding_rect(debug_image, hand_rect)

                            debug_image = self.hand_pose.draw_text(
                                debug_image,
                                hand_rect,
                                handedness,
                                self.hand_pose.labels[hand_sign_id])
                            if self.hand_pose.labels[hand_sign_id] != 'Error' and self.hand_pose.labels[hand_sign_id] != 'Finger heart':
                                hand_pose_pred.append(self.hand_pose.labels[hand_sign_id])
                        if handedness.classification[0].label == "Left":
                            lh_one_frame.append(preprocessed_hand_landmarks)
                        else:
                            rh_one_frame.append(preprocessed_hand_landmarks)
                        if self.hand_pose.labels[hand_sign_id] != 'Error':
                            block_pos_recog = 4
            else:
                rh_one_frame.append(self.hand_pose.zeros)
                lh_one_frame.append(self.hand_pose.zeros)
                hand_pose_pred.clear()

            key = cv.waitKey(1)
            if key == 27:  # ESC
                # frame  = 0
                break

            if key == 32:
                output_list.clear()
                predictions_list.append('')
                pose_sequence.clear()

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
            self.draw_output_list(debug_image, output_list)
            # print(output_list)
            cv.imshow('KUMPAS FSL TRANSLATOR', debug_image)
            if len(pose_one_frame) == 0:
                pose_one_frame.append(self.body_seq.zeros)
            if len(lh_one_frame) == 0:
                lh_one_frame.append(self.hand_pose.zeros)
            if len(rh_one_frame) == 0:
                rh_one_frame.append(self.hand_pose.zeros)
            pose_one_frame[0].extend(lh_one_frame[0])
            pose_one_frame[0].extend(rh_one_frame[0])
            pose_sequence.append(pose_one_frame[0])

            if block_pos_recog > 0:
                block_pos_recog -= 1

        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":    
    sign_text = SignText()
    sign_text.main()