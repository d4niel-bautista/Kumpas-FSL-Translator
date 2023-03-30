import os
from PIL import Image, ImageTk
import customtkinter as ctk
from color import Color
from sign_recog import SignTextTranslator
import cv2
from collections import deque
import copy
import threading

class SpeechToSign(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = Color()
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.h = 600
        self.w = 800
        self.window_width = self.w
        self.window_height = self.h
        self.x_coordinate = int((self.screen_width/2) - (self.window_width/2))
        self.y_coordinate = int((self.screen_height/2) - (self.window_height/1.9))
        self.geometry(f"{self.window_width}x{self.window_height}+{self.x_coordinate}+{self.y_coordinate}")
        self.grid_propagate(False)
        
        self.title('SPEECH TO SIGN')

        self.titleFrame = ctk.CTkFrame(master=self, fg_color=self.color.white, width = self.window_width, height= self.window_height * .13, corner_radius=0, border_width=0)
        self.titleFrame.grid(row=0, column=0, sticky='new', padx=0, pady=0)
        self.titleFrame.grid_propagate(False)
        self.titleFrame.grid_columnconfigure(2, weight=1)
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.returnImg = ctk.CTkImage(Image.open(self.current_path + "/img/return.png"), size=(50,50))
        self.returnBtn = ctk.CTkButton(master=self.titleFrame, image=self.returnImg,  text="", fg_color=self.color.transparent, width=self.window_width * .015, height=self.window_height * .0588, border_width=0)
        self.returnBtn.grid(row=0, column=0,sticky='wns', padx=self.window_width * .0083, pady=(((self.window_height * .13)/2)/2)/2)
        self.speechToSignLogo = ctk.CTkImage(Image.open(self.current_path + "/img/speech-sign.png"), size=(self.window_width * .0533,self.window_height * .0914))
        self.speechToSignLogoLabel = ctk.CTkLabel(master=self.titleFrame, image=self.speechToSignLogo, text=" SPEECH TO SIGN", compound='left',text_color=self.color.black,font=ctk.CTkFont(size=25))
        self.speechToSignLogoLabel.grid(pady=self.window_height * .0114, padx=self.window_width * .012, row=0,column=2,sticky='nswe')

        self.camera_frame = ctk.CTkFrame(master=self, width=700, height=505)
        self.camera_frame.grid(row=1, column=0)
        self.camera_frame.grid_propagate(False)
        self.camera_frame.grid_columnconfigure(0, weight=1)
        self.sign_text = SignTextTranslator()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        self.cam_widget = ctk.CTkLabel(self.camera_frame, text="CAMERA")
        self.cam_widget.grid(column=0, pady=10)
        self.pose_sequence = deque(maxlen=20)

        self.max_frames = 20
        self.use_brect = True
        self.sign_detected = False
        self.grab_set()
        self.start_camera = threading.Thread(target=self.loop, args=(), daemon=True)
        self.start_camera.start()
    
    def loop(self):
        pose_one_frame = deque(maxlen=1)
        lh_one_frame = deque(maxlen=1)
        rh_one_frame = deque(maxlen=1)
        print('test')
        ret, image = self.cap.read()

        if not ret:
            self.cam_widget.after(1, self.loop)
        
        image = cv2.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        
        # Detection implementation
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = self.sign_text.face_mesh.process(image)
        handData = self.sign_text.hands.process(image)
        pose_res = self.sign_text.pose.process(image)
        image.flags.writeable = True

        if pose_res.pose_landmarks:
            pose_res_list = self.sign_text.calc_pose_res(debug_image, pose_res.pose_landmarks)
            preprocessed_pose_res_list = self.sign_text.preprocess_pose_res(
                    pose_res_list)
            self.sign_text.draw_pose_connections(debug_image, pose_res_list)
            pose_one_frame.append(preprocessed_pose_res_list)
        else:
            if len(self.sign_text.pose_zeros) > 99:
                self.sign_text.pose_zeros = [0.0] * 99
                pose_one_frame.append(self.sign_text.pose_zeros)
            else:
                pose_one_frame.append(self.sign_text.pose_zeros)

        if len(self.pose_sequence) == self.max_frames:
            sign_id, lstm_proba = self.sign_text.body_sequence_recog(self.pose_sequence)
            if lstm_proba >= 70:
                if self.sign_text.body_sequence_labels[sign_id] != 'error':
                    sign_detected = True
                    print(self.sign_text.body_sequence_labels[sign_id], lstm_proba)
            else:
                sign_detected = False
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                landmark_list = self.sign_text.calc_face_landmarks(debug_image, face_landmarks)

                brect = self.sign_text.calc_bounding_rect(debug_image, face_landmarks)
                pre_processed_landmark_list = self.sign_text.preprocess_face_landmarks(
                    landmark_list)
                facial_emotion_id = self.sign_text.face_expre_recog(pre_processed_landmark_list)
                if self.sign_text.face_expre_labels[facial_emotion_id] != "Neutral":
                    debug_image = self.sign_text.draw_bounding_rect(self.use_brect, debug_image, brect)
                    debug_image = self.sign_text.draw_face_text(
                            debug_image,
                            brect,
                            self.sign_text.face_expre_labels[facial_emotion_id])
                    
        if handData.multi_hand_landmarks is not None:
            if len(handData.multi_handedness) == 1:
                for hand_landmarks, handedness in zip(handData.multi_hand_landmarks,
                                                    handData.multi_handedness):
                    brect = self.sign_text.calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = self.sign_text.calc_face_landmarks(debug_image, hand_landmarks)
                    pre_processed_landmark_list = self.sign_text.preprocess_hand_landmarks(
                        landmark_list, handedness.classification[0].label)
                    debug_image = self.sign_text.draw_hand_landmarks(debug_image, landmark_list)
                    if not self.sign_detected:
                        hand_sign_id = self.sign_text.hand_pose_recog(pre_processed_landmark_list)
                        debug_image = self.sign_text.draw_bounding_rect(self.use_brect, debug_image, brect)
                        debug_image = self.sign_text.draw_hand_text(
                            debug_image,
                            brect,
                            handedness,
                            self.sign_text.hand_pose_labels[hand_sign_id])
                    if handedness.classification[0].label == "Left":
                        lh_one_frame.append(pre_processed_landmark_list)
                        rh_one_frame.append(self.sign_text.hand_zeros)
                    else:
                        rh_one_frame.append(pre_processed_landmark_list)
                        lh_one_frame.append(self.sign_text.hand_zeros)
            elif len(handData.multi_handedness) == 2:
                for hand_landmarks, handedness in zip(handData.multi_hand_landmarks,
                                                    handData.multi_handedness):
                    brect = self.sign_text.calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = self.sign_text.calc_face_landmarks(debug_image, hand_landmarks)
                    pre_processed_landmark_list = self.sign_text.preprocess_hand_landmarks(
                        landmark_list, handedness.classification[0].label)
                    debug_image = self.sign_text.draw_hand_landmarks(debug_image, landmark_list)
                    if not sign_detected:
                        hand_sign_id = self.sign_text.hand_pose_recog(pre_processed_landmark_list)
                        debug_image = self.sign_text.draw_bounding_rect(self.use_brect, debug_image, brect)
                        debug_image = self.sign_text.draw_hand_text(
                            debug_image,
                            brect,
                            handedness,
                            self.sign_text.hand_pose_labels[hand_sign_id])
                    if handedness.classification[0].label == "Left":
                        lh_one_frame.append(pre_processed_landmark_list)
                    else:
                        rh_one_frame.append(pre_processed_landmark_list)
        else:
            rh_one_frame.append(self.sign_text.hand_zeros)
            lh_one_frame.append(self.sign_text.hand_zeros)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            frame  = 0
            self.cam_widget.after(1, self.loop)

        opencv_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGBA)
        captured_image = Image.fromarray(opencv_image)
        photo_image = ImageTk.PhotoImage(image=captured_image)
        self.cam_widget.photo_image = photo_image
        self.cam_widget.configure(image=photo_image)
        
        # cv2.putText(debug_image, str(frame), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        # cv2.imshow('Facial Emotion Recognition', debug_image)
        if len(pose_one_frame) == 0:
            pose_one_frame.append(self.sign_text.pose_zeros)
        if len(lh_one_frame) == 0:
            lh_one_frame.append(self.sign_text.hand_zeros)
        if len(rh_one_frame) == 0:
            rh_one_frame.append(self.sign_text.hand_zeros)
        pose_one_frame[0].extend(lh_one_frame[0])
        pose_one_frame[0].extend(rh_one_frame[0])
        self.pose_sequence.append(pose_one_frame[0])
        self.cam_widget.after(1, self.loop)
    
