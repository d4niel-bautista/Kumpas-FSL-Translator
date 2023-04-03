import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import os

class FacialExpressionRecognition(object):
    def __init__(
        self,
        model_path='models/weights/face_expre_weights.tflite',
        num_threads=4,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.last_index = 2

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)
        if np.max(result) >= 0.85:
            result_index = np.argmax(np.squeeze(result))
            self.last_index =  result_index
            return result_index
        else:
            return self.last_index

class HandPoseRecognition(object):
    def __init__(
        self,
        model_path='models/weights/hand_pose_weights.tflite',
        num_threads=4,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index

class BodySequenceRecognition(object):
    def __init__(
        self,
        model_path='models/weights/body_sequence_weights.h5'
    ):
        # self.interpreter = tf.lite.Interpreter(model_path=model_path,
        #                                        num_threads=num_threads)
        self.classes = os.listdir('data/body_sequence')
        self.model = Sequential()
        self.model.add(LSTM(16, return_sequences=True, activation='relu', input_shape=(20,183)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(12, return_sequences=True, activation='relu'))
        self.model.add(LSTM(12, return_sequences=False, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(len(self.classes), activation='softmax'))
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        # self.interpreter.allocate_tensors()
        # self.input_details = self.interpreter.get_input_details()
        # self.output_details = self.interpreter.get_output_details()
        # self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.load_weights(model_path)

    def __call__(
        self,
        sequence,
    ):
        # input_details_tensor_index = self.input_details[0]['index']
        # self.interpreter.set_tensor(
        #     input_details_tensor_index,
        #     np.array([landmark_list], dtype=np.float32))
        # self.interpreter.invoke()

        # output_details_tensor_index = self.output_details[0]['index']
        # print(len(landmark_list), np.array([landmark_list]))
        result = self.model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        top_predict = np.argsort(result)[::-1][0]
        top_percent = f"{result[top_predict] * 100:5.1f}"
        # result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))
        proba = float(top_percent)
        return result_index, proba
    
