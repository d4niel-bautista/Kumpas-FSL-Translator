import numpy as np
import os
from collections import deque
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import InputLayer, Dense, Dropout, LSTM
from keras.models import Sequential
import tensorflow as tf
import sys
sys.stdout.flush()
sys.stderr.flush()

RANDOM_SEED = 42

def train_lstm():
    sequence = []
    label = []
    path = 'data/body_sequence'
    train_path = 'models/weights/body_sequence_weights.h5'
    label_map = {label:num for num, label in enumerate(os.listdir(path))}
    for word in os.listdir(path):
        for seq in os.listdir(os.path.join(path, word)):
            frames = []
            for data in os.listdir(os.path.join(path, word, seq)):
                loaded_npy = np.load(os.path.join(path, word, seq, data))
                frames.append(loaded_npy)
            sequence.append(frames)
            label.append(label_map[word])

    X = np.array(sequence)
    y = to_categorical(label).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=RANDOM_SEED)
    model = Sequential()
    model.add(LSTM(16, return_sequences=True, activation='relu', input_shape=(20,183)))
    model.add(Dropout(0.2))
    model.add(LSTM(12, return_sequences=True, activation='relu'))
    model.add(LSTM(12, return_sequences=False, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(len(label_map), activation='softmax'))
    model.summary()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        train_path, verbose=1, save_weights_only=True)

    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train,
        y_train,
        epochs=1000,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=[cp_callback, es_callback]
    )

def train_hand_pose():
    pass

def train_face_expre():
    pass