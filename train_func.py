import numpy as np
import os
from models.recog_models import BodySequenceRecognition, HandPoseRecognition, FacialExpressionRecognition
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import InputLayer, Dense, Dropout, LSTM
from keras.models import Sequential
import tensorflow as tf
import shutil
import csv

RANDOM_SEED = 42

def train_lstm(app):
    for i in os.listdir('temp/body_sequence'):
        for j in os.listdir(os.path.join('temp/body_sequence', i)):
            shutil.copytree(os.path.join('temp/body_sequence', i, j), os.path.join('data/body_sequence', i, j))
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

    for i in app.gui_scrll_frame.winfo_children():
        i.configure(state='normal')
    for j in app.bot_frame.winfo_children():
        j.configure(state='normal')
    app.sign_menu.configure(state='normal')
    app.train_btn.configure(state='normal')
    app.status_label.configure(text='')
    app.main_app.body_seq.recog_model = BodySequenceRecognition()
    app.main_app.body_seq.labels = app.main_app.body_seq.get_labels()
    app.word_var.set(0)
    app.update()
    app.main_app.clear_output()

def train_hand_pose(app):
    shutil.rmtree('data/hand')
    shutil.copytree('temp/hand', 'data/hand')
    dataset = 'data/hand/hand_pose_data.csv'
    model_save_path = 'models/weights/hand_pose_weights.hdf5'
    tflite_save_path = 'models/weights/hand_pose_weights.tflite'
    with open('data/hand/hand_pose_labels.csv',
            encoding='utf-8-sig') as g:
            hand_pose_labels = csv.reader(g)
            NUM_CLASSES = len([row[0] for row in hand_pose_labels])
    X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)
    model = Sequential()
    model.add(InputLayer((21 * 2, )))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.summary()
    # Model checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        model_save_path, verbose=1, save_weights_only=True)
    # Callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)
    # Model compilation
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
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

    # #########TESTING############
    # # Model evaluation
    # val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
    # # Loading the saved model
    # model.load_weights(model_save_path)
    # # Inference test
    # predict_result = model.predict(np.array([X_test[0]]))
    # print(np.squeeze(predict_result))
    # print(np.argmax(np.squeeze(predict_result)))

    # Save as a model dedicated to inference
    model.save(model_save_path, include_optimizer=False)
    # Transform model (quantization)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    open(tflite_save_path, 'wb').write(tflite_quantized_model)

    for i in app.gui_scrll_frame.winfo_children():
        i.configure(state='normal')
    for j in app.bot_frame.winfo_children():
        j.configure(state='normal')
    app.sign_menu.configure(state='normal')
    app.train_btn.configure(state='normal')
    app.status_label.configure(text='')
    app.main_app.hand_pose.recog_model = HandPoseRecognition()
    app.main_app.hand_pose.labels = app.main_app.hand_pose.get_labels()
    app.word_var.set(0)
    app.update()
    app.main_app.clear_output()

def train_face_expre(app):
    shutil.rmtree('data/face')
    shutil.copytree('temp/face', 'data/face')
    dataset = 'data/face/face_expre_data.csv'
    model_save_path = 'models/weights/face_expre_weights.hdf5'
    tflite_save_path = 'models/weights/face_expre_weights.tflite'
    with open('data/face/face_expre_labels.csv',
            encoding='utf-8-sig') as g:
            face_expre_labels = csv.reader(g)
            NUM_CLASSES = len([row[0] for row in face_expre_labels])
    X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (478 * 2) + 1)))
    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.8, random_state=RANDOM_SEED)
    model = tf.keras.models.Sequential([
    tf.keras.layers.Input((478 * 2, )),
    tf.keras.layers.Dense(20, activation='elu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='elu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.summary()
    # Model checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        model_save_path, verbose=1, save_weights_only=False)
    # Callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

    # Model compilation
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
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

    # #########TESTING############

    # val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
    # model = tf.keras.models.load_model(model_save_path)
    # # Inference test
    # predict_result = model.predict(np.array([X_test[0]]))
    # print(np.squeeze(predict_result))
    # print(np.argmax(np.squeeze(predict_result)))

    # Save as a model dedicated to inference
    model.save(model_save_path, include_optimizer=False)

    # Transform model (quantization)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    open(tflite_save_path, 'wb').write(tflite_quantized_model)

    for i in app.gui_scrll_frame.winfo_children():
        i.configure(state='normal')
    for j in app.bot_frame.winfo_children():
        j.configure(state='normal')
    app.sign_menu.configure(state='normal')
    app.train_btn.configure(state='normal')
    app.status_label.configure(text='')
    app.main_app.face_expre.recog_model = FacialExpressionRecognition()
    app.main_app.face_expre.labels = app.main_app.face_expre.get_labels()
    app.word_var.set(0)
    app.update()
    app.main_app.clear_output()