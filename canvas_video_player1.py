""" tkVideo: Python module for playing videos (without sound) inside tkinter Label widget using Pillow and imageio

Copyright © 2020 Xenofon Konitsas <konitsasx@gmail.com>
Released under the terms of the MIT license (https://opensource.org/licenses/MIT) as described in LICENSE.md

"""

import tkinter as tk  # for Python3
import threading

from PIL import Image, ImageTk
import time
import cv2
import os
class CanvasVideoPlayer():
    """ 
        Main class of tkVideo. Handles loading and playing 
        the video inside the selected label.
        
        :keyword path: 
            Path of video file
        :keyword label: 
            Name of label that will house the player
        :param loop:
            If equal to 0, the video only plays once, 
            if not it plays in an infinite loop (default 0)
        :param size:
            Changes the video's dimensions (2-tuple, 
            default is 640x360) 
    
    """
    def __init__(self, path, label, loop = 0, size = (100,100)):
        self.path = path
        self.label = label
        self.loop = loop
        self.size = size
    
    def load(self, path, label, loop):
        """
            Loads the video's frames recursively onto the selected label widget's image parameter.
            Loop parameter controls whether the function will run in an infinite loop
            or once.
        """
        path = "test.mp4"
        cap = cv2.VideoCapture(path)
        # frame_data = imageio.get_reader(path)
        frames = []
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            resized_image = img.resize((1000,270), Image.ANTIALIAS)
            photo_image = ImageTk.PhotoImage(image=resized_image)
            frames.append(photo_image)
        cap.release()
        cv2.destroyAllWindows()
        bgimg = label.create_image(500,135, image=frames[0])
        logo = (Image.open("./img/sign-text.png"))
        #Resize the Image using resize method
        resized_logo= logo.resize((64,64), Image.ANTIALIAS)
        new_image= ImageTk.PhotoImage(resized_logo)

        #Add image to the Canvas Items
        label.create_image(600,50, image=new_image)
        label.create_text(720,50, text="SIGN TO TEXT", fill="black", font=('Helvetica 15 bold'))
        while True:
            for i in frames:
                label.itemconfig(bgimg, image=i)
                # img = Image.fromarray(i)
                # photo_image = ImageTk.PhotoImage(image=img)
                # try:
                #     label.itemconfig(bgimg, image=photo_image)
                # except Exception as e:
                #     print("display failure:  ", e)
                time.sleep(0.02)
            

        # if loop == 1:
        #     while True:
        #         for image in frame_data.iter_data():
        #             frame_image = ImageTk.PhotoImage(Image.fromarray(image).resize(self.size))
        #             label.create_image(50,50, image=frame_image)
        #             # time.sleep(0.01)
        #             label.create_text(300, 50, text="HELLO WORLD", fill="black", font=('Helvetica 15 bold'))
        #             # time.sleep(0.01)
        # else:
        #     while True:
        #         for image in frame_data.iter_data():
        #             frame_image = ImageTk.PhotoImage(Image.fromarray(image).resize(self.size))
        #             label.create_image(50,50, image=frame_image)
        #             # time.sleep(0.01)
        #             label.create_text(300, 50, text="HELLO WORLD", fill="black", font=('Helvetica 15 bold'))
        #             time.sleep(0.01)

    def play(self):
        """
            Creates and starts a thread as a daemon that plays the video by rapidly going through
            the video's frames.
        """
        thread = threading.Thread(target=self.load, args=(self.path, self.label, self.loop))
        thread.daemon = 1
        thread.start()
