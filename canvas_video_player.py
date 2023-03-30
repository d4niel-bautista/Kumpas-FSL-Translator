""" tkVideo: Python module for playing videos (without sound) inside tkinter canvas widget using Pillow and imageio

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
        the video inside the selected canvas.
        
        :keyword path: 
            Path of video file
        :keyword canvas: 
            Name of canvas that will house the player
        :param loop:
            If equal to 0, the video only plays once, 
            if not it plays in an infinite loop (default 0)
        :param size:
            Changes the video's dimensions (2-tuple, 
            default is 640x360) 
    
    """
    def __init__(self, video_path, canvas, img_path, loop = True, vid_size = (100,100), img_size=(64,64), img_pos = (0,0), text="TEST", text_pos=(0,0), font='Helvetica 15 bold', desc_pos=(700, 100), desc_text="wow", desc_font='Helvetica 19' ):
        self.video_path = video_path
        self.canvas = canvas
        self.loop = loop
        self.vid_size = vid_size
        self.frames = []
        self.loaded_vid = False
        self.img_path = img_path
        self.img_size = img_size
        self.img_pos =img_pos
        self.text = text
        self.text_pos = text_pos
        self.font = font
        self.desc_pos = desc_pos
        self.desc_font = desc_font
        self.desc_text = desc_text
    def load(self, path, canvas, loop):
        """
            Loads the video's frames recursively onto the selected canvas widget's image parameter.
            Loop parameter controls whether the function will run in an infinite loop
            or once.
        """
        path = "test.mp4"
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            resized_image = img.resize(self.vid_size, Image.ANTIALIAS)
            photo_image = ImageTk.PhotoImage(image=resized_image)
            frames.append(photo_image)
        cap.release()
        cv2.destroyAllWindows()
        bgimg = canvas.create_image(500,135, image=frames[0])
        self.loaded_vid = True
        print(self.loaded_vid)
        logo = (Image.open(self.img_path))
        resized_logo= logo.resize(self.img_size, Image.ANTIALIAS)
        new_image= ImageTk.PhotoImage(resized_logo)
        canvas.create_image(self.img_pos[0],self.img_pos[1], image=new_image)
        canvas.create_text(self.text_pos[0],self.text_pos[1], text=self.text, fill="black", font=(self.font))
        canvas.create_text(self.desc_pos[0],self.desc_pos[1], text=self.desc_text, fill="black", font=(self.desc_font))
        while self.loop:
            for i in frames:
                canvas.itemconfig(bgimg, image=i)
                time.sleep(0.02)
    
    # def add_image(self, path, size, x,y):
    #     self.img_thread = threading.Thread(target=self.create_image, args=(path, self.canvas, size, x,y))
    #     self.img_thread.daemon = 1
    #     self.img_thread.start()
    
    # def create_image(self, path, canvas, size, x,y):
    #     while not self.loaded_vid:
    #         print(self.loaded_vid)
    #         continue
        
        print('test')
        
    def play(self):
        """
            Creates and starts a thread as a daemon that plays the video by rapidly going through
            the video's frames.
        """
        self.vid_thread = threading.Thread(target=self.load, args=(self.video_path, self.canvas, self.loop))
        self.vid_thread.daemon = 1
        self.vid_thread.start()
