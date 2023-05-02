import os
from PIL import Image, ImageTk
import customtkinter as ctk
from color import Color
import cv2
from collections import deque
import copy
import threading
from direct.actor.Actor import Actor

class SpeechToSign(ctk.CTk):
    def __init__(self, x, y):
        super().__init__()
        self.h = 480
        self.w = 220
        self.x = x
        self.y = y
        self.title('OPTIONS')
        
        for i in range(5):
            btn = ctk.CTkButton(self, text=i)
            btn.grid()
            print('x')

        self.geometry(f"{self.w}x{self.h}+{self.x}+{self.y}")
        self.grid_propagate(False)
        self.grab_set()


x = SpeechToSign(100,100)
x.mainloop()
    
