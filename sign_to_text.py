import os
from PIL import Image, ImageTk
import customtkinter as ctk
from color import Color
class SignToText(ctk.CTkToplevel):
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

        self.title('SIGN TO TEXT')

        self.titleFrame = ctk.CTkFrame(master=self, fg_color=self.color.white, width = self.window_width, height= self.window_height * .13, corner_radius=0, border_width=0)
        self.titleFrame.grid(row=0, column=0, sticky='new', padx=0, pady=0)
        self.titleFrame.grid_propagate(False)
        self.titleFrame.grid_columnconfigure((0), weight=1)
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.signToTextLogo = ctk.CTkImage(Image.open(self.current_path + "/img/sign-text.png"), size=(self.window_width * .0533,self.window_height * .0914))
        self.signToTextLogoLabel = ctk.CTkLabel(master=self.titleFrame, image=self.signToTextLogo, text=" SIGN TO TEXT", compound='left',text_color=self.color.black,font=ctk.CTkFont(size=25))
        self.signToTextLogoLabel.grid(pady=self.window_height * .0114, padx=self.window_width * .01666, row=0,column=0,sticky='nswe')

        self.grab_set()