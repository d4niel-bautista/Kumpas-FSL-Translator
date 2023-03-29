from tkinter import *
import customtkinter as ctk
import os
from PIL import Image, ImageTk
from color import Color
from sign_to_text import SignToText
from speech_to_sign import SpeechToSign
from canvas_vid import CanvasVideoPlayer
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('Main Menu')
        ctk.set_appearance_mode("Dark")
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.h = 700
        self.w = 1200
        self.window_width = self.w
        self.window_height = self.h
        self.x_coordinate = int((self.screen_width/2) - (self.window_width/2))
        self.y_coordinate = int((self.screen_height/2) - (self.window_height/1.9))
        self.geometry(f"{self.window_width}x{self.window_height}+{self.x_coordinate}+{self.y_coordinate}")
        self.grid_propagate(False)
        self.grid_rowconfigure((0), weight=1)
        self.grid_columnconfigure((0), weight=1)
        self.color = Color()
        #####LEFT FRAME#####
        self.leftFrame = ctk.CTkFrame(master=self, fg_color=self.color.very_dark_gray, width = self.window_width * .21, height= self.window_height * .975, corner_radius=0,border_width=0)
        self.leftFrame.grid(row=0, column=0, sticky='w', padx=0, pady=0)
        self.leftFrame.grid_propagate(False)
        self.leftFrame.grid_rowconfigure(2, weight=1)
        self.leftFrame.grid_columnconfigure((0), weight=1)

        
        self.separator = ctk.CTkFrame(master=self.leftFrame,fg_color=self.color.black, height=2, border_width=0, width=self.window_width * .16)
        self.separator.grid(row=0, column=0, pady=(int(self.window_height * .1058), int(self.window_height * .0147)), columnspan=2)
        self.leftFrame_firstFrame = ctk.CTkFrame(master=self.leftFrame, fg_color=self.color.very_light_gray, width = self.window_width * .15, height= self.window_height * .3176, corner_radius=15, border_width=0)
        self.leftFrame_firstFrame.grid(row=1, column=0, sticky='n', padx=0, pady=0, columnspan=2)

        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.questionmark = ctk.CTkImage(Image.open(self.current_path + "/img/shutdown.png"), size=(self.window_width * 0.0325,self.window_height * .0557))
        self.leftFrame_shutdown = ctk.CTkButton(master=self.leftFrame, image=self.questionmark, command=self.destroy, text="", fg_color=self.color.very_dark_gray, width=self.window_width * .015, height=self.window_height * .0588, border_width=0)
        self.leftFrame_shutdown.grid(row=2, column=0,sticky='ws', padx=self.window_width * .0083, pady=self.window_height * .02857)
        
        #####RIGHT FRAME#####
        self.rightFrame = ctk.CTkFrame(master=self, fg_color=self.color.white, width = self.window_width * .80, height= self.window_height * .975, corner_radius=0, border_width=0)
        self.rightFrame.grid(row=0, column=1, sticky='new', padx=0, pady=0)
        self.rightFrame.grid_propagate(False)
            #####TITLE FRAME#####
        self.titleFrame = ctk.CTkFrame(master=self.rightFrame, fg_color=self.color.dark_grayish_red, width = self.window_width * .80, height= self.window_height * .2, corner_radius=0, border_width=0)
        self.titleFrame.grid(row=0, column=0, sticky='new', padx=0, pady=0)
        self.titleFrame.grid_propagate(False)
        
        
        self.titleGroup = ctk.CTkFrame(master=self.titleFrame,fg_color=self.color.dark_grayish_red)
        self.titleGroup.grid(row=0, column=0, sticky='nsew', padx=self.window_width * 0.15, pady=0)
       
        
        self.kumpasLogo = ctk.CTkImage(Image.open(self.current_path + "/img/kumpas.png"), size=(self.window_width * .1533,self.window_height * .15428))
        self.kumpasLogoLabel = ctk.CTkLabel(master=self.titleFrame, image=self.kumpasLogo, text="", fg_color=self.color.transparent)
        self.kumpasLogoLabel.place(relx = 0.34, rely = 0.4, anchor = 'e')
        self.titleSeparator = ctk.CTkFrame(master=self.titleFrame,fg_color=self.color.white, height=2, border_width=0, width=self.window_width * .43)
        self.titleSeparator.place(relx = 0.468, rely = 0.7, anchor = 'center')
        self.titleLabel = ctk.CTkLabel(master=self.titleFrame, text="FILIPINO SIGN LANGUAGE TRANSLATOR", text_color=self.color.white,font=ctk.CTkFont(size=25))
        self.titleLabel.place(relx = 0.443, rely = 0.83, anchor = 'center')
            #####MAIN FRAME#####
        self.mainFrame = ctk.CTkFrame(master=self.rightFrame, fg_color=self.color.white, width = self.window_width * .80, height= self.window_height * .758, corner_radius=0, border_width=0)
        self.mainFrame.grid(row=1, column=0, sticky='news', padx=0, pady=0)
        self.mainFrame.grid_propagate(False)
                #####MAIN FRAME - SIGN TO TEXT#####
                
        self.mainFrame_signToText = ctk.CTkFrame(master=self.mainFrame, fg_color=self.color.white, width = self.window_width * .80, height= (self.window_height * .758)/2, corner_radius=0, border_width=0)
        self.mainFrame_signToText.grid(row=0, column=0, sticky='nsew', padx=0, pady=0)
        self.mainFrame_signToText.grid_propagate(False)
        self.mainFrame_signToText.grid_columnconfigure((0,1),weight=1)

        self.mainFrame_signToTextGroup = ctk.CTkFrame(master=self.mainFrame_signToText, fg_color=self.color.transparent, width = (self.window_width * .80)/2, height= (self.window_height * .758)/2, corner_radius=0, border_width=0)
        self.mainFrame_signToTextGroup.grid(row=0, column=0, sticky='nsew', padx=0, pady=0)
        self.mainFrame_signToTextGroup.grid_propagate(False)
        self.mainFrame_signToTextGroup.bind('<Button-1>', self.signToText)
        self.signToTextLogo = ctk.CTkImage(Image.open(self.current_path + "/img/sign-text.png"), size=(self.window_width * .0533,self.window_height * .0914))
        self.signToTextLogoLabel = ctk.CTkLabel(master=self.mainFrame_signToTextGroup, image=self.signToTextLogo, text=" SIGN TO TEXT", compound='left',text_color=self.color.black,font=ctk.CTkFont(size=25))
        self.signToTextLogoLabel.grid(pady=self.window_height * .0114, padx=self.window_width * .01666, row=0,column=1,sticky='nswe')
        self.signToTextLogoLabel.bind('<Button-1>', self.signToText)
        self.signToTextDesc = ctk.CTkLabel(master=self.mainFrame_signToTextGroup,wraplength=int(((self.window_width * .80)/2)-30), fg_color=self.color.transparent,text="IT IS USED TO TRANSLATE FILIPINO SIGN LAdasdsadsadsadNGUAGE TO TEXTIT IS USED TO TRANSLATE FILIPINO SIGN LAdasdsadsadsadNGUAGE TO TEXTIT IS USED TO TRANSLATE FILIPINO SIGN LAdasdsadsadsadNGUAGE TO TEXTIT IS USED TO TRANSLATE FILIPINO SIGN LAdasdsadsadsadNGUAGE TO TEXT",text_color=self.color.black,font=ctk.CTkFont(size=15))
        self.signToTextDesc.grid(pady=self.window_height * .0114, padx=self.window_width * .01666, row=1,column=1,sticky='nswe')
        self.signToTextDesc.bind('<Button-1>', self.signToText)
        
        # self.canvas = ctk.CTkCanvas(master=self.mainFrame_signToText, highlightthickness=0, width = self.window_width * .80)
        # self.video = CanvasVideoPlayer(self.canvas,'test.mp4', refresh_rate=0.01)
        # self.video.start()
        # # self.lblVideo = Label(self.canvas, borderwidth=0)
        # # self.lblVideo.grid(pady=0, padx=0, row=0,column=0,sticky='news', ipadx=0,ipady=0)
        
        # self.canvas.grid(row=0, column=0, padx=0, pady=0, sticky='news')
        # self.canvas.create_text(300, 50, text="HELLO WORLD", fill="black", font=('Helvetica 15 bold'))
        # self.signToTextLogoLabel = ctk.CTkLabel(master=self.mainFrame_signToText,text=" SIGN TO TEXT", text_color='#000000',font=ctk.CTkFont(size=25))
        # self.signToTextLogoLabel.grid(pady=8, padx=20, row=0,column=0,sticky='nswe')

        
        
        
                #####MAIN FRAME - SPEECH TO SIGN#####
        self.mainFrame_speechToSign = ctk.CTkFrame(master=self.mainFrame, fg_color=self.color.white, width = self.window_width * .80, height= (self.window_height * .758)/2, corner_radius=0, border_width=0)
        self.mainFrame_speechToSign.grid(row=1, column=0, sticky='nesw', padx=0, pady=0)
        self.mainFrame_speechToSign.grid_propagate(False)
        self.mainFrame_speechToSign.grid_columnconfigure((0,1),weight=1)
        
        # self.lblVideo1 = Canvas(self.mainFrame_speechToSign, borderwidth=0)
        # self.lblVideo1.grid(pady=0, padx=0, row=0,column=0, sticky='news', ipadx=0,ipady=0)
        
        self.mainFrame_speechToSignGroup = ctk.CTkFrame(master=self.mainFrame_speechToSign,fg_color=self.color.white, width = (self.window_width * .80)/2, height= (self.window_height * .758)/2, corner_radius=0, border_width=0)
        self.mainFrame_speechToSignGroup.grid(row=0, column=1, sticky='nsew', padx=0, pady=0)
        self.mainFrame_speechToSignGroup.grid_propagate(False)
        self.mainFrame_speechToSignGroup.bind('<Button-1>', self.speechToSign)
        self.speechToSignLogo = ctk.CTkImage(Image.open(self.current_path + "/img/speech-sign.png"), size=(self.window_width * .0533,self.window_height * .0914))
        self.speechToSignLogoLabel = ctk.CTkLabel(master=self.mainFrame_speechToSignGroup, image=self.speechToSignLogo, text="SPEECH TO SIGN ", compound='right',text_color=self.color.black,font=ctk.CTkFont(size=25))
        self.speechToSignLogoLabel.grid(pady=self.window_height * .0114, padx=self.window_width * .01666, row=0,column=1,sticky='nswe')
        self.speechToSignLogoLabel.bind('<Button-1>', self.speechToSign)
        self.speechToSignDesc = ctk.CTkLabel(master=self.mainFrame_speechToSignGroup,wraplength=int(((self.window_width * .80)/2)-30), text="DESCRIPTIONdsadswwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwadasdsadsadsad AWITTTTT",text_color=self.color.black,font=ctk.CTkFont(size=15))
        self.speechToSignDesc.grid(pady=self.window_height * .0114, padx=self.window_width * .01666, row=1,column=1,sticky='nswe')
        self.speechToSignDesc.bind('<Button-1>', self.speechToSign)
                #####BOTTOM FRAME#####
        self.bottomFrame = ctk.CTkFrame(master=self, fg_color=self.color.very_light_gray, width = self.window_width, height= self.window_height * .049, corner_radius=0, border_width=0)
        self.bottomFrame.grid(row=1, column=0, sticky='sew', padx=0, pady=0, columnspan=2)
        #self.overrideredirect(True)
        self.signToText_window = None
        self.speechToSign_window = None

    def signToText(self,a):
        if self.signToText_window is None or not self.signToText_window.winfo_exists():
            self.signToText_window = SignToText(self)  # create window if its None or destroyed
        else:
            self.signToText_window.focus()  # if window exists focus it

    def speechToSign(self,a):
        if self.speechToSign_window is None or not self.speechToSign_window.winfo_exists():
            self.speechToSign_window = SpeechToSign(self)  # create window if its None or destroyed
        else:
            self.speechToSign_window.focus()  # if window exists focus it





if __name__ == "__main__":
    app = App()
    app.mainloop()