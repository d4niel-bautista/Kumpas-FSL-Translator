import tkinter as tk
import customtkinter as ctk
import os
from PIL import Image, ImageTk
from color import Color
from sign_to_text import SignToText
from speech_to_sign import SpeechToSign
from canvas_video_player import CanvasVideoPlayer

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
        self.mainFrame_signToText.grid_columnconfigure(0,weight=1)

        self.lblVideo = ctk.CTkCanvas(self.mainFrame_signToText, borderwidth=0,highlightthickness=0,bg='black', height= (self.window_height * .758)/2)
        self.lblVideo.grid(pady=0, padx=0, row=0,column=0, sticky='news', ipadx=0,ipady=0)
        self.player = CanvasVideoPlayer(canvas=self.lblVideo, video_path='videos/test.mp4', img_path='img/sign-text.png', 
                                        vid_size=(int(self.window_width * .80),int((self.window_height * .758)/2)), vid_pos=(0,0),
                                        img_pos=((self.window_width * .80)*.4,((self.window_height * .758)/2)*.131926), img_size=(64,64),
                                        text_pos=((self.window_width * .80)*.27,((self.window_height * .758)/2)*.18846), text="SIGN TO TEXT", font='Helvetica 19 bold',
                                        desc_pos=((self.window_width * .80)*.303,((self.window_height * .758)/2)* .5277), desc_text="Sign language to text\ntranslation is a process\nof converting sign language\ninto written text.", desc_font='Helvetica 19')
        self.player.play()
        self.lblVideo.bind('<Button-1>', self.signToText)

                #####MAIN FRAME - SPEECH TO SIGN#####
        self.mainFrame_speechToSign = ctk.CTkFrame(master=self.mainFrame, fg_color=self.color.white, width = self.window_width * .80, height= (self.window_height * .758)/2, corner_radius=0, border_width=0)
        self.mainFrame_speechToSign.grid(row=1, column=0, sticky='nesw', padx=0, pady=0)
        self.mainFrame_speechToSign.grid_propagate(False)
        self.mainFrame_speechToSign.grid_columnconfigure(0,weight=1)
        
        self.lblVideo1 = ctk.CTkCanvas(self.mainFrame_speechToSign, borderwidth=0,highlightthickness=0, height= (self.window_height * .758)/2)
        self.lblVideo1.grid(pady=0, padx=0, row=0,column=0, sticky='news', ipadx=0,ipady=0)
        self.player1 = CanvasVideoPlayer(canvas=self.lblVideo1, video_path='videos/slow.mp4', img_path='img/speech-sign.png',
                                         vid_size=(int(self.window_width * .80),int((self.window_height * .758)/2)), vid_pos=(0,0),
                                         img_pos=((self.window_width * .80) *.5729,((self.window_height * .758)/2) * .15077), img_size=(64,64),
                                         text_pos=((self.window_width * .80) * .72916,((self.window_height * .758)/2) * .18846), text="SPEECH TO SIGN", font='Helvetica 19 bold',
                                         desc_pos=((self.window_width * .80) * .7,((self.window_height * .758)/2) * .5277), desc_text="Speech to sign translation\nuses speech recognition\ntechnology to convert spoken\nwords into animation.", desc_font='Helvetica 19' )
        self.player1.play()
        self.lblVideo1.bind('<Button-1>', self.speechToSign)
       
                #####BOTTOM FRAME#####
        self.bottomFrame = ctk.CTkFrame(master=self, fg_color=self.color.very_light_gray, width = self.window_width, height= self.window_height * .049, corner_radius=0, border_width=0)
        self.bottomFrame.grid(row=1, column=0, sticky='sew', padx=0, pady=0, columnspan=2)
        self.signToText_window = None
        self.speechToSign_window = None
        #self.overrideredirect(1)

    def signToText(self,a):
        if self.signToText_window is None or not self.signToText_window.winfo_exists():
            self.signToText_window = SignToText(self)  # create window if its None or destroyed
            self.withdraw()
            self.signToText_window.returnBtn.configure(command=self.clicked_signToText)
            
        else:
            self.signToText_window.focus()  # if window exists focus it
            self.withdraw()
            self.signToText_window.returnBtn.configure(command=self.clicked_signToText)


    def speechToSign(self,a):
        if self.speechToSign_window is None or not self.speechToSign_window.winfo_exists():
            self.speechToSign_window = SpeechToSign(self)  # create window if its None or destroyed
            self.withdraw()
            self.speechToSign_window.returnBtn.configure(command=self.clicked_speechToSign)
        else:
            self.speechToSign_window.focus()  # if window exists focus it
            self.withdraw()
            self.speechToSign_window.returnBtn.configure(command=self.clicked_speechToSign)

    def clicked_speechToSign(self):
        self.deiconify()
        self.speechToSign_window.destroy()
    
    def clicked_signToText(self):
        self.deiconify()
        self.signToText_window.destroy()
        





if __name__ == "__main__":
    app = App()
    app.mainloop()