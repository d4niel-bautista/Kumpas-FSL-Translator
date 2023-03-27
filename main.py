import customtkinter as ctk
import os
from PIL import Image

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

        #####LEFT FRAME#####
        self.leftFrame = ctk.CTkFrame(master=self, fg_color="#4B4B4B", width = self.window_width * .21, height= self.window_height * .975, corner_radius=0,border_width=0)
        self.leftFrame.grid(row=0, column=0, sticky='w', padx=0, pady=0)
        self.leftFrame.grid_propagate(False)
        self.leftFrame.grid_rowconfigure(2, weight=1)
        self.leftFrame.grid_columnconfigure((0), weight=1)

        
        self.separator = ctk.CTkFrame(master=self.leftFrame,fg_color="#000000", height=2, border_width=0, width=self.window_width * .16)
        self.separator.grid(row=0, column=0, pady=(int(self.window_height * .1058), int(self.window_height * .0147)), columnspan=2)
        self.leftFrame_firstFrame = ctk.CTkFrame(master=self.leftFrame, fg_color="#D9D9D9", width = self.window_width * .15, height= self.window_height * .3176, corner_radius=15, border_width=0)
        self.leftFrame_firstFrame.grid(row=1, column=0, sticky='n', padx=0, pady=0, columnspan=2)

        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.questionmark = ctk.CTkImage(Image.open(self.current_path + "/img/shutdown.png"), size=(64,64))
        self.leftFrame_shutdown = ctk.CTkButton(master=self.leftFrame, image=self.questionmark, text="", fg_color="#4B4B4B", width=self.window_width * .015, height=self.window_height * .0588, border_width=0)
        self.leftFrame_shutdown.grid(row=2, column=0,sticky='ws', padx=10, pady=20)
        
        #####RIGHT FRAME#####
        self.rightFrame = ctk.CTkFrame(master=self, fg_color="#FFFFFF", width = self.window_width * .80, height= self.window_height * .975, corner_radius=0, border_width=0)
        self.rightFrame.grid(row=0, column=1, sticky='new', padx=0, pady=0)
        self.rightFrame.grid_propagate(False)
            #####TITLE FRAME#####
        self.titleFrame = ctk.CTkFrame(master=self.rightFrame, fg_color="#949090", width = self.window_width * .80, height= self.window_height * .2, corner_radius=0, border_width=0)
        self.titleFrame.grid(row=0, column=0, sticky='new', padx=0, pady=0)
        self.titleFrame.grid_propagate(False)
        
        
        self.titleGroup = ctk.CTkFrame(master=self.titleFrame,fg_color="#949090")
        self.titleGroup.grid(row=0, column=0, sticky='nsew', padx=180, pady=0)
       
        
        self.kumpasLogo = ctk.CTkImage(Image.open(self.current_path + "/img/kumpas.png"), size=(184,108))
        self.kumpasLogoLabel = ctk.CTkLabel(master=self.titleFrame, image=self.kumpasLogo, text="", fg_color='transparent')
        self.kumpasLogoLabel.place(relx = 0.34, rely = 0.4, anchor = 'e')
        self.titleSeparator = ctk.CTkFrame(master=self.titleFrame,fg_color="#FFFFFF", height=2, border_width=0, width=self.window_width * .43)
        self.titleSeparator.place(relx = 0.468, rely = 0.7, anchor = 'center')
        self.titleLabel = ctk.CTkLabel(master=self.titleFrame, text="FILIPINO SIGN LANGUAGE TRANSLATOR", text_color='#FFFFFF',font=ctk.CTkFont(size=25))
        self.titleLabel.place(relx = 0.443, rely = 0.83, anchor = 'center')
            #####MAIN FRAME#####
        self.mainFrame = ctk.CTkFrame(master=self.rightFrame, fg_color="#FFFFFF", width = self.window_width * .80, height= self.window_height * .758, corner_radius=0, border_width=0)
        self.mainFrame.grid(row=1, column=0, sticky='news', padx=0, pady=0)
        self.mainFrame.grid_propagate(False)
                #####MAIN FRAME - SIGN TO TEXT#####
        self.mainFrame_signToText = ctk.CTkFrame(master=self.mainFrame, fg_color="#FFFFFF", width = self.window_width * .80, height= (self.window_height * .758)/2, corner_radius=0, border_width=0)
        self.mainFrame_signToText.grid(row=0, column=0, sticky='nsew', padx=0, pady=0)
        self.mainFrame_signToText.grid_propagate(False)

        self.mainFrame_signToTextGroup = ctk.CTkFrame(master=self.mainFrame_signToText, fg_color="#FFFFFF", width = (self.window_width * .80)/2, height= (self.window_height * .758)/2, corner_radius=0, border_width=0)
        self.mainFrame_signToTextGroup.grid(row=0, column=0, sticky='nsew', padx=0, pady=0)
        self.mainFrame_signToTextGroup.grid_propagate(False)
        self.signToTextLogo = ctk.CTkImage(Image.open(self.current_path + "/img/sign-text.png"), size=(64,64))
        self.signToTextLogoLabel = ctk.CTkLabel(master=self.mainFrame_signToTextGroup, image=self.signToTextLogo, text=" SIGN TO TEXT", compound='left',text_color='#000000',font=ctk.CTkFont(size=25))
        self.signToTextLogoLabel.grid(pady=8, padx=20, row=0,column=1)
        self.signToTextDesc = ctk.CTkLabel(master=self.mainFrame_signToTextGroup, text="DESCRIPTION AWITTTTTTTTTTTTTTT",text_color='#000000',font=ctk.CTkFont(size=25))
        self.signToTextDesc.grid(pady=8, padx=20, row=1,column=1)
                #####MAIN FRAME - SPEECH TO SIGN#####
        self.mainFrame_speechToSign = ctk.CTkFrame(master=self.mainFrame, fg_color="#FFFFFF", width = self.window_width * .80, height= (self.window_height * .758)/2, corner_radius=0, border_width=0)
        self.mainFrame_speechToSign.grid(row=1, column=0, sticky='nesw', padx=0, pady=0)
        self.mainFrame_speechToSign.grid_propagate(False)
        self.mainFrame_speechToSign.grid_columnconfigure((0,1),weight=1)

        self.mainFrame_speechToSignGroup = ctk.CTkFrame(master=self.mainFrame_speechToSign,fg_color="#FFFFFF", width = (self.window_width * .80)/2, height= (self.window_height * .758)/2, corner_radius=0, border_width=0)
        self.mainFrame_speechToSignGroup.grid(row=0, column=1, sticky='e', padx=0, pady=0)
        self.mainFrame_speechToSignGroup.grid_propagate(False)
        self.speechToSignLogo = ctk.CTkImage(Image.open(self.current_path + "/img/speech-sign.png"), size=(64,64))
        self.speechToSignLogoLabel = ctk.CTkLabel(master=self.mainFrame_speechToSignGroup, image=self.speechToSignLogo, text="SPEECH TO SIGN ", compound='right',text_color='#000000',font=ctk.CTkFont(size=25))
        self.speechToSignLogoLabel.grid(pady=8, padx=20, row=0,column=1,sticky='e')
        self.speechToSignDesc = ctk.CTkLabel(master=self.mainFrame_speechToSignGroup, text="DESCRIPTION AWITTTTTT",text_color='#000000',font=ctk.CTkFont(size=25))
        self.speechToSignDesc.grid(pady=8, padx=20, row=1,column=1,sticky='e')
                #####BOTTOM FRAME#####
        self.bottomFrame = ctk.CTkFrame(master=self, fg_color="#D9D9D9", width = self.window_width, height= self.window_height * .049, corner_radius=0, border_width=0)
        self.bottomFrame.grid(row=1, column=0, sticky='sew', padx=0, pady=0, columnspan=2)
        #self.overrideredirect(True)







if __name__ == "__main__":
    app = App()
    app.mainloop()