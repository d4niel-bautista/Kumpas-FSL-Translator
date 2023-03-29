
import customtkinter as ctk

class SpeechToSign(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.grid_rowconfigure((0), weight=1)
        self.grid_columnconfigure((0), weight=1)
        
        self.title('SPEECH TO SIGN')

        self.label = ctk.CTkLabel(self, text="SPEECH TO SIGN")
        self.label.pack(padx=20, pady=20)

        self.grab_set()