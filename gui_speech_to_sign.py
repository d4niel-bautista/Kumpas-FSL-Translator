import os
import customtkinter as ctk
from tkinter import Frame
from PIL import Image

class SpeechToSignGUI():
    def __init__(self, app):
        self.app = app
        self.app.main_gui.grid_columnconfigure((0,1), weight=1)
        self.app.main_gui.iconbitmap("img/kumpas_icon.ico")
        self.panda_frame_height = 510
        self.panda_frame = Frame(self.app.main_gui, bg="red", height=self.panda_frame_height, width=self.app.w)
        self.panda_frame.grid(row=0, column=0, sticky='nsew')

        self.bottom_frame = ctk.CTkFrame(self.app.main_gui, fg_color="transparent", corner_radius=0)
        self.bottom_frame.grid_propagate(False)
        self.bottom_frame.grid(row=1, column=0, sticky='nsew')
        self.bottom_frame.grid_columnconfigure(0, weight=1)

        self.mic_icon = ctk.CTkImage(Image.open("img/mic.png"), size=(64, 64))
        self.mic_recording = ctk.CTkImage(Image.open("img/mic_recording.png"), size=(64, 64))
        self.mic_hover = ctk.CTkImage(Image.open("img/mic_hover.png"), size=(64, 64))
        self.record_btn = ctk.CTkButton(self.bottom_frame, background_corner_colors=None, bg_color='transparent', hover=None, height=69, width=69, corner_radius=0, text='', image=self.mic_icon, fg_color='transparent', border_width=0, command=self.record_clicked)
        self.record_btn.grid_propagate(False)
        self.record_btn.grid(row=1, column=0, pady=2)
        self.record_btn.bind("<Enter>", lambda e, x=self.record_btn:x.configure(image=self.mic_hover))
        self.record_btn.bind("<Leave>", lambda e, x=self.record_btn:x.configure(image=self.mic_icon))
        self.clicked_record = False
        self.text_font = ctk.CTkFont(family="Calibri", size=26)
        self.text_string = ctk.CTkLabel(self.bottom_frame, anchor='w', text='', font=self.text_font, text_color="black")
        self.text_string.grid(row=2, column=0, padx=30)

        self.options_frame = ctk.CTkFrame(self.app.main_gui, fg_color="#333333", corner_radius=0, width=self.app.extra - self.app.w, height=self.app.h)
        self.options_frame.grid(row=0, column=1, rowspan=2, sticky='nsew')
        self.options_frame.grid_propagate(False)
        self.options_frame.grid_columnconfigure(0, weight=1)
        self.options_frame.grid_columnconfigure((0,2), weight=1)

        self.gui_bg = ctk.CTkFrame(self.options_frame, width=180)
        self.gui_bg.grid(column=1, pady=(20,2))
        self.sign_menu = ctk.CTkLabel(self.gui_bg, text='Words with Animation')
        self.sign_menu.grid(column=1, pady=4)
        self.gui_scrll_frame = ctk.CTkScrollableFrame(self.gui_bg, width=180, corner_radius=0, height=500, bg_color='transparent')
        self.gui_scrll_frame.grid_columnconfigure((0,3), weight=1)
        self.gui_scrll_frame.grid(column=1)

        self.clear_output_btn = ctk.CTkButton(self.options_frame, text='Clear Output', width=75, fg_color='#ECECEC', hover_color='#D5D5D5', text_color='#333333', command=lambda x=self.text_string: x.configure(text=''))
        self.clear_output_btn.place(relx=0.0376, rely=0.8244)
        self.load_words()
    
    def record_clicked(self):
        if not self.clicked_record:
            self.clicked_record = True
            self.record_btn.unbind("<Enter>")
            self.record_btn.unbind("<Leave>")
            self.record_btn.configure(image=self.mic_recording, state='disabled')
            self.app.record_clicked()
    
    def reset_mic_btn(self):
        self.clicked_record = False
        self.record_btn.configure(image=self.mic_icon, state='normal')
        self.record_btn.bind("<Enter>", lambda e, x=self.record_btn:x.configure(image=self.mic_hover))
        self.record_btn.bind("<Leave>", lambda e, x=self.record_btn:x.configure(image=self.mic_icon))

    def load_words(self):
        for i in self.gui_scrll_frame.winfo_children():
            for c in i.winfo_children():
                c.destroy()
            i.destroy()
        for i, item in enumerate(self.app.anims.keys()):
            anim_word = ctk.CTkLabel(self.gui_scrll_frame, text=item)
            anim_word.grid(row=i, column=1, sticky='w')
    
