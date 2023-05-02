from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties, TextNode
from tkinter import Frame
import customtkinter as ctk
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import *
from PIL import Image
import threading
import speech_recognition as sr

class SpeechSign(ShowBase):

    def __init__(self):
        ShowBase.__init__(self, windowType = 'none')
        
        base.startTk()
        self.main_gui = base.tkRoot
        self.screen_width = self.main_gui.winfo_screenwidth()
        self.screen_height = self.main_gui.winfo_screenheight()
        self.h = 620
        self.w = 700
        self.x = int((self.screen_width/2) - (self.w/2))
        self.y = int((self.screen_height/2) - (self.h/1.9))
        self.main_gui.geometry(f"{self.w}x{self.h}+{self.x}+{self.y}")
        self.main_gui.grid_propagate(False)
        self.main_gui.title('KUMPAS FSL TRANSLATOR')
        self.main_gui.resizable(False, False)

        self.panda_frame_height = 510
        self.panda_frame = Frame(self.main_gui, bg="red", height=self.panda_frame_height, width=self.w)
        self.panda_frame.grid(row=0, column=0)
        props = WindowProperties()
        props.set_parent_window(self.panda_frame.winfo_id())
        props.set_origin(0, 0)
        props.set_size(self.panda_frame.winfo_width(), self.panda_frame.winfo_height())

        base.make_default_pipe()
        base.open_default_window(props = props)

        self.panda_frame.bind("<Configure>", self.resize)

        self.anims = {
                'Alin': 'Animations/Alin-rigAction.egg',
                'Attached': 'Animations/Attached-rigAction.egg', #wasak
                'Bad': 'Animations/Bad-rigAction.egg',
                'Bakit': 'Animations/Bakit-rigAction.egg',
                'Balanse': 'Animations/Balanse-rigAction.egg',
                'Beautiful': 'Animations/Beautiful-rigAction.egg',
                'Big': 'Animations/Big-rigAction.egg',
                'Boastful': 'Animations/Boastful-rigAction.egg',
                'Bored': 'Animations/Bored-rigAction.egg',
                'Bumili': 'Animations/Bumili-rigAction.egg',
                'Cold': 'Animations/Cold-rigAction.egg',
                'Complain': 'Animations/Complain-rigAction.egg',
                'Day': 'Animations/Day-rigAction.egg',
                'Utang': 'Animations/Debt-rigAction.egg', #wasak
                'Difficult': 'Animations/Difficult-rigAction.egg',
                'Dollar': 'Animations/Dolyar-rigAction.egg',
                'Dumb': 'Animations/Dumb-rigAction.egg',
                'Easy': 'Animations/Easy-rigAction.egg',
                'Fast': 'Animations/Fast-rigAction.egg',
                'How Are You': 'Animations/How_are_you-rigAction.egg',
                'Hug': 'Animations/Hug-rigAction.egg', #wasak
                'Love': 'Animations/Love-Amy-rigAction.egg', #wasak
                'Tuesday': 'Animations/Tuesday-rigAction.egg',
                'Ugly': 'Animations/Ugly-rigAction.egg', #wasak
                'Wednesday': 'Animations/Wednesday-rigAction.egg', #wasak
                'Linggo': 'Animations/Week-rigAction.egg', #wasak
                'Worried': 'Animations/Worried-rigAction.egg', #wasak
                'Yours': 'Animations/Yours-rigAction.egg',
                'Yourself': 'Animations/Yourself-rigAction.egg'
            }

        self.cam.setPos(0, -58, 6)
        self.cam.node().getDisplayRegion(0).setSort(20)
        
        self.character = Actor('Animations/Alin.egg', self.anims)
        self.character.reparentTo(render)
        self.character.setPos(0,0,-30)
        self.character.setScale(33,33,33)
        self.mic_icon = ctk.CTkImage(Image.open("img/mic.png"), size=(64, 64))
        self.mic_recording = ctk.CTkImage(Image.open("img/mic_recording.png"), size=(64, 64))
        self.mic_hover = ctk.CTkImage(Image.open("img/mic_hover.png"), size=(64, 64))
        self.record_btn = ctk.CTkButton(self.main_gui, background_corner_colors=None, bg_color='transparent', hover=None, height=69, width=69, corner_radius=0, text='', image=self.mic_icon, fg_color='transparent', border_width=0, command=self.record_clicked)
        self.record_btn.grid_propagate(False)
        self.record_btn.grid(row=1, column=0, pady=2)
        self.record_btn.bind("<Enter>", self.hover_effect)
        self.record_btn.bind("<Leave>", self.hover_remove)
        self.clicked_record = False
        self.text_font = ctk.CTkFont(family="Calibri", size=26)
        self.text_string = ctk.CTkLabel(self.main_gui, anchor='w', text='', font=self.text_font, text_color="black")
        self.text_string.grid(row=2, column=0, padx=30)
        self.recognizer = sr.Recognizer()
        self.mic_index = 2
        self.text = TextNode('text')
        self.text.setWordwrap(6)
        self.text.setAlign(TextNode.ACenter)
        self.textNodePath = aspect2d.attachNewNode(self.text)
        self.textNodePath.setPos(0.8, 255, 0.3)
        self.textNodePath.setScale(0.13)
        base.disableMouse()
    
    def hover_effect(self, e):
        self.record_btn.configure(image=self.mic_hover)

    def hover_remove(self, e):
        self.record_btn.configure(image=self.mic_icon)

    def resize(self, event):
        props = WindowProperties()
        props.set_origin(0, 0)
        props.set_size(self.panda_frame.winfo_width(), self.panda_frame.winfo_height())
        base.win.request_properties(props)
    
    def record_clicked(self):
        if not self.clicked_record:
            self.clicked_record = True
            self.record_btn.unbind("<Enter>")
            self.record_btn.unbind("<Leave>")
            self.record_btn.configure(image=self.mic_recording, state='disabled')
            self.record_thread = threading.Thread(target=self.get_speech, daemon=True)
            self.record_thread.start()
    
    def display_word(self, word):
        self.text.setText(word)
    
    def process_text(self, text):
        speech_text = text.title()
        to_process = []
        phrase = []
        for word in speech_text.split():
            for anim in list(self.anims.keys()):
                if word == anim:
                    to_process.append(anim)
                    continue
                if word in anim.split(" ") and word != anim:
                    phrase.append(word)
                if " ".join(phrase) in list(self.anims.keys()):
                    to_process.append(" ".join(phrase))
                    phrase.clear()

        animSequence = Sequence(name='anim sequence')
        for word in to_process:
            animSequence.append(Func(self.display_word, word))
            animSequence.append(self.character.actorInterval(self.character.play(word), playRate=4))
            animSequence.append(Wait(0.2))
        animSequence.append(Func(self.display_word, ''))
        animSequence.start()
        self.text_string.configure(text=" ".join(to_process).capitalize())
        self.record_btn.configure(image=self.mic_icon, state='normal')
        self.clicked_record = False
        self.record_btn.bind("<Enter>", self.hover_effect)
        self.record_btn.bind("<Leave>", self.hover_remove)

    def get_speech(self):
        text = ''
        while True:
            if self.clicked_record:
                try:
                    with sr.Microphone(self.mic_index) as mic:
                        self.recognizer.adjust_for_ambient_noise(mic, 0.2)
                        audio = self.recognizer.record(mic, duration=4)
                        text = self.recognizer.recognize_google(audio)
                        text.lower()
                        self.process_text(text)
                        break
                except BaseException as e:
                    self.text_string.configure(text="Unrecognized. Please try again.")
                    self.clicked_record = False
                    self.record_btn.configure(image=self.mic_icon, state='normal')
                    self.record_btn.bind("<Enter>", self.hover_effect)
                    self.record_btn.bind("<Leave>", self.hover_remove)
                    print(e)
                    break
        

if __name__ == "__main__":   
    speech_sign = SpeechSign()
    speech_sign.run()