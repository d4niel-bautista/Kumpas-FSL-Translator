import customtkinter as ctk
from PIL import Image, ImageTk
import threading
from moviepy.editor import VideoFileClip
import time

class CanvasVideoPlayer():
    def __init__(self, canvas, file_path, refresh_rate=0.02, looping=True, play_video=True, anchor='nw'):
        self.canvas = canvas
        self.file_path = file_path
        self.refresh_rate = refresh_rate
        self.looping = looping
        self.play_video = play_video
        self.anchor = anchor
        self.frames = []

    def get_video_frames(self):
        file = VideoFileClip(self.file_path)
        size = file.size
        vid_length = file.duration
        clip = file.subclip(0, vid_length)
        frames = file.iter_frames()
        for i in frames:
            i = Image.fromarray(i)
            i.resize(size=size)
            photo_image = ImageTk.PhotoImage(image=i)
            self.frames.append(photo_image)
        
        self.img = self.canvas.create_image(int(self.canvas.winfo_width()/2), int(self.canvas.winfo_height()/2), anchor=self.anchor, image=self.frames[0])
        self.play()
        
    def start(self):
        thread = threading.Thread(target=self.get_video_frames, args=(), daemon=True)
        thread.start()
    
    def play(self):
        self.canvas.create_text(500, 300, text="HELLO WORLD", fill="black", font=('Helvetica 15 bold'))
        while self.play_video:
            for frame in self.frames:
                self.canvas.itemconfig(self.img, image=frame)
                time.sleep(self.refresh_rate)
            if not self.looping:
                break