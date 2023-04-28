import customtkinter as ctk

class SignToText(ctk.CTk):
    def __init__(self, x, y, object):
        super().__init__()
        self.h = 480
        self.w = 250
        self.x = x
        self.y = y
        self.title('OPTIONS')
        for i in object.body_seq.labels:
            btn = ctk.CTkButton(self, text=i)
            btn.grid()

        self.geometry(f"{self.w}x{self.h}+{self.x}+{self.y}")
        self.grid_propagate(False)