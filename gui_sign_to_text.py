import customtkinter as ctk
import shutil
from models.recog_models import BodySequenceRecognition, HandPoseRecognition, FacialExpressionRecognition
import train_func
import threading
from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.inf)

class SignToText(ctk.CTk):
    def __init__(self, x, y, main_app):
        super().__init__()
        self.h = 480+32
        self.w = 220
        self.x = x
        self.y = y
        self.main_app = main_app
        self.title('OPTIONS')
        
        self.geometry(f"{self.w}x{self.h}+{self.x}+{self.y}")
        self.grid_propagate(False)
        self.grid_columnconfigure((0,2), weight=1)
        self.grab_set()
        self.overrideredirect(True)
        self.attributes('-topmost',True)
        
        self.sign_menu_var = ctk.StringVar(value="Body Gesture")
        self.sign_menu = ctk.CTkOptionMenu(self, values=['Body Gesture', 'Hand Pose', 'Facial Expression'], variable=self.sign_menu_var, command=self.sign_dropdown, anchor='c')
        self.sign_menu.grid(column=1, pady=8)

        self.body_seq_labels = self.main_app.body_seq.get_labels()
        self.hand_pose_labels = self.main_app.hand_pose.get_labels()
        self.face_expre_labels = self.main_app.face_expre.get_labels()
        self.body_seq_label_map = {label:num for num, label in enumerate(self.main_app.body_seq.get_labels())}
        self.hand_pose_label_map = {label:num for num, label in enumerate(self.main_app.hand_pose.get_labels())}
        self.face_expre_label_map = {label:num for num, label in enumerate(self.main_app.face_expre.get_labels())}
        self.to_delete_body_seq = []
        self.to_delete_hand_pose = []
        self.to_delete_face_expre = []

        self.gui_bg = ctk.CTkFrame(self, width=180)
        self.gui_bg.grid(column=1, pady=(5,2))
        self.frame_label = ctk.CTkLabel(self.gui_bg, text='', fg_color='transparent')
        self.frame_label.grid(column=1, pady=1)
        self.gui_scrll_frame = ctk.CTkScrollableFrame(self.gui_bg, width=180, corner_radius=0, height=280)
        self.gui_scrll_frame.grid_columnconfigure((0,3), weight=1)
        self.gui_scrll_frame.grid(column=1)

        self.body_seq_var = ctk.IntVar()
        self.hand_pose_var = ctk.IntVar()
        self.face_expre_var = ctk.IntVar()

        self.delete_icon = ctk.CTkImage(Image.open("img/delete.png"), size=(24, 24))

        self.gui_footer = ctk.CTkFrame(self.gui_bg, width=180, fg_color='transparent')
        self.gui_footer.grid(column=1, pady=4)
        self.bot_frame = ctk.CTkFrame(self.gui_footer, fg_color='transparent')
        self.bot_frame.grid(column=1, columnspan=2, sticky='nsew')
        self.bot_frame.grid_columnconfigure((0,3), weight=1)
        self.add_btn = ctk.CTkButton(self.bot_frame, text='+', width=35)
        self.add_btn.grid(row=0, column=1, sticky='e', padx=(0,5))
        self.get_data_btn = ctk.CTkButton(self.bot_frame, text='Collect Data', width=75)
        self.get_data_btn.grid(row=0, column=2, sticky='w')
        self.train_btn = ctk.CTkButton(self, text="TRAIN", command=self.train)
        self.train_btn.grid(column=1, pady=4)
        self.body_seq_init()
    
    def sign_dropdown(self, value):
        if value == 'Body Gesture':
            self.body_seq_init()
        elif value == 'Hand Pose':
            self.hand_pose_init()
        elif value == 'Facial Expression':
            self.face_expre_init()
            
    def body_seq_init(self):
        for i in self.gui_scrll_frame.winfo_children():
            for c in i.winfo_children():
                c.destroy()
            i.destroy()
        for i, item in enumerate(self.body_seq_labels):
            btn_body_seq_label = ctk.CTkRadioButton(self.gui_scrll_frame, text=item, variable=self.body_seq_var, value=i, width=120)
            btn_body_seq_label.grid(row=i, column=1, sticky='e')
            btn_body_seq_label._text_label.configure(wraplength=78)
            delete_body_seq_label = ctk.CTkButton(self.gui_scrll_frame, background_corner_colors=None, bg_color='transparent', hover=None, height=24, width=24, corner_radius=0, text='', image=self.delete_icon, fg_color='transparent', border_width=0, command=lambda x=item: self.body_seq_delete(x))
            delete_body_seq_label.grid(row=i, column=2, sticky='e')

    def hand_pose_init(self):
        for i in self.gui_scrll_frame.winfo_children():
            for c in i.winfo_children():
                c.destroy()
            i.destroy()
        for i, item in enumerate(self.hand_pose_labels):
            btn_hand_pose_label = ctk.CTkRadioButton(self.gui_scrll_frame, text=item, variable=self.hand_pose_var, value=i, width=120)
            btn_hand_pose_label.grid(row=i, column=1, sticky='e')
            btn_hand_pose_label._text_label.configure(wraplength=78)
            delete_hand_pose_label = ctk.CTkButton(self.gui_scrll_frame, background_corner_colors=None, bg_color='transparent', hover=None, height=24, width=24, corner_radius=0, text='', image=self.delete_icon, fg_color='transparent', border_width=0, command=lambda x=i: self.hand_pose_delete(x))
            delete_hand_pose_label.grid(row=i, column=2, sticky='e')

    def face_expre_init(self):
        for i in self.gui_scrll_frame.winfo_children():
            for c in i.winfo_children():
                c.destroy()
            i.destroy()
        for i, item in enumerate(self.face_expre_labels):
            btn_face_expre_label = ctk.CTkRadioButton(self.gui_scrll_frame, text=item, variable=self.face_expre_var, value=i, width=120)
            btn_face_expre_label.grid(row=i, column=1, sticky='e')
            btn_face_expre_label._text_label.configure(wraplength=78)
            delete_face_expre_label = ctk.CTkButton(self.gui_scrll_frame, background_corner_colors=None, bg_color='transparent', hover=None, height=24, width=24, corner_radius=0, text='', image=self.delete_icon, fg_color='transparent', border_width=0, command=lambda x=i: self.face_expre_delete(x))
            delete_face_expre_label.grid(row=i, column=2, sticky='e')

    def body_seq_delete(self, x):
        if len(self.body_seq_labels) == 1:
            return
        self.body_seq_labels.remove(x)
        self.to_delete_body_seq.append(x)
        self.body_seq_init()
    
    def delete_body_seq(self):
        for i in self.to_delete_body_seq:
            shutil.rmtree("data/body_sequence/" + i)
    
    def hand_pose_delete(self, x):
        if len(self.hand_pose_labels) == 1:
            return
        if not x == len(self.hand_pose_labels) - 1:
            pass
        self.hand_pose_labels.pop(x)
        # self.to_delete_hand_pose.append(self.hand_pose_label_map[removed])
        self.hand_pose_init()

    # def subtract_by_one(self, n, x):
    #     if n > x:
    #         return n - 1
    #     else:
    #         return n

    def face_expre_delete(self, x):
        if len(self.face_expre_labels) == 1:
            return
        tmp_hp = np.loadtxt('data/face/face_expre_data.csv', delimiter=',', usecols=list(range((21 * 2) + 1)), dtype='object')
        if not x == len(self.face_expre_labels) - 1:
            tmp_hp[:, 0] = tmp_hp[:, 0].astype(int)
            tmp_hp = tmp_hp[(tmp_hp[:, 0] != x)]
            print(tmp_hp[:, 0])
        #     tmp_hp[(tmp_hp[:, 0] != x)]
            # tmp_hp[:, 0] = self.subtract_by_one(tmp_hp[:, 0].any(), x)
            # for i in tmp_hp[:, 0]:
            #     print(type(i), i)
        np.save("test.npy", tmp_hp)
        z = np.load("test.npy", allow_pickle=True)
        for i in z[:, 0]:
            if z[i, 0] > x:
                print(z[i, 0], x)
        print(z[0, 0], x)
        z[:, 0] = z[:, 0] - 1 if z[:, 0].any() > x else z[:, 0]
        # print(z[:, 0])

        self.face_expre_labels.pop(x)
        # self.to_delete_face_expre.append(self.face_expre_label_map[removed])
        self.face_expre_init()
    
    def train(self):
        for i in self.gui_scrll_frame.winfo_children():
            i.configure(state='disabled')
        print(self.sign_menu_var.get())
        
        