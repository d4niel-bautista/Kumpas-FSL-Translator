import customtkinter as ctk
import shutil
from models.recog_models import BodySequenceRecognition, HandPoseRecognition, FacialExpressionRecognition
import train_func
from PIL import Image
import numpy as np
import os
import csv
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
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.geometry(f"{self.w}x{self.h}+{self.x}+{self.y}")
        self.grid_propagate(False)
        self.grid_columnconfigure((0,2), weight=1)
        self.grab_set()
        self.overrideredirect(True)
        self.attributes('-topmost',True)
        
        self.sign_menu_var = ctk.StringVar(value="Body Gesture")
        self.sign_menu = ctk.CTkOptionMenu(self, values=['Body Gesture', 'Hand Pose', 'Facial Expression'], variable=self.sign_menu_var, command=self.sign_dropdown, anchor='c')
        self.sign_menu.grid(column=1, pady=(14,4))

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
        self.frame_label = ctk.CTkLabel(self.gui_bg, text=self.sign_menu_var.get(), fg_color='transparent')
        self.frame_label.grid(column=1, pady=1)
        self.gui_scrll_frame = ctk.CTkScrollableFrame(self.gui_bg, width=180, corner_radius=0, height=280)
        self.gui_scrll_frame.grid_columnconfigure((0,3), weight=1)
        self.gui_scrll_frame.grid(column=1)

        self.word_var = ctk.IntVar()

        self.delete_icon = ctk.CTkImage(Image.open("img/delete.png"), size=(24, 24))

        self.gui_footer = ctk.CTkFrame(self.gui_bg, width=180, fg_color='transparent')
        self.gui_footer.grid(column=1, pady=4)
        self.bot_frame = ctk.CTkFrame(self.gui_footer, fg_color='transparent')
        self.bot_frame.grid(column=1, columnspan=2, sticky='nsew')
        self.bot_frame.grid_columnconfigure((0,3), weight=1)
        self.add_btn = ctk.CTkButton(self.bot_frame, text='+', width=35, command=self.add_new_dialog)
        self.add_btn.grid(row=0, column=1, sticky='e', padx=(0,5))
        self.collect_data_btn = ctk.CTkButton(self.bot_frame, text='Collect Data', width=75, command=self.collect_data)
        self.collect_data_btn.grid(row=0, column=2, sticky='w')
        self.train_btn = ctk.CTkButton(self, text="TRAIN", command=self.train)
        self.train_btn.grid(column=1, pady=4)
        self.body_seq_init()
        self.create_temp_folders()
    
    def sign_dropdown(self, value):
        self.frame_label.configure(text=self.sign_menu_var.get())
        self.word_var.set(0)
        if value == 'Body Gesture':
            self.body_seq_init()
        elif value == 'Hand Pose':
            self.hand_pose_init()
        elif value == 'Facial Expression':
            self.face_expre_init()
    
    def create_temp_folders(self):
        for i in os.listdir('data'):
            if not os.path.exists(os.path.join('temp', i)):
                os.makedirs(os.path.join('temp', i))
            
    def body_seq_init(self):
        for i in self.gui_scrll_frame.winfo_children():
            for c in i.winfo_children():
                c.destroy()
            i.destroy()
        for i, item in enumerate(self.body_seq_labels):
            btn_body_seq_label = ctk.CTkRadioButton(self.gui_scrll_frame, text=item, variable=self.word_var, value=i, width=120)
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
            btn_hand_pose_label = ctk.CTkRadioButton(self.gui_scrll_frame, text=item, variable=self.word_var, value=i, width=120)
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
            btn_face_expre_label = ctk.CTkRadioButton(self.gui_scrll_frame, text=item, variable=self.word_var, value=i, width=120)
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
        if os.path.exists('temp/hand/hand_pose_data.csv'):
            tmp_hp = np.loadtxt('temp/hand/hand_pose_data.csv', delimiter=',', usecols=list(range((21 * 2) + 1)), dtype='object')
            tmp_hp[:, 0] = tmp_hp[:, 0].astype(int)
            tmp_hp[:, 1:] = tmp_hp[:, 1:].astype(float)
        else:
            tmp_hp = np.loadtxt('data/hand/hand_pose_data.csv', delimiter=',', usecols=list(range((21 * 2) + 1)), dtype='object')
            tmp_hp[:, 0] = tmp_hp[:, 0].astype(int)
            tmp_hp[:, 1:] = tmp_hp[:, 1:].astype(float)
        if not x == len(self.hand_pose_labels) - 1:
            tmp_hp = tmp_hp[(tmp_hp[:, 0] != x)]
            tmp_hp_copy = tmp_hp.copy()
            np.subtract(1, tmp_hp[:, 0], out=tmp_hp[:, 0], where=tmp_hp_copy[:, 0] > x)
            tmp_hp[:, 0] = abs(tmp_hp[:, 0])
            # np.save("temp/hand/hand_pose_data.csv", tmp_hp)
            np.savetxt("temp/hand/hand_pose_data.csv",tmp_hp,delimiter=',', fmt='%s')
        else:
            tmp_hp = tmp_hp[(tmp_hp[:, 0] != x)]
            tmp_hp[:, 0] = abs(tmp_hp[:, 0])
            # np.save("temp/hand/hand_pose_data.csv", tmp_hp)
            np.savetxt("temp/hand/hand_pose_data.csv",tmp_hp,delimiter=',', fmt='%s')
        self.hand_pose_labels.pop(x)
        with open('temp/hand/hand_pose_labels.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i in self.hand_pose_labels:
                writer.writerow([i])
        self.hand_pose_init()

    def face_expre_delete(self, x):
        if len(self.face_expre_labels) == 1:
            return
        if os.path.exists('temp/face/face_expre_data.csv'):
            tmp_fe = np.loadtxt('temp/face/face_expre_data.csv', delimiter=',', usecols=list(range((478 * 2) + 1)), dtype='object')
            tmp_fe[:, 0] = tmp_fe[:, 0].astype(int)
            tmp_fe[:, 1:] = tmp_fe[:, 1:].astype(float)
        else:
            tmp_fe = np.loadtxt('data/face/face_expre_data.csv', delimiter=',', usecols=list(range((478 * 2) + 1)), dtype='object')
            tmp_fe[:, 0] = tmp_fe[:, 0].astype(int)
            tmp_fe[:, 1:] = tmp_fe[:, 1:].astype(float)
        if not x == len(self.face_expre_labels) - 1:
            tmp_fe = tmp_fe[(tmp_fe[:, 0] != x)]
            tmp_fe_copy = tmp_fe.copy()
            np.subtract(1, tmp_fe[:, 0], out=tmp_fe[:, 0], where=tmp_fe_copy[:, 0] > x)
            tmp_fe[:, 0] = abs(tmp_fe[:, 0])
            # np.save("temp/face/face_expre_data.npy", tmp_fe)
            np.savetxt("temp/face/face_expre_data.csv",tmp_fe,delimiter=',', fmt=('%s, %f'))
        else:
            tmp_fe = tmp_fe[(tmp_fe[:, 0] != x)]
            tmp_fe[:, 0] = abs(tmp_fe[:, 0])
            # np.save("temp/face/face_expre_data.npy", tmp_fe)
            np.savetxt("temp/face/face_expre_data.csv",tmp_fe,delimiter=',', fmt=('%s, %f'))
        self.face_expre_labels.pop(x)
        with open('temp/face/face_expre_labels.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i in self.face_expre_labels:
                writer.writerow([i])
        self.face_expre_init()
    
    def train(self):
        for i in self.gui_scrll_frame.winfo_children():
            i.configure(state='disabled')
        for j in self.bot_frame.winfo_children():
            j.configure(state='disabled')
        self.sign_menu.configure(state='disabled')
        self.train_btn.configure(state='disabled')
    
    def add_new_dialog(self):
        frame_add_new = ctk.CTkToplevel(self)
        frame_add_new.attributes('-topmost',True)
        frame_add_new.geometry(f"{200//2}x{170//2}+{self.screen_width//2}+{self.screen_height//1.9}")
        frame_add_new.grid_propagate(False)
        frame_add_new.grid_columnconfigure((0,2), weight=1)
        frame_add_new.grab_set()
        add_new_label = ctk.CTkLabel(frame_add_new, text='Enter a new word to add:', fg_color='transparent')
        add_new_label.grid(column=1, pady=1)
        new_word_entry = ctk.CTkEntry(frame_add_new)
        new_word_entry.grid(column=1)
        bot_frame = ctk.CTkFrame(frame_add_new, fg_color='transparent')
        bot_frame.grid(column=1, columnspan=2, sticky='nsew')
        bot_frame.grid_columnconfigure((0,3), weight=1)
        add_btn = ctk.CTkButton(bot_frame, text='Add', width=35)
        add_btn.grid(row=0, column=1, sticky='e', padx=(0,5))
        cancel_btn = ctk.CTkButton(bot_frame, text='Cancel', width=75)
        cancel_btn.grid(row=0, column=2, sticky='w')
    
    def add_word(self, word):
        pass

    
    def collect_data(self):
        if self.collect_data_btn.cget('text') == 'Collect Data':
            self.sign_menu.configure(state='disabled')
            self.add_btn.configure(state='disabled')
            for i in self.gui_scrll_frame.winfo_children():
                i.configure(state='disabled')
            self.train_btn.configure(state='disabled')
            self.collect_data_btn.configure(fg_color='red3', text='STOP')
        elif self.collect_data_btn.cget('text') == 'STOP':
            self.sign_menu.configure(state='normal')
            self.add_btn.configure(state='normal')
            for i in self.gui_scrll_frame.winfo_children():
                i.configure(state='normal')
            self.train_btn.configure(state='normal')
            self.collect_data_btn.configure(fg_color=self.add_btn.cget('fg_color'), text='Collect Data')
        self.main_app.to_add_data_idx = self.word_var.get()
        if self.sign_menu_var.get() == 'Body Gesture':
            self.main_app.body_seq.collect_data = not self.main_app.body_seq.collect_data
            print(self.main_app.body_seq.collect_data)
        elif self.sign_menu_var.get() == 'Hand Pose':
            self.main_app.hand_pose.collect_data = not self.main_app.hand_pose.collect_data
            print(self.main_app.hand_pose.collect_data)
        elif self.sign_menu_var.get() == 'Facial Expression':
            self.main_app.face_expre.collect_data = not self.main_app.face_expre.collect_data
            print(self.main_app.face_expre.collect_data)
        
        
        