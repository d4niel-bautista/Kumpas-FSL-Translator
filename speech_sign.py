import time
from QPanda3D.Panda3DWorld import Panda3DWorld
from QPanda3D.QPanda3DWidget import QPanda3DWidget
import speech_recognition as sr
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
from panda3d.core import *
from direct.interval.IntervalGlobal import *
from direct.actor.Actor import Actor
import threading

class PandaTest(Panda3DWorld):
    
    anims = {
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
    app = QApplication(sys.argv)
    recorded_text = QLabel()

    def __init__(self, width=1024, height=768):
        Panda3DWorld.__init__(self, width=width, height=height)
        self.cam.setPos(0, -58, 6)
        self.buff.setClearColorActive(True)
        self.buff.setClearColor(VBase4(235, 236, 240, 1))
        self.cam.node().getDisplayRegion(0).setSort(20)
        
        self.character = Actor('Animations/Alin.egg', self.anims)

        self.character.reparentTo(render)
        self.character.setPos(0,0,-30)
        self.character.setScale(33,33,33)

    def remove_word(self, index, word, text):
        first_part = text[:index]
        second_part = text[len(word)+1:]
        print('f', first_part, index)
        print('s', second_part, len(word))
        print('w', first_part + second_part)
        return first_part+second_part

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

        animSequence = Sequence(name='test')
        for word in to_process:
            animSequence.append(self.character.actorInterval(self.character.play(word), playRate=4))
        self.recorded_text.setText(" ".join(to_process).capitalize())
        time.sleep(1)
        animSequence.start()
    
    def start_speech(self):
        start = threading.Thread(target=self.get_speech)
        start.start()

    def get_speech(self):
        r = sr.Recognizer()
        text = ''
        while True:
            try:
                with sr.Microphone() as mic:
                    r.adjust_for_ambient_noise(mic, 0.2)
                    audio = r.record(mic, duration=4)
                    text = r.recognize_google(audio)
                    text.lower()
                    break
            except:
                self.recorded_text.setText("Unrecognized. Please try again.")
                break
        self.process_text(text)

    def main(self):
        
        appw=QMainWindow()
        appw.setGeometry(360, 80, 700, 620)
        main_widget=QWidget()
        main_widget.setLayout(QVBoxLayout())
        pandaWidget = QPanda3DWidget(self)
        btn_widget = QWidget()
        btn_widget.setMaximumHeight(100)
        btn_widget.setLayout(QVBoxLayout())
        main_widget.layout().addWidget(pandaWidget)
        main_widget.layout().addWidget(btn_widget)
        mic_btn = QPushButton()
        mic_btn.setIcon(QIcon('img/mic.png'))
        mic_btn.setIconSize(QSize(32, 32))
        mic_btn.clicked.connect(self.start_speech)
        
        self.recorded_text.setFont(QFont('Calibri', 20))
        self.recorded_text.setMaximumHeight(50)
        self.recorded_text.setMaximumWidth(680)
        btn_widget.layout().addWidget(mic_btn)
        btn_widget.layout().addWidget(self.recorded_text)

        appw.setCentralWidget(main_widget)
        qr = appw.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        appw.move(qr.topLeft())

        appw.show()
        sys.exit(self.app.exec_())

if __name__ == "__main__":    
    z = PandaTest()
    z.main()

        
        
    