# -*- coding: utf-8-*-
"""
Module : buttons_example
Author : Saifeddine ALOUI
Description :
    This is an example of how we can put togather a simple Panda3D Word 
    wrapped inside a QMainWindow and add QT pushbuttons that interact with the world.
"""
import time
from QPanda3D.Panda3DWorld import Panda3DWorld
from QPanda3D.QPanda3DWidget import QPanda3DWidget
import speech_recognition as sr
# import PyQt5 stuff
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
from panda3d.core import *
from direct.interval.LerpInterval import LerpHprInterval
from direct.interval.IntervalGlobal import *
from direct.gui.OnscreenImage import OnscreenImage
from direct.actor.Actor import Actor
from direct.interval.ActorInterval import ActorInterval
from direct.gui.OnscreenText import OnscreenText
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
        self.buff.setClearColor(VBase4(0, 0, 0, 1))
        self.cam.node().getDisplayRegion(0).setSort(20)



   



        #Create a panda
        
        self.panda = Actor('Animations/Alin.egg', self.anims)

        self.panda.reparentTo(render)
        self.panda.setPos(0,0,-30)
        self.panda.setScale(33,33,33)
                  
        # Now create some lights to apply to everything in the scene.
         
        # Create Ambient Light
        # ambientLight = AmbientLight( 'ambientLight' )
        # ambientLight.setColor( Vec4( 0.1, 0.1, 0.1, 1 ) )
        # ambientLightNP = render.attachNewNode( ambientLight )
        # render.setLight(ambientLightNP)
         
        # # Directional light 01
        # directionalLight = DirectionalLight( "directionalLight" )
        # directionalLight.setColor( Vec4( 0.8, 0.1, 0.1, 1 ) )
        # directionalLightNP = render.attachNewNode( directionalLight )
        # # This light is facing backwards, towards the camera.
        # directionalLightNP.setHpr(180, -20, 0)
        # directionalLightNP.setPos(10,-100,10)
        # render.setLight(directionalLightNP)
         
        # # If we did not call setLightOff() first, the green light would add to
        # # the total set of lights on this object.  Since we do call
        # # setLightOff(), we are turning off all the other lights on this
        # # object first, and then turning on only the green light.
        # self.panda.setLightOff()

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
            animSequence.append(self.panda.actorInterval(self.panda.play(word), playRate=4))
        self.recorded_text.setText(" ".join(to_process).capitalize())
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
        # Here, we create the panda3D widget and specify that we want the widget to stretch
        # The keep_ratio parameter tells weather the painter should keep the ratio of the original
        # environment or it can devorm it. If ratio is kept, then the original will be cropped
        main_widget=QWidget()
        main_widget.setLayout(QVBoxLayout())
        # hide_widg = QFrame()
        # hide_widg.setLayout(QVBoxLayout())
        # SPanda3D Widget
        pandaWidget = QPanda3DWidget(self)#, stretch=True, keep_ratio=False)
        # Buttons Widget
        btn_widget = QWidget()
        btn_widget.setMaximumHeight(100)
        btn_widget.setLayout(QVBoxLayout())
        # Add them to the window
        main_widget.layout().addWidget(pandaWidget)
        main_widget.layout().addWidget(btn_widget)
        mic_btn = QPushButton()
        # mic_btn.setMaximumWidth(100)
        # mic_btn.setMaximumHeight(100)
        mic_btn.setIcon(QIcon('mic icon.png'))
        mic_btn.setIconSize(QSize(32, 32))
        mic_btn.clicked.connect(self.start_speech)
        
        self.recorded_text.setFont(QFont('Calibri', 20))
        self.recorded_text.setMaximumHeight(50)
        self.recorded_text.setMaximumWidth(680)
        btn_widget.layout().addWidget(mic_btn)
        btn_widget.layout().addWidget(self.recorded_text)

    

        # Now let's create some buttons


        # btn_alin=QPushButton("Alin")
        # btn_widget.layout().addWidget(btn_alin)
        # btn_alin.clicked.connect(world.Alin)
        
        # btn_attached=QPushButton("Attached")
        # btn_widget.layout().addWidget(btn_attached)
        # btn_attached.clicked.connect(world.Attached)

        # btn_bad=QPushButton("Bad")
        # btn_widget.layout().addWidget(btn_bad)
        # btn_bad.clicked.connect(world.Bad)

        # btn_bakit=QPushButton("Bakit")
        # btn_widget.layout().addWidget(btn_bakit)
        # btn_bakit.clicked.connect(world.Bakit)

        # btn_balanse=QPushButton("Balanse")
        # btn_widget.layout().addWidget(btn_balanse)
        # btn_balanse.clicked.connect(world.Balanse)

        # btn_beautiful=QPushButton("Beautiful")
        # btn_widget.layout().addWidget(btn_beautiful)
        # btn_beautiful.clicked.connect(world.Beautiful)

        # btn_big=QPushButton("Big")
        # btn_widget.layout().addWidget(btn_big)
        # btn_big.clicked.connect(world.Big)

        # btn_boastful=QPushButton("Boastful")
        # btn_widget.layout().addWidget(btn_boastful)
        # btn_boastful.clicked.connect(world.Boastful)

        # btn_bored=QPushButton("Bored")
        # btn_widget.layout().addWidget(btn_bored)
        # btn_bored.clicked.connect(world.Bored)

        # btn_bumili=QPushButton("Bumili")
        # btn_widget.layout().addWidget(btn_bumili)
        # btn_bumili.clicked.connect(world.Bumili)

        # btn_cold=QPushButton("Cold")
        # btn_widget.layout().addWidget(btn_cold)
        # btn_cold.clicked.connect(world.Cold)

        # btn_day=QPushButton("Day")
        # btn_widget.layout().addWidget(btn_day)
        # btn_day.clicked.connect(world.Day)

        # btn_debt=QPushButton("Debt")
        # btn_widget.layout().addWidget(btn_debt)
        # btn_debt.clicked.connect(world.Debt)

        # btn_difficult=QPushButton("Difficult")
        # btn_widget.layout().addWidget(btn_difficult)
        # btn_difficult.clicked.connect(world.Difficult)

        # btn_dolyar=QPushButton("Dolyar")
        # btn_widget.layout().addWidget(btn_dolyar)
        # btn_dolyar.clicked.connect(world.Dolyar)

        # btn_dumb=QPushButton("Dumb")
        # btn_widget.layout().addWidget(btn_dumb)
        # btn_dumb.clicked.connect(world.Dumb)

        # btn_easy=QPushButton("Easy")
        # btn_widget.layout().addWidget(btn_easy)
        # btn_easy.clicked.connect(world.Easy)

        # btn_fast=QPushButton("Fast")
        # btn_widget.layout().addWidget(btn_fast)
        # btn_fast.clicked.connect(world.Fast)

        # btn_howAreYou=QPushButton("How Are You")
        # btn_widget.layout().addWidget(btn_howAreYou)
        # btn_howAreYou.clicked.connect(world.HowAreYou)


        # btn_hug=QPushButton("Hug")
        # btn_widget.layout().addWidget(btn_hug)
        # btn_hug.clicked.connect(world.Hug)





        # enterText = QLineEdit()
        # # enterText.resize(720, 100)
        # btn_widget.layout().addWidget(enterText)
        # goTsl=QPushButton("Translate")
        # btn_widget.layout().addWidget(goTsl)
        # goTsl.clicked.connect(lambda x: world.process_text(enterText.text()))


        # bla = QFrame()
        # bla.setLayout(QVBoxLayout())
        # show_bt=QPushButton("show")
        # bla.layout().addWidget(show_bt)
        # show_bt.clicked.connect(lambda x: show(hide_widg))

        # hide_btn=QPushButton("hide")
        # bla.layout().addWidget(hide_btn)
        # hide_btn.clicked.connect(lambda x: hide(hide_widg))

        # bla.hide()
        appw.setCentralWidget(main_widget)
        #hide_widg.layout().addWidget(main_widget)
        # hide_widg.layout().addWidget(bla)

        appw.show()
        sys.exit(self.app.exec_())
    # def Alin(self):
    #     self.panda.play('Alin')

    # def Attached(self):
    #     self.panda.play('Attached')
    
    # def Bad(self):
    #     self.panda.play('Bad')

    # def Bakit(self):
    #     self.panda.play('Bakit')

    # def Balanse(self):
    #     self.panda.play('Balanse')

    # def Beautiful(self):
    #     self.panda.play('Beautiful')

    # def Big(self):
    #     self.panda.play('Big')

    # def Boastful(self):
    #     self.panda.play('Boastful')

    # def Bored(self):
    #     self.panda.play('Bored')

    # def Bumili(self):
    #     self.panda.play('Bumili')

    # def Cold(self):
    #     self.panda.play('Cold')

    # def Day(self):
    #     self.panda.play('Day')

    # def Debt(self):
    #     self.panda.play('Debt')

    # def Difficult(self):
    #     self.panda.play('Difficult')

    # def Dolyar(self):
    #     self.panda.play('Dolyar')

    # def Dumb(self):
    #     self.panda.play('Dumb')

    # def Easy(self):
    #     self.panda.play('Easy')

    # def Fast(self):
    #     self.panda.play('Fast')

    # def HowAreYou(self):
    #     self.panda.play('HowAreYou')

    # def Hug(self):
    #     self.panda.play('Hug')


if __name__ == "__main__":    
    z = PandaTest()
    z.main()

        
        
    