import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtSql import *

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lane Detection")
        self.setUIComponenets()
        self.show()

    def setUIComponenets(self):
      self.menuBar = QMenuBar(self)

      font = QFont()
      font.setFamily("Arial")
      font.setPointSize(16)

      self.label = QLabel(self)
      self.label.setText("Hello Wooorld")
      self.label.setFont(font)
      self.label.move(100, 100)

      self.showMaximized()
      
      
