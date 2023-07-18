import sys
import os
from rdsurface import *
from PyQt5 import QtWidgets


class MyForm(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.maintfacts)
        self.ui.pushButton_2.clicked.connect(self.plotn)
        self.ui.pushButton_3.clicked.connect(self.gnb)
        self.ui.pushButton_4.clicked.connect(self.bnb)
        self.ui.pushButton_5.clicked.connect(self.rdftrs)

    def maintfacts(self):
        os.system("python mfactors1.py")

    def plotn(self):
        os.system("python plot1.py")

    def gnb(self):
        os.system("python gnb1.py")

    def bnb(self):
        os.system("python bnb1.py")

    def rdftrs(self):
        os.system("python roadftr1.py")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyForm()
    myapp.show()
    sys.exit(app.exec_())
