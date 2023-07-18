import sys
import os
from roadftr import *
from PyQt5 import QtWidgets, QtGui, QtCore

import sqlite3
con = sqlite3.connect('rsurface1')

# import MySQLdb as mdb
# con = mdb.connect('localhost', 'team1', 'test623', 'drless1');


class MyForm(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.insertvalues)
        # self.ui.pushButton_2.clicked.connect(self.testdetails)

    def insertvalues(self):
        with con:
            cur = con.cursor()
            id = str(self.ui.lineEdit_9.text())
            location = str(self.ui.lineEdit_3.text())
            s1 = str(self.ui.lineEdit_4.text())
            s2 = str(self.ui.lineEdit_5.text())
            s3 = str(self.ui.lineEdit_6.text())
            s4 = str(self.ui.lineEdit_2.text())
            s5 = str(self.ui.lineEdit_11.text())
            cur.execute('INSERT INTO rftr VALUES(?,?,?,?,?,?,?)', (id, location, s1, s2, s3, s4, s5))
            con.commit()

#  def testdetails(self):
#    os.system("python ptests1.py")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyForm()
    myapp.show()
    sys.exit(app.exec_())
