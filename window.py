# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(540, 289)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 50, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.fdEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.fdEdit.setGeometry(QtCore.QRect(140, 60, 141, 20))
        self.fdEdit.setObjectName("fdEdit")
        self.sinParamEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.sinParamEdit.setGeometry(QtCore.QRect(140, 10, 141, 31))
        self.sinParamEdit.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.sinParamEdit.setObjectName("sinParamEdit")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(0, 0, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.tMaxEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.tMaxEdit.setGeometry(QtCore.QRect(140, 110, 141, 20))
        self.tMaxEdit.setObjectName("tMaxEdit")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(0, 100, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(0, 150, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.SnrEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.SnrEdit.setGeometry(QtCore.QRect(140, 160, 141, 20))
        self.SnrEdit.setObjectName("SnrEdit")
        self.winSizeEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.winSizeEdit.setGeometry(QtCore.QRect(140, 200, 141, 20))
        self.winSizeEdit.setObjectName("winSizeEdit")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(0, 190, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(0, 230, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.winStepEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.winStepEdit.setGeometry(QtCore.QRect(140, 240, 141, 20))
        self.winStepEdit.setObjectName("winStepEdit")
        self.generateSignalButton = QtWidgets.QPushButton(self.centralwidget)
        self.generateSignalButton.setGeometry(QtCore.QRect(290, 10, 151, 31))
        self.generateSignalButton.setObjectName("generateSignalButton")
        self.generateSpectrogramButton = QtWidgets.QPushButton(self.centralwidget)
        self.generateSpectrogramButton.setGeometry(QtCore.QRect(290, 50, 151, 31))
        self.generateSpectrogramButton.setObjectName("generateSpectrogramButton")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(290, 150, 221, 111))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.FFTradioButton = QtWidgets.QRadioButton(self.groupBox)
        self.FFTradioButton.setGeometry(QtCore.QRect(10, 20, 51, 17))
        self.FFTradioButton.setObjectName("FFTradioButton")
        self.AKFradioButton = QtWidgets.QRadioButton(self.groupBox)
        self.AKFradioButton.setGeometry(QtCore.QRect(10, 40, 51, 17))
        self.AKFradioButton.setObjectName("AKFradioButton")
        self.MMDradioButton = QtWidgets.QRadioButton(self.groupBox)
        self.MMDradioButton.setGeometry(QtCore.QRect(10, 80, 61, 17))
        self.MMDradioButton.setObjectName("MMDradioButton")
        self.ARradioButton = QtWidgets.QRadioButton(self.groupBox)
        self.ARradioButton.setGeometry(QtCore.QRect(10, 60, 41, 17))
        self.ARradioButton.setObjectName("ARradioButton")
        self.generateRhoButton = QtWidgets.QPushButton(self.centralwidget)
        self.generateRhoButton.setGeometry(QtCore.QRect(290, 90, 151, 51))
        self.generateRhoButton.setObjectName("generateRhoButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Нелинейный спектральный анализ"))
        self.label.setText(_translate("MainWindow", "Частота\n"
"дискретизации"))
        self.label_2.setText(_translate("MainWindow", "Параметры\n"
"синусоид"))
        self.label_3.setText(_translate("MainWindow", "Максимальное\n"
"время"))
        self.label_4.setText(_translate("MainWindow", "SNR"))
        self.label_5.setText(_translate("MainWindow", "Размер окна"))
        self.label_6.setText(_translate("MainWindow", "Шаг окна"))
        self.generateSignalButton.setText(_translate("MainWindow", "Сгенерировать сигнал"))
        self.generateSpectrogramButton.setText(_translate("MainWindow", "Построить спектрограмму"))
        self.groupBox.setTitle(_translate("MainWindow", "Способ построения спектра"))
        self.FFTradioButton.setText(_translate("MainWindow", "FFT"))
        self.AKFradioButton.setText(_translate("MainWindow", "AKF"))
        self.MMDradioButton.setText(_translate("MainWindow", "MMD"))
        self.ARradioButton.setText(_translate("MainWindow", "AR"))
        self.generateRhoButton.setText(_translate("MainWindow", "Построить вероятность\n"
"обнаружения"))
