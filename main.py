# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.2

import cv2
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
#from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QMessageBox


os.chdir(os.path.dirname(os.path.abspath(__file__)))
cwd = os.getcwd()


class Ui_MainWindow(object):
    ARTISTS = ['Adele', 'Angelina Jolie', 'Arnold Schwarzenegger', 'Bon Jovi', 'Brad Pitt', 'Chuck Norris', 'Conor Mcgregor',
          'Cristiano Ronaldo', 'Ed Sheeran', 'Eddie Murphy']
    face_cascade = cv2.CascadeClassifier(os.path.join(cwd, 'haarcascade_frontalface_default.xml'))
    face_image = None
    IMAGE_SIZE = 160
    X = []
    y = []
    model = None
    

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(1274, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.photo = QtWidgets.QLabel(self.centralwidget)
        self.photo.setGeometry(QtCore.QRect(0, 0, 961, 671))
        self.photo.setText("")
        self.photo.setScaledContents(False)
        self.photo.setAlignment(QtCore.Qt.AlignCenter)
        self.photo.setObjectName("photo")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(945, -10, 41, 681))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(970, 0, 307, 676))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.labelSettings = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelSettings.sizePolicy().hasHeightForWidth())
        self.labelSettings.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.labelSettings.setFont(font)
        self.labelSettings.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.labelSettings.setObjectName("labelSettings")
        self.verticalLayout.addWidget(self.labelSettings)
        self.groupBoxImage = QtWidgets.QGroupBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBoxImage.sizePolicy().hasHeightForWidth())
        self.groupBoxImage.setSizePolicy(sizePolicy)
        self.groupBoxImage.setMinimumSize(QtCore.QSize(0, 100))
        self.groupBoxImage.setBaseSize(QtCore.QSize(0, 0))
        self.groupBoxImage.setObjectName("groupBoxImage")
        self.gridLayoutWidget = QtWidgets.QWidget(self.groupBoxImage)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(10, 10, 291, 81))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setHorizontalSpacing(10)
        self.gridLayout.setVerticalSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.labelPath = QtWidgets.QLabel(self.gridLayoutWidget)
        self.labelPath.setObjectName("labelPath")
        self.gridLayout.addWidget(self.labelPath, 0, 0, 1, 1)
        self.buttonLoadPhoto = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.buttonLoadPhoto.setObjectName("buttonLoadPhoto")
        self.buttonLoadPhoto.clicked.connect(self.clickedLoadPhoto)
        self.gridLayout.addWidget(self.buttonLoadPhoto, 1, 1, 1, 1)
        self.pathLine = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.pathLine.setObjectName("pathLine")
        self.gridLayout.addWidget(self.pathLine, 0, 1, 1, 1)
        self.verticalLayout.addWidget(self.groupBoxImage)
        self.groupBoxCNN = QtWidgets.QGroupBox(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBoxCNN.sizePolicy().hasHeightForWidth())
        self.groupBoxCNN.setSizePolicy(sizePolicy)
        self.groupBoxCNN.setMinimumSize(QtCore.QSize(0, 300))
        self.groupBoxCNN.setObjectName("groupBoxCNN")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.groupBoxCNN)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(10, 20, 291, 171))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.neuronsDense = QtWidgets.QComboBox(self.gridLayoutWidget_2)
        self.neuronsDense.setObjectName("neuronsDense")
        self.neuronsDense.addItem("")
        self.neuronsDense.addItem("")
        self.neuronsDense.addItem("")
        self.neuronsDense.addItem("")
        self.gridLayout_2.addWidget(self.neuronsDense, 4, 1, 1, 1)
        self.labelDense = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.labelDense.setObjectName("labelDense")
        self.gridLayout_2.addWidget(self.labelDense, 4, 0, 1, 1)
        self.filtersConv2D2 = QtWidgets.QComboBox(self.gridLayoutWidget_2)
        self.filtersConv2D2.setObjectName("filtersConv2D2")
        self.filtersConv2D2.addItem("")
        self.filtersConv2D2.addItem("")
        self.filtersConv2D2.addItem("")
        self.filtersConv2D2.addItem("")
        self.gridLayout_2.addWidget(self.filtersConv2D2, 2, 1, 1, 1)
        self.labelConv2D1 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.labelConv2D1.setObjectName("labelConv2D1")
        self.gridLayout_2.addWidget(self.labelConv2D1, 1, 0, 1, 1)
        self.labelLayer = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.labelLayer.setObjectName("labelLayer")
        self.gridLayout_2.addWidget(self.labelLayer, 0, 0, 1, 1)
        self.kernelConv2D2 = QtWidgets.QComboBox(self.gridLayoutWidget_2)
        self.kernelConv2D2.setObjectName("kernelConv2D2")
        self.kernelConv2D2.addItem("")
        self.kernelConv2D2.addItem("")
        self.gridLayout_2.addWidget(self.kernelConv2D2, 2, 2, 1, 1)
        self.labelFilters = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.labelFilters.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelFilters.sizePolicy().hasHeightForWidth())
        self.labelFilters.setSizePolicy(sizePolicy)
        self.labelFilters.setMinimumSize(QtCore.QSize(0, 0))
        self.labelFilters.setObjectName("labelFilters")
        self.gridLayout_2.addWidget(self.labelFilters, 0, 1, 1, 1)
        self.kernelConv2D1 = QtWidgets.QComboBox(self.gridLayoutWidget_2)
        self.kernelConv2D1.setObjectName("kernelConv2D1")
        self.kernelConv2D1.addItem("")
        self.kernelConv2D1.addItem("")
        self.kernelConv2D1.addItem("")
        self.gridLayout_2.addWidget(self.kernelConv2D1, 1, 2, 1, 1)
        self.labelConv2D2 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.labelConv2D2.setObjectName("labelConv2D2")
        self.gridLayout_2.addWidget(self.labelConv2D2, 2, 0, 1, 1)
        self.labelKernelSize = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.labelKernelSize.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelKernelSize.sizePolicy().hasHeightForWidth())
        self.labelKernelSize.setSizePolicy(sizePolicy)
        self.labelKernelSize.setMinimumSize(QtCore.QSize(0, 0))
        self.labelKernelSize.setObjectName("labelKernelSize")
        self.gridLayout_2.addWidget(self.labelKernelSize, 0, 2, 1, 1)
        self.kernelConv2D3 = QtWidgets.QComboBox(self.gridLayoutWidget_2)
        self.kernelConv2D3.setObjectName("kernelConv2D3")
        self.kernelConv2D3.addItem("")
        self.kernelConv2D3.addItem("")
        self.gridLayout_2.addWidget(self.kernelConv2D3, 3, 2, 1, 1)
        self.labelConv2D3 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.labelConv2D3.setObjectName("labelConv2D3")
        self.gridLayout_2.addWidget(self.labelConv2D3, 3, 0, 1, 1)
        self.filtersConv2D3 = QtWidgets.QComboBox(self.gridLayoutWidget_2)
        self.filtersConv2D3.setObjectName("filtersConv2D3")
        self.filtersConv2D3.addItem("")
        self.filtersConv2D3.addItem("")
        self.filtersConv2D3.addItem("")
        self.gridLayout_2.addWidget(self.filtersConv2D3, 3, 1, 1, 1)
        self.filtersConv2D1 = QtWidgets.QComboBox(self.gridLayoutWidget_2)
        self.filtersConv2D1.setObjectName("filtersConv2D1")
        self.filtersConv2D1.addItem("")
        self.filtersConv2D1.addItem("")
        self.filtersConv2D1.addItem("")
        self.gridLayout_2.addWidget(self.filtersConv2D1, 1, 1, 1, 1)
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.groupBoxCNN)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(10, 200, 291, 80))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.labelBatchSize = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.labelBatchSize.setObjectName("labelBatchSize")
        self.gridLayout_3.addWidget(self.labelBatchSize, 0, 4, 1, 1)
        self.labelEpochs = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.labelEpochs.setObjectName("labelEpochs")
        self.gridLayout_3.addWidget(self.labelEpochs, 0, 3, 1, 1)
        self.labelValidation = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.labelValidation.setObjectName("labelValidation")
        self.gridLayout_3.addWidget(self.labelValidation, 0, 5, 1, 1)
        self.epochs = QtWidgets.QSpinBox(self.gridLayoutWidget_3)
        self.epochs.setMinimum(5)
        self.epochs.setMaximum(30)
        self.epochs.setObjectName("epochs")
        self.gridLayout_3.addWidget(self.epochs, 1, 3, 1, 1)
        self.batchSize = QtWidgets.QComboBox(self.gridLayoutWidget_3)
        self.batchSize.setObjectName("batchSize")
        self.batchSize.addItem("")
        self.batchSize.addItem("")
        self.batchSize.addItem("")
        self.gridLayout_3.addWidget(self.batchSize, 1, 4, 1, 1)
        self.validationSplit = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_3)
        self.validationSplit.setMinimum(0.05)
        self.validationSplit.setMaximum(0.3)
        self.validationSplit.setSingleStep(0.05)
        self.validationSplit.setObjectName("validationSplit")
        self.gridLayout_3.addWidget(self.validationSplit, 1, 5, 1, 1)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.groupBoxCNN)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 280, 291, 51))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.buttonTrain = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.buttonTrain.setObjectName("buttonTrain")
        self.horizontalLayout.addWidget(self.buttonTrain)
        self.buttonTrain.clicked.connect(self.clickedTrainModel)
        self.buttonLoadModel = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.buttonLoadModel.setObjectName("buttonLoadModel")
        self.buttonLoadModel.clicked.connect(self.clickedLoadModel)
        self.horizontalLayout.addWidget(self.buttonLoadModel)
        self.gridLayoutWidget_4 = QtWidgets.QWidget(self.groupBoxCNN)
        self.gridLayoutWidget_4.setGeometry(QtCore.QRect(10, 340, 291, 118))
        self.gridLayoutWidget_4.setObjectName("gridLayoutWidget_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.gridLayoutWidget_4)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_2.setObjectName("label_2")
        self.gridLayout_4.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_3.setObjectName("label_3")
        self.gridLayout_4.addWidget(self.label_3, 2, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.gridLayoutWidget_4)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout_4.addWidget(self.lineEdit, 1, 1, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.gridLayoutWidget_4)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout_4.addWidget(self.lineEdit_2, 2, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 0, 1, 1, 1)
        self.verticalLayout.addWidget(self.groupBoxCNN)
        self.pushButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setMinimumSize(QtCore.QSize(0, 40))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.clickedRunFaceRecognition)
        self.verticalLayout.addWidget(self.pushButton)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1274, 21))
        self.menubar.setObjectName("menubar")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionParameters = QtWidgets.QAction(MainWindow)
        self.actionParameters.setObjectName("actionParameters")
        self.actionTutorial = QtWidgets.QAction(MainWindow)
        self.actionTutorial.setObjectName("actionTutorial")
        self.menuHelp.addAction(self.actionParameters)
        self.menuHelp.addAction(self.actionTutorial)
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Facial recognition system"))
        self.labelSettings.setText(_translate("MainWindow", "Settings"))
        self.groupBoxImage.setTitle(_translate("MainWindow", "Load image"))
        self.labelPath.setText(_translate("MainWindow", "Path"))
        self.buttonLoadPhoto.setText(_translate("MainWindow", "Load"))
        self.groupBoxCNN.setTitle(_translate("MainWindow", "Convolutional neural network architecture "))
        self.neuronsDense.setItemText(0, _translate("MainWindow", "256"))
        self.neuronsDense.setItemText(1, _translate("MainWindow", "512"))
        self.neuronsDense.setItemText(2, _translate("MainWindow", "1024"))
        self.neuronsDense.setItemText(3, _translate("MainWindow", "2048"))
        self.labelDense.setText(_translate("MainWindow", "Dense"))
        self.filtersConv2D2.setItemText(0, _translate("MainWindow", "32"))
        self.filtersConv2D2.setItemText(1, _translate("MainWindow", "64"))
        self.filtersConv2D2.setItemText(2, _translate("MainWindow", "128"))
        self.filtersConv2D2.setItemText(3, _translate("MainWindow", "256"))
        self.labelConv2D1.setText(_translate("MainWindow", "Conv2D"))
        self.labelLayer.setText(_translate("MainWindow", "Layer"))
        self.kernelConv2D2.setItemText(0, _translate("MainWindow", "3"))
        self.kernelConv2D2.setItemText(1, _translate("MainWindow", "5"))
        self.labelFilters.setText(_translate("MainWindow", "Filters | Neurons"))
        self.kernelConv2D1.setItemText(0, _translate("MainWindow", "3"))
        self.kernelConv2D1.setItemText(1, _translate("MainWindow", "5"))
        self.kernelConv2D1.setItemText(2, _translate("MainWindow", "7"))
        self.labelConv2D2.setText(_translate("MainWindow", "Conv2D"))
        self.labelKernelSize.setText(_translate("MainWindow", "Kernel size"))
        self.kernelConv2D3.setItemText(0, _translate("MainWindow", "3"))
        self.kernelConv2D3.setItemText(1, _translate("MainWindow", "5"))
        self.labelConv2D3.setText(_translate("MainWindow", "Conv2D"))
        self.filtersConv2D3.setCurrentText(_translate("MainWindow", "64"))
        self.filtersConv2D3.setItemText(0, _translate("MainWindow", "64"))
        self.filtersConv2D3.setItemText(1, _translate("MainWindow", "128"))
        self.filtersConv2D3.setItemText(2, _translate("MainWindow", "256"))
        self.filtersConv2D1.setItemText(0, _translate("MainWindow", "32"))
        self.filtersConv2D1.setItemText(1, _translate("MainWindow", "64"))
        self.filtersConv2D1.setItemText(2, _translate("MainWindow", "128"))
        self.labelBatchSize.setText(_translate("MainWindow", "Batch size"))
        self.labelEpochs.setText(_translate("MainWindow", "Epochs"))
        self.labelValidation.setText(_translate("MainWindow", "Validation split"))
        self.batchSize.setItemText(0, _translate("MainWindow", "8"))
        self.batchSize.setItemText(1, _translate("MainWindow", "16"))
        self.batchSize.setItemText(2, _translate("MainWindow", "32"))
        self.buttonTrain.setText(_translate("MainWindow", "Train model"))
        self.buttonLoadModel.setText(_translate("MainWindow", "Load pre-trained model"))
        self.label_2.setText(_translate("MainWindow", "Name"))
        self.label_3.setText(_translate("MainWindow", "Confidence"))
        self.label.setText(_translate("MainWindow", "System output"))
        self.pushButton.setText(_translate("MainWindow", "Run Face Recognition"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionParameters.setText(_translate("MainWindow", "Parameters"))
        self.actionParameters.setStatusTip(_translate("MainWindow", "Parameters description"))
        self.actionParameters.setShortcut(_translate("MainWindow", "F5"))
        self.actionParameters.triggered.connect(self.parametersMenubar)
        self.actionTutorial.setText(_translate("MainWindow", "Tutorial"))
        self.actionTutorial.setStatusTip(_translate("MainWindow", "Application manual"))
        self.actionTutorial.setShortcut(_translate("MainWindow", "F6"))
        self.actionTutorial.triggered.connect(self.tutorialMenubar)
        

    def process_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        detected_faces = self.face_cascade.detectMultiScale(img, 1.3, 5)
        X_test = []
        try:
            (x,y,w,h) = detected_faces[0]
            roi = img[y:y+h, x:x+w]
            new_array = cv2.resize(roi, (self.IMAGE_SIZE, self.IMAGE_SIZE))
            X_test.append(new_array)
            X_test = np.array(X_test).reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 3)
            X_test = X_test/255.0
            return X_test
        except:
            self.faceNotDetectedPopup()
            return None
        
#################### Button Events ####################

    def clickedLoadPhoto(self):
        path = self.pathLine.text()
        image = QtGui.QPixmap(path)
        # tu ryj
        scaledImage = image.scaled(self.photo.size(), QtCore.Qt.KeepAspectRatio)
        self.photo.setPixmap(scaledImage)
        self.pathLine.clear()
        if image.isNull():
            self.loadErrorPopup()
        else:
            self.face_image = self.process_image(path)
            #self.drawFaceRect()


    def importTrainData(self): 
        with open(os.path.join(cwd, 'faces_x_pickle.pkl'), 'rb') as pickle_file:
            self.X = pickle.load(pickle_file) 
        with open(os.path.join(cwd, 'faces_y_pickle.pkl'), 'rb') as pickle_file:
            self.y = pickle.load(pickle_file)


    def clickedTrainModel(self):
        conv_filters = []
        conv_filters.append([int(self.filtersConv2D1.currentText()),  int(self.kernelConv2D1.currentText())])
        conv_filters.append([int(self.filtersConv2D2.currentText()),  int(self.kernelConv2D2.currentText())])
        conv_filters.append([int(self.filtersConv2D3.currentText()),  int(self.kernelConv2D3.currentText())])

        neuronsDense = int(self.neuronsDense.currentText())
        epochs = self.epochs.value()
        batch_size = int(self.batchSize.currentText())
        validation_split = self.validationSplit.value()
     
        model = Sequential([
            Conv2D(conv_filters[0][0], kernel_size=conv_filters[0][1], activation='relu', input_shape=(160,160,3)),
            MaxPooling2D(2,2),
            
            Conv2D(conv_filters[1][0], kernel_size=conv_filters[1][1], activation='relu'),
            MaxPooling2D(2,2),
            
            Conv2D(conv_filters[2][0], kernel_size=conv_filters[2][1], activation='relu'), 
            MaxPooling2D(2,2),
            
            Flatten(),
            Dense(neuronsDense, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy', 
                    optimizer='adam',
                    metrics=['accuracy'])
        
        self.importTrainData()
        self.trainingPopup()
        history = model.fit(self.X, self.y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
        self.trainingDonePopup(history)

    
    def clickedLoadModel(self):
        self.model = tf.keras.models.load_model('saved_model/')


    def clickedRunFaceRecognition(self):
        if (self.face_image is not None) and (self.model is not None):
            prediction = self.model.predict(self.face_image)
            confidence = max(prediction[0])*100
            print(confidence)
            if confidence < 50:
                self.lineEdit.setText('Unknown')
                self.lineEdit_2.setText('<50%')
            else:
                confidence = str(confidence)
                confidence = confidence[:5] + '%'
                print(confidence)
                print(prediction)
                artist = self.ARTISTS[np.argmax(prediction)]
                self.lineEdit.setText(artist)
                self.lineEdit_2.setText(confidence)
                self.faceRecognizedPopup(artist)
        else:
            self.runFaceRecErrorPopup()
            

#################### Popup Windows ####################

    def loadErrorPopup(self):
        msg = QMessageBox()
        msg.setWindowTitle('Load image - Error')
        msg.setText("The image was not loaded. Check the file path.")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()        


    def runFaceRecErrorPopup(self):
        msg = QMessageBox()
        msg.setWindowTitle('Run - Error')
        msg.setText("Model or image does not exist.")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_()  

    
    def faceNotDetectedPopup(self):
        msg = QMessageBox()
        msg.setWindowTitle('No face detected- Error')
        msg.setText("Face not detected. Try another image.")
        msg.setIcon(QMessageBox.Critical)
        msg.exec_() 

    def trainingPopup(self):
        msg = QMessageBox()
        msg.setWindowTitle('Training model - Info')
        msg.setText("Training network.\nPlease wait.")
        msg.setIcon(QMessageBox.Information)
        msg.exec_() 

    def trainingDonePopup(self, history):
        msg = QMessageBox()
        msg.setWindowTitle('Training model - Info')

        score = history.history
        val_acc = round(score["val_accuracy"][len(score["val_accuracy"])-1]* 100, 1) 
        acc = round(score["accuracy"][len(score["accuracy"])-1]* 100, 1) 

        msg.setText(f"Done.\nTrain accuracy: {acc}%\nValidation accuracy: {val_acc}%")
        msg.setIcon(QMessageBox.Information)
        msg.exec_() 

    def parametersMenubar(self):
        msg = QMessageBox()
        msg.setWindowTitle('Parameters - Description')
        
        msg.setText("""
        Conv2D - 2D convolution layer (e.g. spatial convolution over images).\n
        Filters - Number of filters to apply on the convolutional window.\n
        Kernel_size - Value specifying the size of the convolution window.\n
        Dense -  Regular densely-connected neural network layer.\n
        Neurons - Number of units in dense layer.\n
        Epochs - Number of epochs during model training.\n
        Batch size -  Number of samples per batch of computation.\n
        Validation split - Fraction of the training data to be used as validation data.\n
        """)

        msg.setInformativeText(""" 
        For more information visit <a href="http://tensorflow.org/">Tensorflow</a>""")
        msg.exec_() 

    def tutorialMenubar(self):
        msg = QMessageBox()
        msg.setWindowTitle('Tutorial')

        msg.setText("""
        1. Load a face image from the Load image submenu. Enter the path to the image
        and click the button.\n
        2. You have two options to choose the model by creating your own colvolutional
        neural network or loading the pre-trained model. Pre-trained model validation 
        accuracy is 89%. To achive better results choose higher filter values and smaller 
        kernel sizes. The best training parameters for the training data would be 15 
        epochs, 16 batch size and 0.1 validation split. If you train your own model,
        it can take some time.\n
        3. Click the Run button and chceck wheter the CNN was right or not. 
        """)
        msg.exec_() 
        
    def faceRecognizedPopup(self, parameter):
        msg = QMessageBox()
        msg.setWindowTitle('Face Recognized - Info')
        msg.setText(f"Face recognized as: {parameter}")
        msg.setIcon(QMessageBox.Information)
        msg.exec_() 


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


