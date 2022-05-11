#!/usr/bin/env python3

# from CODE.detectYesNo import check_chess
from PIL.Image import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QSizePolicy
from matplotlib import pyplot as plt

import cv2
import numpy as np
import sys
import time
import os
import math
import threading

# import checkAlign
from connectPLC import PLC
from detectYesNo import check_chess
from detectYesNo import Detect
from checkOnJig import CheckOn
import checkAlign

import codecs
import serial
import time
import wheel

arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class Thread(QThread):
    progress = pyqtSignal()

    def run(self):
        while True:
            self.progress.emit()
            time.sleep(0.5)

class Query(QThread):
    progress = pyqtSignal()

    def run(self):
        while True:
            self.progress.emit()
            time.sleep(0.5)

class Camera(QThread):
    setup = pyqtSignal()

    def run(self):
        self.setup.emit()

class App(QMainWindow):
    def __init__(self):
        super().__init__()

        # QT Config
        self.title = "PLC-UET"
        self.icon = QIcon(resource_path('data/icon/uet.png'))

        # Declare Main Variable
        self.total = 0
        self.number_tested = 0
        self.number_success = 0
        self.number_error1 = 0
        self.number_error2 = 0
        self.number_error3 = 0
        self.count = 0
        self.count_error = 0

        #Nháp
        self.result = np.zeros(192, dtype=int)
        self.status = ''

        #Camera
        self.cap_detect = any
        self.cap_check = any
        self.get_cap_detect = False
        self.get_cap_check = False

        #NUC - PLC
        self.Controller = PLC()
        self.command = ''
        self.prev_command = ''
        self.status_cam_checked = ''
        self.status_cam_inJig = ''
        self.jig_signal = False


        self.checkError = " "

        # Run QT
        self.initUI()
    
    def initUI(self):

        # Config Main Window
        self.setWindowTitle(self.title)
        self.setWindowIcon(self.icon)
        self.setWindowState(Qt.WindowFullScreen)
        self.setStyleSheet("background-color: rgb(255, 255, 255);")

        # Config Auto Fit Screen Scale Variables
        self.sg = QDesktopWidget().screenGeometry()
        self.width_rate = self.sg.width() / 1920
        self.height_rate = self.sg.height() / 1080
        self.font_rate = math.sqrt(self.sg.width()*self.sg.width() + self.sg.height()*self.sg.height()) / math.sqrt(1920*1920 + 1080*1080)
        
        # Show MCNEX LOGO
        self.mcnex_logo = QLabel(self)
        self.mcnex_pixmap = QPixmap(resource_path('data/icon/mcnex.png')).scaled(181 * self.width_rate, 141 * self.width_rate, Qt.KeepAspectRatio)
        self.mcnex_logo.setPixmap(self.mcnex_pixmap)
        self.mcnex_logo.setGeometry(50 * self.width_rate, 1 * self.height_rate, 181 * self.width_rate, 141 * self.height_rate)
        
         # Show UET LOGO
        self.uet_logo = QLabel(self)
        self.uet_pixmap = QPixmap(resource_path('data/icon/uet.png')).scaled(111 * self.width_rate, 111 * self.width_rate, Qt.KeepAspectRatio)
        self.uet_logo.setPixmap(self.uet_pixmap)
        self.uet_logo.setGeometry(250 * self.width_rate, 10 * self.height_rate, 111 * self.width_rate, 111 * self.height_rate)

        # Show Title
        self.title_label = QLabel("HỆ THỐNG KIỂM TRA LINH KIỆN", self)
        self.title_label.setGeometry(400 * self.width_rate, 17 * self.height_rate, 1800 * self.width_rate, 95 * self.height_rate)
        font_title = QFont('', int(25 * self.font_rate), QFont.Bold)
        self.title_label.setFont(font_title)
        self.title_label.setStyleSheet("color: rgb(0, 0, 0);")

         # Show Current Time
        self.time_label = QLabel(self)
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setGeometry(1470 * self.width_rate, 20 * self.height_rate, 430 * self.width_rate, 95 * self.height_rate)
        font_timer = QFont('', int(25 * self.font_rate), QFont.Bold)
        self.time_label.setFont(font_timer)
        timer = QTimer(self)
        timer.timeout.connect(self.updateTimer)
        timer.start(1000)
        self.time_label.setStyleSheet("color: rgb(0, 0, 0);")

        # Show Detect Camera
        # self.cam1_name = QLabel("DETECT CAMERA", self)
        # self.cam1_name.setGeometry(55 * self.width_rate, 127 * self.height_rate, 582 * self.width_rate, 60 * self.height_rate)
        # self.cam1_name.setAlignment(Qt.AlignCenter)
        # self.cam1_name.setStyleSheet("background-color: rgb(50, 130, 184);"
        #                             "color: rgb(255, 255, 255);"
        #                             "font: bold 14pt;")
        # self.cam1 = QLabel(self)
        # self.cam1.setGeometry(55 * self.width_rate, 185 * self.height_rate, 582 * self.width_rate, 410 * self.height_rate)
        # self.cam1.setStyleSheet("border-color: rgb(50, 130, 184);"
        #                         "border-width: 5px;"
        #                         "border-style: inset;")
        
        # Show Check Camera
        self.cam2_name = QLabel("CHECK CAMERA", self)
        self.cam2_name.setGeometry(1265 * self.width_rate, 127 * self.height_rate, 610 * self.width_rate, 60 * self.height_rate)
        self.cam2_name.setAlignment(Qt.AlignCenter)
        self.cam2_name.setStyleSheet("background-color: rgb(50, 130, 184);"
                                    "color: rgb(255, 255, 255);"
                                    "font: bold 14pt;")
        self.cam2 = QLabel(self)
        self.cam2.setGeometry(1265 * self.width_rate, 185 * self.height_rate, 610 * self.width_rate, 400 * self.height_rate)
        self.cam2.setStyleSheet("border-color: rgb(50, 130, 184);"
                                "background-color: rgb(220, 220, 220);"
                                "border-width: 5px;"
                                "border-style: inset;")

        # Set Font
        self.font = QFont('', int(13 * self.font_rate), QFont.Bold)

        

        
        # Trays Information
        self.tray = []
        for i in range(2):
            k = 2-i
            tray_name = QLabel("TRAY {}".format(k), self)
            tray_name.setGeometry((55 + 406*(i+1) - 5) * self.width_rate, 606 * self.height_rate, 359 * self.width_rate, 55 * self.height_rate)
            tray_name.setAlignment(Qt.AlignCenter)
            tray_name.setStyleSheet("background-color:rgb(50, 130, 184);"
                                    "color: rgb(255, 255, 255);"
                                    "font: bold 14pt;")
            table_margin = QLabel(self)
            table_margin.setGeometry((55 + 406*(i+1) - 5) * self.width_rate, 661 * self.height_rate, 359 * self.width_rate, 405 * self.height_rate)
            table_margin.setStyleSheet("border-color: rgb(50, 130, 184);"
                                        "border-width: 5px;"
                                        "border-style: inset;")
            table = QTableWidget(8, 6, self)
            table.setGeometry((55 + 406*(i+1)) * self.width_rate, 666 * self.height_rate, int(349 * self.width_rate) + 1, int(396 * self.height_rate) + 0.5)
            table.horizontalHeader().hide()
            table.verticalHeader().hide()
            for j in range(6):
                table.setColumnWidth(j, 58 * self.width_rate)
            for j in range(8):
                table.setRowHeight(j, 49 * self.height_rate)
            table.setFont(self.font)
            table.setStyleSheet("background-color:rgb(220, 220, 220);"
                                "color: rgb(255, 255, 255);")
            self.tray.append(table)

#=====
        tray_name = QLabel("NG {}".format(2), self)
        tray_name.setGeometry((55 + 406*0 - 5) * self.width_rate, 606 * self.height_rate, 359 * self.width_rate, 55 * self.height_rate)
        tray_name.setAlignment(Qt.AlignCenter)
        tray_name.setStyleSheet("background-color:rgb(50, 130, 184);"
                                "color: rgb(255, 255, 255);"
                                "font: bold 14pt;")
        table_margin = QLabel(self)
        table_margin.setGeometry((55 + 406*0 - 5) * self.width_rate, 661 * self.height_rate, 359 * self.width_rate, 405 * self.height_rate)
        table_margin.setStyleSheet("border-color: rgb(50, 130, 184);"
                                    "border-width: 5px;"
                                    "border-style: inset;")
        table = QTableWidget(8, 6, self)
        table.setGeometry((55 + 406*0) * self.width_rate, 666 * self.height_rate, int(349 * self.width_rate) + 1, int(396 * self.height_rate) + 0.5)
        table.horizontalHeader().hide()
        table.verticalHeader().hide()
        for j in range(6):
            table.setColumnWidth(j, 58 * self.width_rate)
        for j in range(8):
            table.setRowHeight(j, 49 * self.height_rate)
        table.setFont(QFont('', int(10 * self.font_rate), QFont.Bold))
        table.setStyleSheet("background-color:rgb(220, 220, 220);"
                            "color: rgb(255, 255, 255);")
        self.tray.append(table)
        
        self.tray2 = []
        for i in range(2):
            tray_name = QLabel("TRAY {}".format(i+3), self)
            tray_name.setGeometry((55 + 406*(i+1)- 5) * self.width_rate, 127 * self.height_rate, 359 * self.width_rate, 55 * self.height_rate)
            tray_name.setAlignment(Qt.AlignCenter)
            tray_name.setStyleSheet("background-color:rgb(50, 130, 184);"
                                    "color: rgb(255, 255, 255);"
                                    "font: bold 14pt;")
            table_margin = QLabel(self)
            table_margin.setGeometry((55 + 406*(i+1) - 5) * self.width_rate, 181 * self.height_rate, 359 * self.width_rate, 405 * self.height_rate)
            table_margin.setStyleSheet("border-color: rgb(50, 130, 184);"
                                        "border-width: 5px;"
                                        "border-style: inset;")
            table = QTableWidget(8, 6, self)
            table.setGeometry((55 + 406*(i+1)) * self.width_rate, 186 * self.height_rate, int(349 * self.width_rate) + 1, int(396 * self.height_rate) + 0.5)
            table.horizontalHeader().hide()
            table.verticalHeader().hide()
            for j in range(6):
                table.setColumnWidth(j, 58 * self.width_rate)
            for j in range(8):
                table.setRowHeight(j, 49 * self.height_rate)
            table.setFont(self.font)
            table.setStyleSheet("background-color:rgb(220, 220, 220);"
                                "color: rgb(255, 255, 255);")
            self.tray2.append(table)

#==================================
        tray_name = QLabel("NG {}".format(1), self)
        tray_name.setGeometry((55 + 406*0 - 5) * self.width_rate, 127 * self.height_rate, 359 * self.width_rate, 55 * self.height_rate)
        tray_name.setAlignment(Qt.AlignCenter)
        tray_name.setStyleSheet("background-color:rgb(50, 130, 184);"
                                "color: rgb(255, 255, 255);"
                                "font: bold 14pt;")
        table_margin = QLabel(self)
        table_margin.setGeometry((55 + 406*0 - 5) * self.width_rate, 181 * self.height_rate, 359 * self.width_rate, 405 * self.height_rate)
        table_margin.setStyleSheet("border-color: rgb(50, 130, 184);"
                                    "border-width: 5px;"
                                    "border-style: inset;")
        table = QTableWidget(8, 6, self)
        table.setGeometry((55 + 406*0) * self.width_rate, 186 * self.height_rate, int(349 * self.width_rate) + 1, int(396 * self.height_rate) + 0.5)
        table.horizontalHeader().hide()
        table.verticalHeader().hide()
        for j in range(6):
            table.setColumnWidth(j, 58 * self.width_rate)
        for j in range(8):
            table.setRowHeight(j, 49 * self.height_rate)
        table.setFont(QFont('', int(10 * self.font_rate), QFont.Bold))
        table.setStyleSheet("background-color:rgb(220, 220, 220);"
                            "color: rgb(255, 255, 255);")
        self.tray2.append(table)

        # Table Info Area        
        self.s_name = QLabel("INFORMATION", self)
        # self.s_name.setGeometry(1450 * self.width_rate, 127 * self.height_rate, 399 * self.width_rate, 60 * self.height_rate)
        self.s_name.setGeometry(1265 * self.width_rate, 605 * self.height_rate, 610 * self.width_rate, 60 * self.height_rate)

        self.s_name.setAlignment(Qt.AlignCenter)
        self.s_name.setStyleSheet("background-color:rgb(50, 130, 184);"
                                    "color: rgb(255, 255, 255);"
                                    "font: bold 14pt;")

        self.statistic_table = QTableWidget(5, 3, self)
        self.statistic_table.setGeometry(1265 * self.width_rate, 663 * self.height_rate, int(610 * self.width_rate) + 1, int(405 * self.height_rate) + 1)
        self.statistic_table.horizontalHeader().hide()
        self.statistic_table.verticalHeader().hide()
        self.statistic_table.setFont(self.font)
        self.statistic_table.setStyleSheet("color: rgb(0, 0, 0);"
                                            "background-color:rgb(220, 220, 220);"
                                            "text-align: center;"
                                            "border-width: 5px;"
                                            "border-style: inset;"
                                            "border-color: rgb(50, 130, 184);")
        for j in range(3):
            self.statistic_table.setColumnWidth(j, 200 * self.width_rate)
        for j in range(5):
            self.statistic_table.setRowHeight(j, 79 * self.height_rate)
        tested_item = QTableWidgetItem("TESTED")
        tested_item.setTextAlignment(Qt.AlignCenter)
        tested_item.setFont(self.font)
        self.statistic_table.setItem(0, 0, tested_item)

        success_item = QTableWidgetItem("SUCCESS")
        success_item.setTextAlignment(Qt.AlignCenter)
        success_item.setFont(self.font)
        self.statistic_table.setItem(1, 0, success_item)

        error1_item = QTableWidgetItem("NEED RETEST")
        error1_item.setTextAlignment(Qt.AlignCenter)
        error1_item.setFont(self.font)
        self.statistic_table.setItem(2, 0, error1_item)

        error2_item = QTableWidgetItem("CONNECTION ERROR")
        error2_item.setTextAlignment(Qt.AlignCenter)
        error2_item.setFont(self.font)
        self.statistic_table.setItem(3, 0, error2_item)

        error3_item = QTableWidgetItem("FAILURE")
        error3_item.setTextAlignment(Qt.AlignCenter)
        error3_item.setFont(self.font)
        self.statistic_table.setItem(4, 0, error3_item)

        # Exit Button
        self.exit_button = QPushButton(self)
        self.exit_pixmap = QPixmap(resource_path('data/icon/close.png')).scaled(100 * self.width_rate, 100 * self.width_rate, Qt.KeepAspectRatio)
        self.exit_icon = QIcon(self.exit_pixmap)
        self.exit_button.setIcon(self.exit_icon)
        self.exit_button.setIconSize(QSize(50, 50))
        self.exit_button.setGeometry(1878 * self.width_rate, -8 * self.height_rate, 50 * self.width_rate, 50 * self.height_rate)
        self.exit_button.setHidden(0)
        self.exit_button.setStyleSheet("border: none")
        self.exit_button.clicked.connect(self.close)

        # Create Thread
        # self.camera_thread = Camera()
        # self.camera_thread.setup.connect(self.setup_camera)
        self.main_thread = Thread()
        self.main_thread.progress.connect(self.main_process)
        # self.plc_thread = Query()
        # self.main_thread.progress.connect(self.get_command)

        # Run Thread
        # self.camera_thread.start()
        # self.main_thread.start()
        # self.plc_thread.start()


    #Hết giao diện ###########################################################################################################


    #Hàm stream CAMERA DETECT lNGên giao diện
    def update_detect_image(self, img):
        rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.cam1.setPixmap(QPixmap.fromImage(convertToQtFormat))
    
    
    
    # Hàm stream CAMERA CHECK lên giao diện
    def update_check_image(self, img):
        image = cv2.resize(img, (810,400))
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.cam2.setPixmap(QPixmap.fromImage(convertToQtFormat))
    
    
    # Hàm cập nhật giờ   
    def updateTimer(self):
        cr_time = QTime.currentTime()
        time = cr_time.toString('hh:mm AP')
        self.time_label.setText(time)
    
    #Hàm set_up camera
    def setup_camera(self):
        self.cap_detect = cv2.VideoCapture(0 ) # Khai báo USB Camera Detect Config
        self.get_cap_detect = True
        self.cap_detect.set(3, 1920)
        self.cap_detect.set(4, 1080)
         # cv2.imshow(self.cap_detect)
        # cv2.waitKey(0)

        self.cap_check = cv2.VideoCapture(1) # Khai báo USB Camera Check Config
        self.get_cap_check = True
        
        self.cap_check.set(3, 1920)
        self.cap_check.set(4, 1080)
        # cv2.imshow(self.cap_check)
        # cv2.waitKey(0)

    # Loop Get Command from PLC
    def get_command(self):
        self.command = self.Controller.queryCommand()
        print(self.command)
        if self.command == "Done_detect":
            self.prev_command = 'Done_detect'
        self.status_cam_checked = self.Controller.status_cam_checked()
        self.status_cam_inJig = self.Controller.status_cam_in_jig()
        self.jig_signal = self.Controller.jig_Signal()
    
    #Khởi tạo bảng giá trị 
    def init_statistic(self):
        print("Dang khoi tao bang gia tri")
        tested = QTableWidgetItem("{}".format(0) + " / {}".format(self.total))
        tested.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(0,1,tested)
        ratio_tested = QTableWidgetItem("{} %".format(0))
        ratio_tested.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(0,2,ratio_tested)

        success = QTableWidgetItem("{}".format(0) + " / {}".format(0))
        success.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(1,1,success)
        ratio_success = QTableWidgetItem("{} %".format(0))
        ratio_success.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(1,2,ratio_success)

        error1 = QTableWidgetItem("{}".format(0) + " / {}".format(0))
        error1.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(2,1,error1)
        ratio_error1 = QTableWidgetItem("{} %".format(0))
        ratio_error1.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(2,2,ratio_error1)

        error2 = QTableWidgetItem("{}".format(0) + " / {}".format(0))
        error2.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(3,1,error2)
        ratio_error2 = QTableWidgetItem("{} %".format(0))
        ratio_error2.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(3,2,ratio_error2)

        error3 = QTableWidgetItem("{}".format(0) + " / {}".format(0))
        error3.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(4,1,error3)
        ratio_error3 = QTableWidgetItem("{} %".format(0))
        ratio_error3.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(4,2,ratio_error3)
    
    #Hàm cập nhật số liệu về tổng số linh kiện có trên tray:
    def update_YesNo_data_to_table(self, data):
        
        # Update Data to Table --> Tính Total

        #Update Data to tray 1, tray 2
        print("dang update YESNO")
        c = 0
        for k in range(2):    
            for j in range(7,-1,-1):
                for i in range(5,-1,-1):
                    self.tray[1-k].setItem(j,i,QTableWidgetItem())
                    if(int(data[c])):
                        self.tray[1-k].item(j,i).setBackground(QColor(102, 102, 255)) #Nếu có linh kiện, đổi màu thành xanh
                        self.total += 1
                    # else:
                    #     self.tray[1-k].item(j,i).setBackground(QColor(255,0, 0)) #Không có linh kiện, đổi màu thành đỏ
                    #     self.total += 1
                    c += 1
        
        #Up Data to Tray 3, tray 4
    
        for k in range(2):    
            for j in range(7,-1,-1):
                for i in range(5, -1, -1):
                    self.tray2[k].setItem(j,i,QTableWidgetItem())
                    if(int(data[c])):
                        self.tray2[k].item(j,i).setBackground(QColor(102, 102, 255)) #Nếu có linh kiện, đổi màu thành xanh
                        

                        self.total += 1
                    # else:
                    #     self.tray2[k].item(j,i).setBackground(QColor(255,0, 0)) #Không có linh kiện, đổi màu thành đỏ
                    #     self.total += 1
                    c += 1
    def reset_data_tray_NG(self):
        
        self.count_error = 0
        for j in range(6):
            for i in range(8):
                self.tray[2].setItem(j,i,QTableWidgetItem())
                self.tray[2].item(j,i).setBackground(QColor(220, 220, 220)) 
        
        for j in range(6):
            for i in range(8):
                self.tray2[2].setItem(j,i,QTableWidgetItem())
                self.tray2[2].item(j,i).setBackground(QColor(220, 220, 220))

    def reset_data_tray_cam_in(self):

        self.count = 0
        for k in range(2):    
            for j in range(7,-1,-1):
                for i in range(5,-1,-1):
                    self.tray[1-k].setItem(j,i,QTableWidgetItem()) 
                    self.tray[1-k].item(j,i).setBackground(QColor(220,220,220))
        for k in range(2):    
            for j in range(7,-1,-1):
                for i in range(5,-1,-1):
                    self.tray2[1-k].setItem(j,i,QTableWidgetItem()) 
                    self.tray2[1-k].item(j,i).setBackground(QColor(220,220,220))     
                    

    #Hàm cập nhật bảng số liệu _ cập nhật Information Table
    def update_data_check(self, data):
        
        print(data)
        self.number_tested += 1

        # Reset giá trị đếm khi kiểm tra hết linh kiện
        if self.count == 192:
            self.count = 0

        if self.count_error == 96:
            self.count_error = 0

            self.reset_data_tray_NG()
    
        # Bỏ qua khi không có linh kiện trong mảng dữ liệu
        
        print("Result = ", self.result)
        print("Count hien tai = ", self.count)
        while int(self.result[self.count]) != 1:
            self.count += 1
        print("count = ", self.count)

        # Cập nhật số liệu Kiểm tra
        tested = QTableWidgetItem("{}".format(self.number_tested) + " / {}".format(self.total))
        tested.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(0,1,tested)
        ratio_tested = QTableWidgetItem("{} %".format(int(self.number_tested / self.total * 100)))
        ratio_tested.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(0,2,ratio_tested)

         # Lấy số liệu linh kiện
        tray_idx = self.count // 48
        row = 7 - self.count % 48 // 6
        col = 5 - self.count % 48 % 6
        # print(tray_idx, row, col)


        #Lay so lieu tren tray NG
        self.count_error = self.number_error1+self.number_error2+self.number_error3
        tray_NG_idx = self.count_error // 48
        row_NG = self.count_error % 48 // 6
        col_NG = self.count_error % 48 % 6
        print("NG: ", self.count_error, tray_NG_idx, row_NG, col_NG)
        

        dispMap =["DF_S" ,"AF_D" ,"FIXD" ,"EMPT" ,"DATA" ,"AWB" ,"CRC"]

        #if tray_idx < 2:
        if data == "OK":
            self.number_success += 1
            
            ok = QTableWidgetItem("OK")
            ok.setTextAlignment(Qt.AlignCenter)
            
            if tray_idx < 2:
                self.tray[1-tray_idx].setItem(row, col, ok)
                self.tray[1-tray_idx].item(row,col).setBackground(QColor(67, 138, 94))
            else:
                self.tray2[tray_idx-2].setItem(row, col, ok)
                self.tray2[tray_idx-2].item(row,col).setBackground(QColor(67, 138, 94))
        else:
            if tray_idx < 2:
                
                self.tray[1-tray_idx].item(row,col).setBackground(QColor(255, 138, 94))
            else:
                
                self.tray2[tray_idx-2].item(row,col).setBackground(QColor(255, 138, 94))
            if tray_NG_idx == 0 :
                if data == "NG":
                    self.number_error1 += 1
                    error1 = QTableWidgetItem("NG")
                    error1.setTextAlignment(Qt.AlignCenter)
                    self.tray2[2].setItem(row_NG, col_NG, error1)
                    self.tray2[2].item(row_NG,col_NG).setBackground(QColor(232, 80, 91))
                    
                elif data == "I2C":
                    self.number_error2 += 1
                    error2= QTableWidgetItem("I2C")
                    error2.setTextAlignment(Qt.AlignCenter)
                    self.tray2[2].setItem(row_NG, col_NG, error2)
                    self.tray2[2].item(row_NG,col_NG).setBackground(QColor(232, 80, 91))

                elif data in dispMap:
                    self.number_error3 +=1
                    error3= QTableWidgetItem(data)
                    error3.setTextAlignment(Qt.AlignCenter)
                    self.tray2[2].setItem(row_NG, col_NG, error3)
                    self.tray2[2].item(row_NG,col_NG).setBackground(QColor(232, 80, 91))
            
            elif tray_NG_idx == 1:
                    if data == "NG":
                        self.number_error1 += 1
                        error1 = QTableWidgetItem("NG")
                        error1.setTextAlignment(Qt.AlignCenter)
                        self.tray[2].setItem(row, col, error1)
                        self.tray[2].item(row,col).setBackground(QColor(232, 80, 91))
                        
                    elif data == "I2C":
                        self.number_error2 += 1
                        error2= QTableWidgetItem("I2C")
                        error2.setTextAlignment(Qt.AlignCenter)
                        self.tray[2].setItem(row_NG, col_NG, error2)
                        self.tray[2].item(row_NG,col_NG).setBackground(QColor(232, 80, 91))

                    elif data in dispMap:
                        self.number_error3 +=1
                        error3= QTableWidgetItem(data)
                        error3.setTextAlignment(Qt.AlignCenter)
                        self.tray[2].setItem(row_NG, col_NG, error3)
                        self.tray[2].item(row_NG,col_NG).setBackground(QColor(232, 80, 91))
        
        
        # Cập nhật số liệu
        success = QTableWidgetItem("{}".format(self.number_success) + " / {}".format(self.number_tested))
        success.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(1,1,success)
        ratio_success = QTableWidgetItem("{} %".format(int(self.number_success / self.number_tested * 100)))
        ratio_success.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(1,2,ratio_success)

        error1 = QTableWidgetItem("{}".format(self.number_error1) + " / {}".format(self.number_tested))
        error1.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(2,1,error1)
        ratio_error1 = QTableWidgetItem("{} %".format(int(self.number_error1 / self.number_tested * 100)))
        ratio_error1.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(2,2,ratio_error1)

        error2 = QTableWidgetItem("{}".format(self.number_error2) + " / {}".format(self.number_tested))
        error2.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(3,1,error2)
        ratio_error2 = QTableWidgetItem("{} %".format(int(self.number_error2 / self.number_tested * 100)))
        ratio_error2.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(3,2,ratio_error2)

        error3 = QTableWidgetItem("{}".format(self.number_error3) + " / {}".format(self.number_tested))
        error3.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(4,1,error3)
        ratio_error3 = QTableWidgetItem("{} %".format(int(self.number_error3 / self.number_tested * 100)))
        ratio_error3.setTextAlignment(Qt.AlignCenter)
        self.statistic_table.setItem(4,2,ratio_error3)
        
        # Linh kiện kiểm tra xong sẽ xóa khỏi mảng dữ liệu
        self.result[self.count] = 0
        self.count += 1
        print("Sau khi da kt count = ", self.count)

    def updateTimer(self):
        cr_time = QTime.currentTime()
        time = cr_time.toString('hh:mm AP')
        self.time_label.setText(time)


    #xử lý ảnh
    
    def main_process(self):

        #Load command
        self.command = self.Controller.queryCommand()
        print(self.command)
        if self.command == "Done_detect":
            self.prev_command = 'Done_detect'
        self.status_cam_checked = self.Controller.status_cam_checked()
        self.status_cam_inJig = self.Controller.status_cam_in_jig()
        self.jig_signal = self.Controller.jig_Signal()

        

        if self.command == 'Detect':
            #Reset Tray

            self.reset_data_tray_cam_in()
            self.reset_data_tray_NG


            if self.get_cap_detect == True:
                ret, image = self.cap_detect.read()
                # image = cv2.resize(image, (int(717 * self.width_rate), int(450 * self.height_rate)), interpolation = cv2.INTER_AREA) # Resize cho Giao diện
                #plt.imshow(image)
                #plt.show()

                #CHECK TRAY
                detect = Detect()
                img = check_chess(image)
                detect.rotated(img)
                detect.image = cv2.imread('rotated_image.jpg', cv2.IMREAD_GRAYSCALE)
                # plt.imshow(detect.image, cmap="gray")
                # plt.show()
                detect.thresh()

                # Detect YES/NO
                self.result = detect.check(detect.crop_tray_1)
                self.result = np.append(self.result, detect.check(detect.crop_tray_2))
                self.result = np.append(self.result, detect.check(detect.crop_tray_3))
                self.result = np.append(self.result, detect.check(detect.crop_tray_4))
                print("KQ detect: ", self.result)
                # plt.imshow(detect.image, cmap="gray")
                # plt.show()
                #up date result to PLC
                self.Controller.data = self.result
                self.Controller.sendData()
            
                self.command = 'Done_detect'
                self.Controller.sendCommand(self.command)

                #update len giao dien
                self.update_YesNo_data_to_table(self.result) 
                # self.init_statistic()

            
        elif self.command == 'Check' and self.prev_command == 'Done_detect':
            # print("Check")
            if self.get_cap_check == True:
                ret, image1 = self.cap_check.read() # Lấy dữ liệu từ camera
                # plt.subplot(2,1)
                # plt.imshow(image)
                #plt.imshow(image1, cmap='gray')
                #plt.show()


                self.update_check_image(image1)

                cv2.imwrite('checkjig.jpg', image1)
                img_check = cv2.imread('checkjig.jpg', cv2.IMREAD_GRAYSCALE)
                check = checkAlign.check(img_check)
                print(check)
                # plt.imshow(img_check, cmap='gray')
                # plt.show()

                if check:
                    self.status_cam_inJig = 'Ok_for_jig'
                    self.Controller.send_status_cam_inJig(self.status_cam_inJig)
                    
                    self.jig_signal = self.Controller.status_cam_in_jig()
                    #Test on Jig
                    if self.jig_signal: #da giap Jig
                        #self.checkError = f.err_Check("a")
                        #====================================================================#
                        time.sleep(1)
                        user = "1"
                        arduino.write(bytes(user,'utf-8'))
                        time.sleep(5)
                        data = arduino.readline()
                        print(data)
                        #Change byte to string
                        errCheck_str = data.decode(encoding="utf-8")
                        #print(len(errCheck_str))
                        errCheck = []
                        if(len(errCheck_str)==3):
                            index = 0
                        else: 
                            if(len(errCheck_str)==9):
                                errCheck.append("0")
                                for i in range(len(errCheck_str)):
                                    errCheck.append(errCheck_str[i])
                            else:
                                for i in range(len(errCheck_str)):
                                    errCheck.append(errCheck_str[i])

                            # print("Err_Check: ",errCheck)
                            if errCheck[1] == "1": index = 2
                            if errCheck[7] == "1": index = 8
                            if errCheck[2] == "1": index = 4
                            if errCheck[4] == "1": index = 5
                            if errCheck[6] == "1": index = 6
                            if errCheck[5] == "1": index = 7
                            if errCheck[3] == "1": index = 3
                            if errCheck[0] == "1": index = 1
                        dispMap =["OK" , "I2C" , "DF_S" ,"AF_D" ,"FIXD" ,"EMPT" ,"DATA" ,"AWB" ,"CRC"]
                        Error = dispMap[index]
                        # Error = "OK"
                        print("loi hien tai:", Error)
                #================================================================================#
                        self.checkError = Error

                        if self.checkError == "OK":
                            self.status_cam_checked = 'OK'
                            print(Error)
                            self.Controller.send_status_cam_check(self.status_cam_checked)
                        else:
                            self.status_cam_checked = 'NG'
                            print(Error)
                            self.Controller.send_status_cam_check(self.status_cam_checked)

                        self.update_data_check(self.checkError)
                    
                else:
                    self.status_cam_inJig = 'Skeef'
                    self.Controller.send_status_cam_inJig(self.status_cam_inJig)

                    self.update_data_check("NG")
            self.prev_command = 'check'

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    ex = App()
    # ex.setup_camera()
    print("123")
    ex.show()
    time.sleep(2)
    ex.init_statistic()
    
    ex.setup_camera()

    ex.main_thread.start()

    sys.exit(app.exec_())

    