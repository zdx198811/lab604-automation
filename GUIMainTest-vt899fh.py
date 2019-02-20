# -*- coding: utf-8 -*-
"""
Created on Feb. 13 2019
Test script for the Lab604 testbed GUI framework. Fronthaul application, using
vt899-fh as backend.
@author: dongxucz
"""

import sys
from PyQt5 import QtCore, QtWidgets, QtGui
import csv as csvlib
from locale import atoi
import numpy as np
import core.vt_device as vt_device
import core.dmt_lib as dmt
from guiunits.plotconstallation import MyDynamicMplCanvas
from guiunits.connectbutton import ConnectBtn
from guiunits.calculator import Example

ServerAddr = "172.24.145.24", 9998
#ServerAddr = "192.168.1.4", 9998


class mydevice(vt_device.VT_Device):
    def __init__(self, devname, addr, preamble_int, frame_len, symbol_rate, sample_rate):
        vt_device.VT_Device.__init__(self, devname)
        self.set_net_addr(addr)
        self.preamble_int = preamble_int
        self.dmt_demod = dmt.DmtDeMod(samples = np.zeros(2*len(preamble_int)),
                                      frame_len = frame_len,
                                      symbol_rate = symbol_rate,
                                      sample_rate = sample_rate,
                                      qam_level = 16)
        self.dmt_demod.set_preamble(preamble_int)


   
class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, datadevice):
        QtWidgets.QMainWindow.__init__(self)

        # register the remote hardware device
        self.datadevice = datadevice
        self.datadevice.open_device()
        self.datadevice.print_commandset()
        
        # setup main window
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Fronthaul demo with backend vt899fh")
        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)
        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)
        self.help_menu.addAction('&About', self.about)
        
        # create the main dialog
        self.main_dialog = QtWidgets.QDialog()
        
        # create three groupbox
        self.createTopFigureGroupBox()
        self.createBottomLeftLedGourpBox()
        self.createBottomRightCommandGroupBox()
        
        # setup main layout
        mainLayout = QtWidgets.QVBoxLayout()
        subLayout = QtWidgets.QHBoxLayout()
        subLayout.addStretch(1)
        subLayout.addWidget(self.BottomLeftLedGourpBox)
        subLayout.addStretch(1)
        subLayout.addWidget(self.BottomRightCommandGroupBox)
        subLayout_widget = QtWidgets.QWidget()
        subLayout_widget.setLayout(subLayout)
        mainLayout.addWidget(self.TopFigureGroupBox)
        mainLayout.addWidget(subLayout_widget)
        # mainLayout.setRowStretch(1, 1)
        # mainLayout.setRowStretch(2, 1)
        # mainLayout.setColumnStretch(0, 1)
        # mainLayout.setColumnStretch(1, 1)
        self.main_dialog.setLayout(mainLayout)
        
        
        self.setCentralWidget(self.main_dialog)
        self.statusBar().showMessage("Not connected. Enter bakcend device IP.")
        
    def createTopFigureGroupBox(self):
        self.TopFigureGroupBox = QtWidgets.QGroupBox("Background information")
        self.inforGraph = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap('./guiunits/image.png')
        self.inforGraph.setPixmap(pixmap)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.inforGraph)
        layout.addStretch(1)
        self.TopFigureGroupBox.setLayout(layout)   
        
    def createBottomLeftLedGourpBox(self):
        self.BottomLeftLedGourpBox = QtWidgets.QGroupBox("channel status")
        self.led = Example()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.led)
        self.BottomLeftLedGourpBox.setLayout(layout)
        
    def createBottomRightCommandGroupBox(self):
        self.BottomRightCommandGroupBox = QtWidgets.QGroupBox("control")
        self.Console = QtWidgets.QTextBrowser()
        self.AddrEdit = QtWidgets.QLineEdit()
        self.ConnectButton = ConnectBtn()
        
        layout = QtWidgets.QVBoxLayout()
        sublayout = QtWidgets.QHBoxLayout()
        sublayout_widget = QtWidgets.QWidget()
        sublayout.addWidget(self.AddrEdit)
        sublayout.addWidget(self.ConnectButton)
        sublayout_widget.setLayout(sublayout)
        layout.addWidget(self.Console)
        layout.addWidget(sublayout_widget)
        self.BottomRightCommandGroupBox.setLayout(layout)
        
    def fileQuit(self):
        # close the VT_Device to inform the backend ending the TCP session.
        self.datadevice.close_device()
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
        """      Fronthaul demo v3.0
        Copyright 2019 Nokia Shanghai Bell.
        This program is a test script for the Lab604 testbed GUI framework.
        Contact: Dongxu Zhang
        Email: dongxu.c.zhang@nokia-sbell.com
        Phone: +8613811230782.""")


if __name__ == '__main__':
    # preamble_file_dir = 'D:/PythonScripts/vadatech/vt898/qam16_Apr26.csv'
    preamble_file_dir = './labdevices/0510/qam16_Apr26.csv'
    with open(preamble_file_dir, 'r') as f_pre_int:
        preamble_int192 = [atoi(item[0]) for item in csvlib.reader(f_pre_int)]
    
    vt899 = mydevice("vt899", ServerAddr, preamble_int192, 192, 1.5, 4)
    
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow(vt899)
    
    aw.ConnectButton.signal_wraper.sgnl.connect(aw.Console.append)
    
    aw.setWindowTitle("Lab604 GUI test")
    aw.show()

    sys.exit(qApp.exec_())
    print("close device")
    vt899.close_device()