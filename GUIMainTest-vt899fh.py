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
from guiunits.mlpplotwidget import fhDemoPlot
from guiunits.connectbutton import ConnectBtn
from guiunits.ledpannel import LedPannel

ServerAddr = "172.24.145.24", 9998
#ServerAddr = "192.168.1.4", 9998


class demo_backend_device(vt_device.VT_Device):
    def __init__(self, devname, addr, preamble_int, frame_len, symbol_rate, sample_rate):
        vt_device.VT_Device.__init__(self, devname, has_gui=True)
        self.set_net_addr(addr)
        self.preamble_int = preamble_int
        self.dmt_demod = dmt.DmtDeMod(samples = np.zeros(2*len(preamble_int)),
                                      frame_len = frame_len,
                                      symbol_rate = symbol_rate,
                                      sample_rate = sample_rate,
                                      qam_level = 16)
        self.dmt_demod.set_preamble(preamble_int)
        self.evm_func = dmt.evm_estimate

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, datadevice):
        QtWidgets.QMainWindow.__init__(self)

        # register the remote hardware device
        self.datadevice = datadevice
        
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
        self.createBottomMiddlePlotGroupBox()
        self.createBottomRightCommandGroupBox()
        
        # setup main layout
        mainLayout = QtWidgets.QVBoxLayout()
        subLayout = QtWidgets.QHBoxLayout()
        subLayout.addWidget(self.BottomLeftLedGourpBox)
        # subLayout.addStretch(1)
        subLayout.addWidget(self.BottomMiddlePlotGourpBox)
        subLayout.addWidget(self.BottomRightCommandGroupBox)
        subLayout_widget = QtWidgets.QWidget()
        subLayout_widget.setLayout(subLayout)
        mainLayout.addWidget(self.TopFigureGroupBox)
        mainLayout.addWidget(subLayout_widget)
        self.main_dialog.setLayout(mainLayout)

        # initialize signal-slot connections        
        self.ConnectButton.clicked.connect(self.openVTdevice)
        self.AddrEdit.returnPressed.connect(self.openVTdevice)
        self.StartStopButton.clicked.connect(self.startAcquisition)
        self.TestConnectionButton.clicked.connect(self.testConnection)
        self.QuitButton.clicked.connect(self.closeEvent)
        self.constellation.sgnlwrapper.sgnl.connect(self.Console.append)
        self.constellation.sgnlwrapper.sgnl_float.connect(self.ledPannelChangeState)
        self.datadevice.guisgnl.connect(self.Console.append)
        
        # create the main timer object
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.constellation.update_figure)
        # set QMainWindow's central widget and show status bar message
        self.setCentralWidget(self.main_dialog)
        self.statusBar().showMessage("Not connected. Enter bakcend device IP.")

    def createTopFigureGroupBox(self):
        self.TopFigureGroupBox = QtWidgets.QGroupBox("Background information")
        self.inforGraph = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap('./guiunits/imags/fh-bkg.png')
        self.inforGraph.setPixmap(pixmap)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.inforGraph)
        # layout.addStretch(1)
        self.TopFigureGroupBox.setLayout(layout)   
        
    def createBottomLeftLedGourpBox(self):
        self.BottomLeftLedGourpBox = QtWidgets.QGroupBox("channel status")
        self.leds = LedPannel()
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.leds)
        self.BottomLeftLedGourpBox.setLayout(layout)
    
    def createBottomMiddlePlotGroupBox(self):
        self.BottomMiddlePlotGourpBox = QtWidgets.QGroupBox("constellation digram")
        self.constellation = fhDemoPlot(parent=None, width=5, height=4,
                                   dpi=100, datadevice=self.datadevice)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.constellation)
        self.BottomMiddlePlotGourpBox.setLayout(layout)
        
    def createBottomRightCommandGroupBox(self):
        self.BottomRightCommandGroupBox = QtWidgets.QGroupBox("control")
        self.Console = QtWidgets.QTextBrowser()
        self.AddrEdit = QtWidgets.QLineEdit()
        self.ConnectButton = ConnectBtn(self.AddrEdit)
        self.TestConnectionButton = QtWidgets.QPushButton("Test Connection")
        self.StartStopButton = QtWidgets.QPushButton("Start acquisition")
        self.QuitButton = QtWidgets.QPushButton("Quit")

        layout = QtWidgets.QVBoxLayout()
        sublayout = QtWidgets.QGridLayout()
        sublayout_widget = QtWidgets.QWidget()
        sublayout.addWidget(self.AddrEdit, 1, 0, 1, 2)
        sublayout.addWidget(self.ConnectButton, 1, 2)
        sublayout.addWidget(self.TestConnectionButton, 2, 0)
        sublayout.addWidget(self.StartStopButton, 2, 1)
        sublayout.addWidget(self.QuitButton, 2, 2)
        sublayout_widget.setLayout(sublayout)
        layout.addWidget(self.Console)
        layout.addWidget(sublayout_widget)
        self.BottomRightCommandGroupBox.setLayout(layout)

    def openVTdevice(self):
        ipaddr = self.AddrEdit.text()
        print((ipaddr, 9998))
        self.datadevice.set_net_addr((ipaddr,9998))
        self.Console.append('connecting to'+ ipaddr)
        self.datadevice.open_device()

    def testConnection(self):
        if self.datadevice.open_state == 1:
            response = self.datadevice.query('hello')
            if response:
                self.TestConnectionButton.setStyleSheet('QPushButton {background-color: #01FF53;}')
        else:
            self.Console.append('not connected (datadevice.open_state = 0)')
            self.TestConnectionButton.setStyleSheet('QPushButton {background-color: #FF0000;}')

    def ledPannelChangeState(self, evm):
        if (evm >= 0.13):
            self.leds.turn_all_off()
        elif ((evm >= 0.1) and (evm < 0.13)):
            self.leds.turn_all_warning()
        else:
            self.leds.turn_all_on()

    def startAcquisition(self):
        if (self.datadevice.open_state == 1):
            self.Console.append("Start data acquisition!")
            self.timer.setInterval(600)
            self.timer.start()
            self.StartStopButton.clicked.disconnect(self.startAcquisition)
            self.StartStopButton.setText("Stop")
            self.StartStopButton.clicked.connect(self.stopAcquisition)
        else:
            self.Console.append("ERROR: Data Device Not Connected.")
        
    def stopAcquisition(self):
        self.Console.append("Stopped!")
        self.timer.stop()
        self.StartStopButton.clicked.disconnect(self.stopAcquisition)
        self.StartStopButton.setText("Start acquisition")
        self.StartStopButton.clicked.connect(self.startAcquisition)

    def fileQuit(self):
        # close the VT_Device to inform the backend ending the TCP session.
        self.datadevice.close_device()
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
        """  Fronthaul demo v3.0
        This program is a test script for the Lab604 testbed GUI framework.
        Currently just for personal practice. Under MIT license.
        Contact: Dongxu Zhang
        Email: dongxu.c.zhang@nokia-sbell.com
        Phone: +8613811230782.""")


if __name__ == '__main__':
    # preamble_file_dir = 'D:/PythonScripts/vadatech/vt898/qam16_Apr26.csv'
    preamble_file_dir = './vtbackendlib/0510vt855fh/qam16_Apr26.csv'
    with open(preamble_file_dir, 'r') as f_pre_int:
        preamble_int192 = [atoi(item[0]) for item in csvlib.reader(f_pre_int)]
    
    vt899 = demo_backend_device("vt899", ServerAddr,
                                preamble_int192, 192, 1.5, 4)
    
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow(vt899)
    
    aw.setWindowTitle("Lab604 GUI test")
    aw.show()

    sys.exit(qApp.exec_())
    print("close device")
    vt899.close_device()