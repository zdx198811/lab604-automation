# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:36:31 2018
Test script for the Lab604 testbed GUI framework.
@author: dongxucz
"""

import sys
from qtpy import QtCore, QtWidgets
import csv as csvlib
from locale import atoi, atof
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import vt_device
import dmt_lib as dmt
from bitstring import BitArray


#ServerAddr = "172.24.145.67", 9998
ServerAddr = "192.168.1.4", 9998
equ_repeat_period = 1
SUB_START = 20
SUB_STOP = 35

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
        

'''
vt855.open_device()
vt855.print_commandset()
vt855.query('hello')
vt855.query('helloworld11')
response1 = vt855.query_bin('getdata 24000')
response2 = vt855.query_bin('getdata 48000')
response3 = vt855.Comm.read_response()
vt855.config('CloseConnection')
vt855.open_state
vt855.close_device()
'''


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100,
                 datadevice=None):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.datadevice = datadevice
        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""

    def compute_initial_figure(self):
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2*np.pi*t)
        self.axes.plot(t, s)


class MyDynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        self.datadevice.open_device()
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(2300)
        self.update_cnt = 0
        
    def compute_initial_figure(self):
        self.axes.plot([0]*20, 'ro-')

    def update_figure(self):
        if (self.update_cnt % equ_repeat_period) == 0:
            re_clbrt = True
        else:
            re_clbrt = False
        self.update_cnt = self.update_cnt + 1
        print('update figure: {}th time.'.format(self.update_cnt))
        
        response = self.datadevice.query_bin('getdata 24000')
        alldata = extract_samples_int(response)
        self.datadevice.dmt_demod.update(alldata, re_calibrate = re_clbrt)
        print('!!!!!!!!!!!{}'.format(self.datadevice.dmt_demod.symbols_iq_shaped.shape))
        cleanxy = channel_filter(self.datadevice.dmt_demod.symbols_iq_shaped, SUB_START, SUB_STOP)
        
        self.axes.cla()
        scatter_x = cleanxy.real
        scatter_y = cleanxy.imag
        self.axes.scatter( scatter_x, scatter_y, s=5)
        self.draw()


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, datadevice):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.main_widget = QtWidgets.QWidget(self)

        qlayout = QtWidgets.QHBoxLayout(self.main_widget)
        cleft = MyDynamicMplCanvas(self.main_widget, width=5, height=4,
                                   dpi=100, datadevice=datadevice)

        qlayout.addWidget(cleft)
        # qlayout.addWidget(cright)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("All hail matplotlib!", 2000)

    def fileQuit(self):
        allchildren = self.main_widget.children()
        # retreive the MyDynamicMplCanvas object
        # which contains the VT_Device instance
        canvs = allchildren[1]
        # close the VT_Device to inform the backend ending the TCP session.
        canvs.datadevice.close_device()
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """GUIMainTest.py
Copyright 2019 Nokia Shanghai Bell.

This program is a test script for the Lab604 testbed GUI framework.

Contact: Dongxu Zhang
         dongxu.c.zhang@nokia-sbell.com
         +8613811230782.
""")


def extract_samples_int(bin_data):
    nsample = int(len(bin_data)/1.5)  # each sample occupies 1.5 bytes
    samples_int = []
    hex_data = bin_data.hex()
    for l in range(nsample):
        sample_bitarray = BitArray('0x'+ hex_data[l*3:(l+1)*3])
        samples_int.append(sample_bitarray.int)
    return samples_int

def channel_filter(raw_iq, start, stop):
    """ select the subcarriers to draw """
    clean_iq = raw_iq[start : stop]
    (N,L) = np.shape(clean_iq)
    return np.reshape(clean_iq, (N*L,), order='F')
                
if __name__ == '__main__':
    preamble_file_dir = 'D:/PythonScripts/vadatech/vt898/qam16_Apr26.csv'
    with open(preamble_file_dir, 'r') as f_pre_int:
        preamble_int192 = [atoi(item[0]) for item in csvlib.reader(f_pre_int)]
    
    vt855 = mydevice("vt855", ServerAddr, preamble_int192, 192, 1.5, 4)
    
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow(vt855)
    aw.setWindowTitle("Lab604 GUI test")
    aw.show()

    sys.exit(qApp.exec_())
    print("close device")
    vt855.close_device()