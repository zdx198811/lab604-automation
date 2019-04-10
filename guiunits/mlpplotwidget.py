# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:51:03 2019

@author: dongxucz
"""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, QObject
import numpy as np



def extract_samples_int(bin_data):
    mview = memoryview(bin_data)
    mview_int8 = mview.cast('b')
    samples_int = mview_int8.tolist()
    return samples_int

def channel_filter(raw_iq, start, stop):
    """ select the subcarriers to draw """
    clean_iq = raw_iq[start : stop]
    (N,L) = np.shape(clean_iq)
    return np.reshape(clean_iq, (N*L,), order='F')

class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=4, height=4, dpi=100,
                 datadevice=None, tight_layout=False):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=tight_layout)
        self.axes = fig.add_subplot(111)
        self.datadevice = datadevice
        # self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class simpleSinePlot(MplCanvas):
    """Simple canvas with a sine plot."""

    def compute_initial_figure(self):
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2*np.pi*t)
        self.axes.plot(t, s)



class SigWrapper(QObject):
    sgnl = pyqtSignal(str)
    sgnl_float = pyqtSignal(float)

class fhDemoPlot(MplCanvas):
    """update plots with new data from self.datadevice."""

    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)
        self.update_cnt = 0
        self.draw()
        self.sgnlwrapper = SigWrapper()
        self._equ_repeat_period = 1
        self._SUB_START = 20
        self._SUB_STOP = 35
        self._PLOT_INTERVAL = 300  # ms
        
    def compute_initial_figure(self):
        self.axes.plot([0]*20, 'ro-')
        
    def send_evm_value(self, evm):
        self.sgnlwrapper.sgnl_float.emit(evm)

    def send_console_output(self, console_output):
        self.sgnlwrapper.sgnl.emit(console_output)
        
    def update_figure(self):
        if (self.update_cnt % self._equ_repeat_period) == 0:
            re_clbrt = True
        else:
            re_clbrt = False
        self.update_cnt = self.update_cnt + 1
        print('update figure: {}th time.'.format(self.update_cnt))
        evm = 1
        if (self.datadevice.open_state == 1):
            response = self.datadevice.query_bin('getdata 28000')
            self.send_console_output('getdata 28000')
            alldata = extract_samples_int(response)
            self.datadevice.dmt_demod.update(alldata, re_calibrate = re_clbrt)
            print('!!!!!!!!!!!{}'.format(self.datadevice.dmt_demod.symbols_iq_shaped.shape))
            cleanxy = channel_filter(self.datadevice.dmt_demod.symbols_iq_shaped,
                                     self._SUB_START, self._SUB_STOP)
            evm = self.datadevice.evm_func(cleanxy, self.datadevice.dmt_demod.qam_level)
        else:
            self.send_console_output('ERROR: data device not opend')
            # raise ValueError('data device has not been opend')
        self.axes.cla()
        self.axes.set_xlim(-1.4, 1.4)
        self.axes.set_ylim(-1.4, 1.4)
        scatter_x = cleanxy.real
        scatter_y = cleanxy.imag
        self.axes.scatter( scatter_x, scatter_y, s=5)
        self.send_console_output('EVM = {}%'.format(str(evm*100)))
        self.send_evm_value(evm)
        self.draw()
        

        
class pon56gDemoPlot(MplCanvas):
    """update plots with new data from self.datadevice."""

    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)
        # self.datadevice.open_device()
        # timer = QtCore.QTimer(self)
        # timer.timeout.connect(self.update_figure)
        # timer.start(_PLOT_INTERVAL)
        self.update_cnt = 0
        self.draw()
        self.sgnlwrapper = SigWrapper()
        
    def compute_initial_figure(self):
        self.axes.plot([0]*20, 'ro-')
        
    def send_evm_value(self, evm):
        self.sgnlwrapper.sgnl_float.emit(evm)

    def send_console_output(self, console_output):
        self.sgnlwrapper.sgnl.emit(console_output)
        
    def update_figure(self):
        if (self.update_cnt % _equ_repeat_period) == 0:
            re_clbrt = True
        else:
            re_clbrt = False
        self.update_cnt = self.update_cnt + 1
        print('update figure: {}th time.'.format(self.update_cnt))
        evm = 1
        if (self.datadevice.open_state == 1):
            response = self.datadevice.query_bin('getdata 28000')
            self.send_console_output('getdata 28000')
            alldata = extract_samples_int(response)
            self.datadevice.dmt_demod.update(alldata, re_calibrate = re_clbrt)
            print('!!!!!!!!!!!{}'.format(self.datadevice.dmt_demod.symbols_iq_shaped.shape))
            cleanxy = channel_filter(self.datadevice.dmt_demod.symbols_iq_shaped,
                                     _SUB_START, _SUB_STOP)
            evm = self.datadevice.evm_func(cleanxy, self.datadevice.dmt_demod.qam_level)
        else:
            self.send_console_output('ERROR: data device not opend')
            # raise ValueError('data device has not been opend')
        self.axes.cla()
        self.axes.set_xlim(-1.4, 1.4)
        self.axes.set_ylim(-1.4, 1.4)
        scatter_x = cleanxy.real
        scatter_y = cleanxy.imag
        self.axes.scatter( scatter_x, scatter_y, s=5)
        self.send_console_output('EVM = {}%'.format(str(evm*100)))
        self.send_evm_value(evm)
        self.draw()