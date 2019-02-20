# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:51:03 2019

@author: dongxucz
"""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy import QtCore, QtWidgets
import numpy as np

_equ_repeat_period = 1
_SUB_START = 20
_SUB_STOP = 35
_PLOT_INTERVAL = 300  # ms

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


class MyStaticMplCanvas(MplCanvas):
    """Simple canvas with a sine plot."""

    def compute_initial_figure(self):
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2*np.pi*t)
        self.axes.plot(t, s)


class MyDynamicMplCanvas(MplCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)
        # self.datadevice.open_device()
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(_PLOT_INTERVAL)
        self.update_cnt = 0
        
    def compute_initial_figure(self):
        self.axes.plot([0]*20, 'ro-')

    def update_figure(self):
        if (self.update_cnt % _equ_repeat_period) == 0:
            re_clbrt = True
        else:
            re_clbrt = False
        self.update_cnt = self.update_cnt + 1
        print('update figure: {}th time.'.format(self.update_cnt))
        
        if (self.datadevice.open_state == 1):
            response = self.datadevice.query_bin('getdata 28000')
            alldata = extract_samples_int(response)
            self.datadevice.dmt_demod.update(alldata, re_calibrate = re_clbrt)
            print('!!!!!!!!!!!{}'.format(self.datadevice.dmt_demod.symbols_iq_shaped.shape))
            cleanxy = channel_filter(self.datadevice.dmt_demod.symbols_iq_shaped, _SUB_START, _SUB_STOP)
        else:
            raise ValueError('data device has not been opend')
        self.axes.cla()
        self.axes.set_xlim(-1.4, 1.4)
        self.axes.set_ylim(-1.4, 1.4)
        scatter_x = cleanxy.real
        scatter_y = cleanxy.imag
        self.axes.scatter( scatter_x, scatter_y, s=5)
        self.draw()