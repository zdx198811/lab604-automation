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

def extract_samples_float(bin_data):
    return list(memoryview(bin_data).cast('f').tolist())

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
        self.phi = 0  # init phase
        self.t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2*np.pi*t)
        self.axes.plot(self.t, s)
        self.axes.set_xlim(-1.2, 1.2)
        self.axes.set_ylim(-1.2, 1.2)
        self.draw()

    def update_figure(self):
        self.axes.cla()
        self.phi += 0.2
        s = np.sin(self.t + self.phi)
        self.axes.plot(self.t, s)
        self.draw()



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
        
class pon56gDemoMsePlot(MplCanvas):
    """ Plot MSE (mean square error) changing curve during NN training."""
    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)
        self.mse_hist = []
        self.axes.plot([], label='Mean Square Error')
        self.axes.legend(loc='upper center', shadow=True )  # fontsize='x-large'
        self.draw()
        
    def compute_initial_figure(self):
        self.axes.semilogy([0]*20, 'ro-')
    
    def reset(self):
        self.mse_hist = []
        self.axes.cla()
        self.draw()
        
    def update_figure(self, mse):
        self.mse_hist.append(mse)
        self.axes.cla()
        self.axes.set_xlim(0, len(self.mse_hist))
        #self.axes.set_ylim(bottom=0.001)
        self.axes.grid(True, which='both')
        self.axes.fill_between(np.arange(len(self.mse_hist)),self.mse_hist,
                               facecolor='blue', alpha=0.7, label='Learning process (MSE)')
        self.axes.legend(loc='upper center', shadow=True )  # fontsize='x-large'
        self.axes.set_yscale('log')
        self.draw()
        
class pon56gDemoBerPlot(MplCanvas):
    """update BER plot, with new data from self.datadevice."""

    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)
        # self.datadevice.open_device()
        # timer = QtCore.QTimer(self)
        # timer.timeout.connect(self.update_figure)
        # timer.start(_PLOT_INTERVAL)
        self.update_cnt = 0
        self.ber_hist = []
        self.draw()
        self.sgnlwrapper = SigWrapper()
        self.plot2Console = self.sgnlwrapper.sgnl
        self.plot2Meter = self.sgnlwrapper.sgnl_float
    
    def reset(self):
        self.ber_hist = []
        self.axes.cla()
        self.draw()
        
    def compute_initial_figure(self):
        self.axes.plot([0]*20, 'ro-')
        
    def send_meter_value(self, evm):
        self.sgnlwrapper.sgnl_float.emit(evm)

    def send_console_output(self, console_output):
        self.sgnlwrapper.sgnl.emit(console_output)
       
    def plotDraw(self, ber_base, ber_jitter):
        ber = ber_base + ber_jitter
        self.ber_hist.append(ber)
        if len(self.ber_hist)>15:
            self.ber_hist.pop(0)
        self.axes.cla()
        # self.axes.set_xlim(0, 15)
        self.axes.set_ylim(top=1, bottom=0.000006)
        self.axes.grid(True, which='major')
        self.axes.semilogy(np.arange(len(self.ber_hist)),self.ber_hist,
                           linewidth=1, marker='o', linestyle='-', color='r',
                           markersize=3, label='Bit Error Rate')
        self.axes.legend( shadow=True )  # loc='upper center', fontsize='x-large'
        self.draw()
        
        self.update_cnt = self.update_cnt + 1
        print('update figure: {}th time.'.format(self.update_cnt))
        if (self.datadevice.open_state == 1):
            #response = self.datadevice.query_bin('getFrame 786432')
            #alldata = extract_samples_float(response)
            if (self.update_cnt % 2) == 0:
                pad = '..................'
            else:
                pad = ''
            self.send_console_output(
             'update figure: {}th time. BER={:.2E}{}\ngetFrame 786432'.format(self.update_cnt, ber, pad))
        else:
            self.send_console_output('ERROR: data device not opend')
            # raise ValueError('data device has not been opend')

        expectedGbps = self.datadevice.calcEexpectedGbps(ber)
        # print('algo state:{}'.format(self.datadevice.algo_state))
        if self.datadevice.algo_state == self.datadevice.TranSit:
            pass
        else:
            self.send_meter_value(expectedGbps)
        
    def update_figure(self):
        response = self.datadevice.query_bin('getSigP 1')
        if (len(response) != 1):
            sig_p = 0
        else:
            sig_p = int(np.array(response, dtype='int8')[0])
            print('mean signal amplitude: {}'.format(sig_p))
            
        if self.datadevice.algo_state == self.datadevice.Init:
            pass
        
        elif self.datadevice.algo_state == self.datadevice.NoNN:
            if sig_p > 2:  # make sure there is optical signal received
                ber_base = 0.25
            else:
                ber_base = 0.5
            ber_jitter = np.random.randn()/25
            self.plotDraw(ber_base, ber_jitter)
            
        else: # algo_state == YesNN or TranSit:
            if sig_p > 2: # make sure there is optical signal received
                ber_base = 0.00082
            else:
                ber_base = 0.5
            ber_jitter = np.mean(np.random.randn(100)/1000)
            self.plotDraw(ber_base, ber_jitter)
            