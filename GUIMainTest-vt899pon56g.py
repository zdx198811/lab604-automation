# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:53:44 2019
Description:
    The GUI app for 56G PON demo. For more information, refer to the 
    corresponding application note in the `lab604-automation` documentation.
@author: dongxucz
"""

import sys
from PyQt5 import QtCore, QtWidgets, QtGui
import csv as csvlib
from locale import atoi
import numpy as np
import core.vt_device as vt_device
from core.ook_lib import OOK_signal
import matplotlib.pyplot as plt
from core.ks_device import M8195A

VT899Addr = "172.24.145.24", 9998
current_folder = 'D:\\PythonScripts\\lab604-automation\\vtbackendlib\\0726vt899pon56g\\'

class pon56g_backend_device(vt_device.VT_Device):
    def __init__(self, devname, addr, preamble_int, frame_len, symbol_rate):
        vt_device.VT_Device.__init__(self, devname)
        self.set_net_addr(addr)
        self.preamble_int = preamble_int
        self.preamble_wave = None
        self.preamble_len = 500
        self.frame_len = frame_len
        self.symbol_rate = symbol_rate
        self.trainer = None
        self.inferencer = None
        self.trainset = None


ook_preamble = OOK_signal(load_file= current_folder+'Jul 6_1741preamble.csv')
vt899 = pon56g_backend_device("vt899pondemo", VT899Addr,
                            ook_preamble.nrz(), 196608, 56)
vt899.open_device()
vt899.print_commandset()
vt899.query('hello')

asdf = ook_preamble.nrz(dtype = 'int8')
vt899.config('prmbl500', asdf.tobytes())
ref_bin = vt899.query_bin('getRef 2000')
vt899.preamble_wave = np.array(memoryview(ref_bin).cast('f').tolist())
#
#ookref = vt899.preamble_int*80
#plt.plot(ookref[0:20], 'ro-')
#plt.plot(vt899.preamble_wave[0:20], 'b*-')

#corr_result = np.array(memoryview(vt899.query_bin('getCorr 1570404')).cast('f').tolist())
#plt.plot(corr_result)

vt899.close_device()
