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

############## Temp. code #################
_SIM = True
###########################################

VT899Addr = "172.24.145.24", 9998
M8195Addr = "172.24.145.77"
current_folder = 'D:\\PythonScripts\\lab604-automation\\vtbackendlib\\0726vt899pon56g\\'

class vtdev(vt_device.VT_Device):
    def __init__(self, devname, addr, frame_len, symbol_rate):
        vt_device.VT_Device.__init__(self, devname)
        self.set_net_addr(addr)
        self.frame_len = frame_len
        self.symbol_rate = symbol_rate
        self.trainer = None
        self.inferencer = None
        self.trainset = None
    
    def set_preamble(self, preamble_int):
        self.preamble_int = preamble_int
        self.preamble_wave = None
        self.preamble_len = len(preamble_int)

class awg(M8195A):
    def __init__(self, addr):
        M8195A.__init__("TCPIP0::{}::inst0::INSTR".format(addr))
        self.set_ext_clk()
        self.set_ref_out()  # set ref_out frequency by configuring two dividers 
                            # default parameters: div1=2, div2=5
        self.set_fs_GHz(nGHz=56)
        self.set_amp(0.6)   # set output amplitude (range between 0 and 1)
        self.set_ofst(0)    # set output offset

if __name__ == '__main__':
    path_str = 'D:\\PythonScripts\\vadatech\\vt899\\NN\\'
    ook_preamble = OOK_signal(load_file= path_str+'Jul 6_1741preamble.csv')
    frame_len = 196608
    trainset = OOK_signal()
    trainset.append(OOK_signal(load_file=path_str+'Jul 9_0841train.csv'))
    trainset.append(OOK_signal(load_file=path_str+'Jul 9_0842train.csv'))
    trainset.append(OOK_signal(load_file=path_str+'Jul 9_0843train.csv'))
    trainset.append(OOK_signal(load_file=path_str+'Jul 9_0845train.csv'))

    if not _SIM:
        # initiate AWG
        m8195a = awg(M8195Addr)
    
    vt899 = vtdev("vt899pondemo", VT899Addr, frame_len, 56)
    vt899.open_device()
    vt899.print_commandset()
    vt899.query('hello')
    
    if not _SIM:
        # send a frame containing preamble
        data_for_prmbl_sync = trainset.slicer(slice(frame_len))
        awg.send_binary_port1(250*(data_for_prmbl_sync-0.5))
    
    ook_prmb = ook_preamble.nrz(dtype = 'int8')
    vt899.config('prmbl500', ook_prmb.tobytes())
    
    ref_bin = vt899.query_bin('getRef 2000')
    vt899.preamble_wave = np.array(memoryview(ref_bin).cast('f').tolist())
    #
    #ookref = vt899.preamble_int*80
    #plt.plot(ookref[0:20], 'ro-')
    #plt.plot(vt899.preamble_wave[0:20], 'b*-')
    
    #corr_result = np.array(memoryview(vt899.query_bin('getCorr 1570404')).cast('f').tolist())
    #plt.plot(corr_result)
    
    vt899.close_device()
    
    





