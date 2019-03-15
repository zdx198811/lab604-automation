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

class vt899(vt_device.VT_Device):
    def __init__(self, devname, addr, preamble_int, frame_len, symbol_rate):
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
        self.preamble_len = 500        

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
    if not _SIM:
        print('initialize AWG')
        m8195a = awg(M8195Addr)
    else:
        print('SIM - initialize AWG')
    
    

rx_symbol = []  # real data after transmission
trainset_com = OOK_signal()  # labels
if hostname[0:4]=='amc7':
    awg.send_binary_port1(126*ook_trainset1.nrz(), rescale = False)
    adc_capture()
    samples_all = normalize_rxsymbols(read_sample_bin_file('/tmp/chan1.bin'))
    samples_frame = extract_frame(samples_all, 196608, ook_preamble.nrz())
    rx_p_ref = [(samples_frame[i]+samples_frame[-500+i])/2 for i in range(500)]
    save_ref_p(ref_p_path, rx_p_ref)
else:
    rx_p_ref = read_sample_bin_file(path_str_base+'rx_p_ref.bin', dtype='d') # 'd' for double (8-bytes floating point)
    
for i, traindata in enumerate([ook_trainset1,ook_trainset2,ook_trainset3,ook_trainset5]): 
    if hostname[0:4] == 'amc7':
        awg.send_binary_port1(126*traindata.nrz(), rescale = False)
        adc_capture()
        command_str = 'mv -f /tmp/chan1.bin /root/pyscripts/NN/chan1-{0}.bin'.format(i)
        call(command_str.split(' '))    
        path_str = '/root/pyscripts/NN/chan1-{0}.bin'.format(i)
    else:
        path_str = path_str_base+'chan1-{0}.bin'.format(i)
    samples_all = normalize_rxsymbols(read_sample_bin_file(path_str))
    samples_frame = extract_frame(samples_all, 196608, ook_preamble.nrz())
    rx_symbol.append(resample_symbols(samples_frame,rx_p_ref=rx_p_ref ))  # frame_len = 196608 including preambles. preamble_len = 500
    trainset_com.append(traindata)
    
rx_symbol = np.array(rx_symbol)
rx_symbol = rx_symbol.flatten()
trainset_nrz = trainset_com.nrz()

ook_preamble = OOK_signal(load_file= current_folder+'Jul 6_1741preamble.csv')
vt899 = pon56g_backend_device("vt899pondemo", VT899Addr,
                            ook_preamble.nrz(), 196608, 56)
vt899.open_device()
vt899.print_commandset()
vt899.query('hello')
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
