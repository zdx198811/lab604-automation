# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 2019
@author: dongxucz (dongxu.c.zhang@nokia-sbell.com)

Description:
    Aquire data points from ADC and share the data with VT_Device_backend
    process via memory mapped file.
    Because vt899's ADC works at 56GHz sampling rate, while the fronthaul demo
    assumes 4GHz sampling rate, signal processing (filtering and downsampling)
    are also done in this module before writing the mmap file.
"""

import mmap
import subprocess
from os import getcwd
import numpy as np
from scipy.signal import decimate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

_SAMPLE_SENT_SIZE = 28000  # 393600/(56/4)
currdir = getcwd()
_sample_bin_path = '/tmp/chan1.bin'
_capture_command = ['/root/1.2.0_R0/tool/amc590tool', 'fpga_capture', '1', 'adc', 'now']

_f = open(currdir + '/labdevices/vt899-fh_mmap_file.bin', 'rb+')
_m = mmap.mmap(_f.fileno(), _SAMPLE_SENT_SIZE,  access=mmap.ACCESS_WRITE)

_fig_nrows = 1
_fig_ncols = 1
Scope_figure, Scope_axs = plt.subplots(nrows=_fig_nrows, ncols=_fig_ncols)
Scope_figure.set_size_inches((5, 5))


class ADC_Scope():
    def __init__(self, axs):
        """ ax is a array of two axes."""
        self.axs = axs
        self.patches_collection = []

    def update(self, data_frame):
        # self.patches_collection.append(self.axs.plot(data_frame, 'bo-'))
        self.patches_collection = self.axs.plot(data_frame, 'bo-')
        return self.patches_collection


def data_feeder_sim():
    for i in range(99999):
        # run the ADC capturing command
        subprocess.run(_capture_command)
        
        with open(_sample_bin_path, "rb") as f_data:
            bytes_data = f_data.read()
            
        mview = memoryview(bytes_data)
        mview_int8 = mview.cast('b')
        samples_56g = mview_int8.tolist()
        samples_8g = decimate(samples_56g, 7)  # n=None, ftype='iir', axis=-1, zero_phase=True
        samples_4g = decimate(samples_8g, 2)
        sample_list_norm = norm_to_128(samples_4g)
        sample_list_bin = sample_list_norm.tobytes()
        _m[0:_SAMPLE_SENT_SIZE] = sample_list_bin
        yield np.array(samples_4g[0:20])


def norm_to_128(originallist):
    meanvalue = np.mean(originallist)
    templist = [item-meanvalue for item in originallist]  # remove offset
    maxvalue = max(np.abs(templist))
    temparray = np.array([item*(127/maxvalue) for item in templist[0:_SAMPLE_SENT_SIZE]])
    return temparray.astype('int8')


if __name__ == '__main__':
    # plot samples periodically
    scope = ADC_Scope(Scope_axs)
    ani = animation.FuncAnimation(Scope_figure, scope.update, data_feeder_sim,
                                  repeat=False, blit=True, interval=900)
    plt.show()
    print('finish plotting')
    _m.close()
    _f.close()
    print('close file')
