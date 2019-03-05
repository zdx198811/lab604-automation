# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 10:42:22 2019
@author: dongxucz (dongxu.c.zhang@nokia-sbell.com)

Description:
    Use this script to simulate vt899-fh-get-sample.py
    It loads pre-saved data points and share the data with VT_Device_backend
    process via memory mapped file.
    Because vt899's ADC works at 56GHz sampling rate, while the fronthaul demo
    assumes 4GHz sampling rate, signal processing (filtering and downsampling)
    are also done in this module before writing the mmap file.
"""

import mmap
from os import listdir, getcwd
import numpy as np
from scipy.signal import decimate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

_SAMPLE_SENT_SIZE = 28000  # 393600/(56/4)
currdir = getcwd()
_sample_bin_path = currdir + '/labdevices/0122vt899fh'
_all_samples = []
_all_samples_bin = []

_f = open(currdir + '/labdevices/vt899-fh_mmap_file.bin', 'rb+')
_m = mmap.mmap(_f.fileno(), _SAMPLE_SENT_SIZE,  access=mmap.ACCESS_WRITE)

_fig_nrows = 1
_fig_ncols = 2
Scope_figure, Scope_axs = plt.subplots(nrows=_fig_nrows, ncols=_fig_ncols)
Scope_figure.set_size_inches((10, 4))

class Dual_Scope():
    def __init__(self, axs):
        """ ax is a array of two axes."""
        self.axs = axs
        if (self.axs.shape != (2,)):
            raise ValueError('axs dimensions should be (2,)')
        self.patches_collection = [0, 0]

    def update(self, data_frame):
        for col in range(2):
            patches = self.axs[col].plot(data_frame[col], 'bo-')
            self.patches_collection[col] = patches[0]
        return self.patches_collection


def data_feeder_sim():
    global _all_samples
    global _all_samples_bin
    loopcnt = len(_all_samples_bin)
    for i in range(99999):
        # print('the {}th plot '.format(i))
        _m[0:_SAMPLE_SENT_SIZE] = _all_samples_bin[i % loopcnt]
        # print('yield data', list(_m[0:20]))
        yield (np.array(_all_samples[i % loopcnt][0:20]),
               np.array(_all_samples[(i-1) % loopcnt][0:20]))

def decimate_for_fh_demo(samples_56g):
    """
    Because vt899's ADC works at 56GHz sampling rate, while the fronthaul demo
    assumes 4GHz sampling rate, signal processing (filtering and downsampling)
    are also done in this module before writing the mmap file.
    """
    samples_8g = decimate(samples_56g, 7)  # n=None, ftype='iir', axis=-1, zero_phase=True
    samples_4g = decimate(samples_8g, 2)
    return samples_4g

def load_local_sample_files(sample_bin_path):
    global _all_samples
    sample_file_list = listdir(sample_bin_path)
    for filename in sample_file_list:
        with open(sample_bin_path+'/'+filename, 'rb') as f_data:
            bytes_data = f_data.read()
            mview = memoryview(bytes_data)
            mview_int8 = mview.cast('b')
            samples_56g = mview_int8.tolist()
            samples_4g = decimate_for_fh_demo(samples_56g)
            _all_samples.append(samples_4g)

def norm_to_127_int8(originallist):
    temparray = norm_to_127(originallist)
    return temparray.astype('int8')

def norm_to_127(samples, remove_bias = True):
    if remove_bias:
        s_shift = (np.array(samples)-np.mean(samples))
        return np.round(127*(s_shift/np.max(np.abs(s_shift))))
    else:
        return np.round(127*(np.array(samples)/np.max(np.abs(samples))))

def gen_bytearrays():
    global _all_samples
    global _all_samples_bin
    for (idx,sample_list) in enumerate(_all_samples):
        sample_list_norm = norm_to_127_int8(sample_list[0:_SAMPLE_SENT_SIZE])
        _all_samples_bin.append(sample_list_norm.tobytes())

if __name__ == '__main__':
    
    # load samples into global variable _all_samples from csv files
    load_local_sample_files(_sample_bin_path)

    # convert samples into bytearray in global variable _all_samples_bin
    gen_bytearrays()
    
    # plot samples periodically
    scope = Dual_Scope(Scope_axs)
    ani = animation.FuncAnimation(Scope_figure, scope.update, data_feeder_sim,
                                  repeat=False, blit=True, interval=1000)
    plt.show()
    print('finish plotting')
    _m.close()
    _f.close()
    print('close file')