# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 17:03:37 2018
@author: dongxucz (dongxu.c.zhang@nokia-sbell.com)

Description:
    Use this script to simulate send_sample_req.py
    It generates random data points and share the data with VT_Device_backend
    process via memory mapped file.
"""

import mmap
from os import listdir
import csv as csvlib
from locale import atoi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bitstring import BitArray

_sample_csv_path = './0510'
_all_samples = []
_all_samples_bin = []
_f = open('mmap_file.bin', 'rb+')
_m = mmap.mmap(_f.fileno(), 48000,  access=mmap.ACCESS_WRITE)

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
        print('the {}th plot '.format(i))
        if i % 2 == 0:
            _m[0:24000] = _all_samples_bin[i % loopcnt]
        else:
            _m[24000:48000] = _all_samples_bin[i % loopcnt]
        print('yield data', list(_m[0:20]))
        yield (np.array(_all_samples[i % loopcnt][0:20]),
               np.array(_all_samples[(i-1) % loopcnt][0:20]))


def load_local_sample_files(csv_path):
    global _all_samples
    sample_file_list = listdir(csv_path)
    for filename in sample_file_list:
        f_data = open(_sample_csv_path+'/'+filename, 'r')
        samples_list = [atoi(item[0]) for item in csvlib.reader(f_data)]
        f_data.close()
        _all_samples.append(samples_list)
    


def gen_bytearrays():
    global _all_samples
    global _all_samples_bin
    for (idx,sample_list) in enumerate(_all_samples):
        to_add = BitArray()
        for sample in sample_list:
            to_add.append(BitArray(int=sample, length=12))
        _all_samples_bin.append(to_add.bytes)


#asdf = _all_samples_bin[0]
#samples_int = []
#alldata = asdf.hex()
#for l in range(16000):  #each packet contains 800 samples
#    sample_bitarray = BitArray('0x'+ alldata[l*3:(l+1)*3])
#    samples_int.append(sample_bitarray.int)
#                


if __name__ == '__main__':
    
    # load samples into global variable _all_samples from csv files
    load_local_sample_files(_sample_csv_path)
    
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
