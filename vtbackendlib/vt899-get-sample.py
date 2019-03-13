# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 2019
@author: dongxucz (dongxu.c.zhang@nokia-sbell.com)

Description:
    Aquire data points from ADC and share the data with vt_device_backend.py
    via memory mapped file.
    This script is intended to be running as a standalone process, together
    with (actually involked automatically by) the vt_device_backend.py script.
"""

import mmap
from time import sleep
from subprocess import check_output
from os import getcwd, listdir
import argparse
import numpy as np
from scipy.signal import decimate
from scipy import interpolate
from scipy.spatial.distance import correlation  # braycurtis,cosine,canberra,chebyshev
import matplotlib.pyplot as plt
import matplotlib.animation as animation

currdir = getcwd()
_sample_bin_path = '/tmp/chan1.bin'
_sample_bin_path_sim_fh = currdir + '/vtbackendlib/0122vt899fh'
_sample_bin_path_sim_pon56g = currdir + '/vtbackendlib/0726vt899pon56g'
_capture_command = '/root/1.2.0_R0/tool/amc590tool fpga_capture 1 adc now'.split(' ')

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

def check_ADC_capture_OK(msg):
    """ check if capture is normal by checking whether 1025 fifo words are read"""
    msg_str = msg.decode()
    msg_str = msg_str.split('\n')
    return int(msg_str[13][30:34])==1025

def adc_capture():
    std_out_bytes = check_output(_capture_command)
    if (check_ADC_capture_OK(std_out_bytes)):
        pass
    else:
        print(std_out_bytes.decode())
        raise ValueError('vt899-get_sample.py -> ADC capture error!')


def resample_symbols(rx_frame,rx_p_ref, intp_n=10):
    """ This function works around the imperfect-sampling-position problem.
        First, the received frame (rx_frame) is interpolated by intp_n times; Then, find a 
        best downsample group by comparing to the reference preamble (rx_p_ref);
        at last, downsample and return the resampled frame (rx_resampled).
    """
    rx_frame = np.concatenate([rx_frame,[rx_frame[-1]]])
    p_len = len(rx_p_ref)
    nsymbol = len(rx_frame)
    # pad the signal with more detail before down-sampling
    x_origin = np.arange(0, nsymbol)
    x_interp = np.arange(0,nsymbol-1, nsymbol/(nsymbol*intp_n))
    f_interp = interpolate.interp1d(x_origin,rx_frame,'cubic')
    rx_interp = f_interp(x_interp)
    rx_interp_left = np.concatenate([[rx_interp[0]]*intp_n,rx_interp[0:-1*intp_n]]) 
    rx_candicate = np.concatenate(
            [np.reshape(rx_interp_left,newshape=(intp_n,-1), order='F'),
             np.reshape(rx_interp,newshape=(intp_n,-1), order='F')])
    # The following line is to sort out a candidate sublist which has the
    # shortest distance from the reference signal. Execept for correlation,
    # other "distances": braycurtis,cosine,canberra,chebyshev,correlation
    dist = [correlation(candi,rx_p_ref) for candi in rx_candicate[:,0:p_len]] 
    rx_resampled = rx_candicate[np.argmin(dist)]
    return rx_resampled

class data_feeder_fh:
    """ Define the generator, iterating over it to get ADC captured data.
    This is for the fronthaul demo.
    """
    def __init__(self, mmap_file, n_sample, sim_flag):
        self._m = mmap_file
        self.n_sample = n_sample
        if sim_flag:
            self._all_samples = []
            self.iterate_fn = self.iterate_fn_sim
        else:
            self.iterate_fn = self.iterate_fn_real
        
    def iterate_fn_real(self):
        """ Use this function as the data generator. """
        for i in range(99999):
            # run the ADC capturing command
            adc_capture()
            with open(_sample_bin_path, "rb") as f_data:
                bytes_data = f_data.read()
            mview = memoryview(bytes_data)
            mview_int8 = mview.cast('b')
            samples_56g = mview_int8.tolist()
            samples_4g = self.decimate_for_fh_demo(samples_56g)
            sample_list_norm = norm_to_127_int8(samples_4g[0:self.n_sample])
            sample_list_bin = sample_list_norm.tobytes()
            self._m[0:self.n_sample] = sample_list_bin
            yield np.array(samples_4g[0:20])
    
    def iterate_fn_sim(self):
        """ Use this function to simulate the real data generator. """
        self.load_local_sample_files(_sample_bin_path_sim_fh)
        _all_samples_bin = []
        for (idx, sample_list) in enumerate(self._all_samples):
            sample_list_norm = norm_to_127_int8(sample_list[0:self.n_sample])
            _all_samples_bin.append(sample_list_norm.tobytes())
        loopcnt = len(_all_samples_bin)
        for i in range(99999):
            # print('the {}th plot '.format(i))
            self._m[0:self.n_sample] = _all_samples_bin[i % loopcnt]
            yield np.array(self._all_samples[i % loopcnt][0:20])

    def load_local_sample_files(self, sample_bin_path):
        sample_file_list = [item for item in listdir(sample_bin_path) if item[-3:]=='bin']
        for filename in sample_file_list:
            with open(sample_bin_path+'/'+filename, 'rb') as f_data:
                bytes_data = f_data.read()
                mview = memoryview(bytes_data)
                mview_int8 = mview.cast('b')
                samples_56g = mview_int8.tolist()
                samples_4g = self.decimate_for_fh_demo(samples_56g)
                self._all_samples.append(samples_4g)

    def decimate_for_fh_demo(self, samples_56g):
        """
        Because vt899's ADC works at 56GHz sampling rate, while the fronthaul demo
        assumes 4GHz sampling rate, signal processing (filtering and downsampling)
        are also done in this module before writing the mmap file.
        """
        samples_8g = decimate(samples_56g, 7)  # n=None, ftype='iir', axis=-1, zero_phase=True
        samples_4g = decimate(samples_8g, 2)
        return samples_4g

class data_feeder_pon56g:
    """ Define the generator, iterating over it to get ADC captured data.
        This is for the high-speed PON demo.
    """
    def __init__(self, mmap_file, n_sample, sim_flag):
        self._m = mmap_file
        self.n_sample = n_sample
        if sim_flag:
            self._all_samples = []
            self.iterate_fn = self.iterate_pon56g_sim
        else:
            self.iterate_fn = self.iterate_pon56g_real
        
    def iterate_pon56g_real(self):
        """ Use this function as the data generator. """
        for i in range(99999):
            # run the ADC capturing command
            adc_capture()
            with open(_sample_bin_path, "rb") as f_data:
                bytes_data = f_data.read()
            mview = memoryview(bytes_data)
            mview_int8 = mview.cast('b')
            samples_56g = mview_int8.tolist()
            sample_list_norm = norm_to_127_int8(samples_56g[0:self.n_sample])
            sample_list_bin = sample_list_norm.tobytes()
            self._m[0:self.n_sample] = sample_list_bin
            yield np.array(samples_56g[0:20])
    
    def iterate_pon56g_sim(self):
        """ Use this function to simulate the real data generator. """
        # print('inside iterate')
        self.load_local_sample_files(_sample_bin_path_sim_pon56g)
        _all_samples_bin = []
        # print('all_samples loaded')
        for (idx, sample_list) in enumerate(self._all_samples):
            sample_list_norm = norm_to_127_int8(sample_list[0:self.n_sample])
            _all_samples_bin.append(sample_list_norm.tobytes())
        loopcnt = len(_all_samples_bin)
        for i in range(99999):
            # print('the {}th plot '.format(i))
            self._m[0:self.n_sample] = _all_samples_bin[i % loopcnt]
            yield np.array(self._all_samples[i % loopcnt][0:20])

    def load_local_sample_files(self, sample_bin_path):
        sample_file_list = [item for item in listdir(sample_bin_path) if item[-3:]=='bin']
        for filename in sample_file_list:
            with open(sample_bin_path+'/'+filename, 'rb') as f_data:
                bytes_data = f_data.read()
                mview = memoryview(bytes_data)
                mview_int8 = mview.cast('b')
                samples_56g = mview_int8.tolist()
                self._all_samples.append(samples_56g)

def norm_to_127_int8(originallist):
    temparray = norm_to_127(originallist)
    return temparray.astype('int8')

def norm_to_127(samples, remove_bias = True):
    if remove_bias:
        s_shift = (np.array(samples)-np.mean(samples))
        return np.round(127*(s_shift/np.max(np.abs(s_shift))))
    else:
        return np.round(127*(np.array(samples)/np.max(np.abs(samples))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("app", type=str, help = "application to run",
                        choices=["fh", "pon56g"])
    parser.add_argument("-s", "--sim", help="simulation mode",
                        action="store_true")
    parser.add_argument("-p", "--plot", help="plot samples",
                        action="store_true")
    args = parser.parse_args()
    sim_flag = args.sim
    plot_flag = args.plot
    app_name = args.app
    
    # Prepare the mmap file
    _f = open(currdir + '/vtbackendlib/vt899-{}_mmap_file.bin'.format(app_name),
              'rb+')
    
    # build the data generator according to the app name
    if app_name == 'fh':
        n_sample = 28000  # 393600/(56/4) see application notes
        _m = mmap.mmap(_f.fileno(), n_sample,  access=mmap.ACCESS_WRITE)
        data_feeder = data_feeder_fh(_m, n_sample, sim_flag)
    elif app_name == 'pon56g':  # vt899 is wrapped as a class
        n_sample = 393600  # see application notes
        _m = mmap.mmap(_f.fileno(), n_sample,  access=mmap.ACCESS_WRITE)
        data_feeder = data_feeder_pon56g(_m, n_sample, sim_flag)
    else:
        raise ValueError('vt899-get-samples.py -> invalid app name')
    
    # Loop over the generator. Plot it or not, according to the `-p` option.
    if plot_flag:  # plot samples periodically
        scope = ADC_Scope(Scope_axs)
        ani = animation.FuncAnimation(Scope_figure, scope.update,
                                      data_feeder.iterate_fn,
                                      repeat=False, blit=True, interval=700)
        plt.show()
    else:
        data_generator = data_feeder.iterate_fn()
        for data_slice in data_generator:
            sleep(0.7)
            print('updated data: '+str(data_slice[0:3])+' ...')
    
    print('finish plotting')
    _m.close()
    _f.close()
    print('close file')
