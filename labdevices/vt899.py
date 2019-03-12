# -*- coding: utf-8 -*-
"""
Created on Jan. 31 2019
@author: dongxucz (dongxu.c.zhang@nokia-sbell.com)

Discription:
    This module is the real work horse of the backend, a.k.a., the command
    executor.
    
    First, all available commands supported by the device should be defined in
    the dict named 'CommandSet'. The element's structure of 'CommandSet': 
        KEY = API type (corresponding to the VT_Device methods at the frontend)
        VALUE = a dict of (command_string : discription_string)
        
    Then, in the handle() function, do everything correspondingly to respond to
    the client's command. handle() returns a result value, can be used to any
    extended use. But returning -1 is a special code, which will let the
    VT_Handler finish the TCP session.
"""

import mmap
import subprocess
import array
from core.vt_comm import commandset_pack
from os import name as os_name
import numpy as np


# subprocess.run(["pwd"])


class Vt899:
    """ Class wrapper for vt899 chassis
    """
    def __init__(self):
        self._MMAP_FILE = './labdevices/vt899-{}_mmap_file.bin'
        self._RAW_BIN = '/tmp/chan1.bin'
        self._N_SAMPLE = 0
        self.CommandSet = {
             'query':{'hello':'return hello this is VT899. For testing connectivity'},
                             
             'config':{'CloseConnection':'Finish this session. Tell the backend to finish TCP session.',
                       'UpdateRate R':'change sample update rate. R=1,2,3, corresponding to 1s, 0.5s, 0.1s.'},
    
             'query_bin':{'getRawBin':'send the whole .bin file (unfiltered data) to frontend'}
              }
             # hidden item - 'ComSet' : return the CommandSet.
             # Only used for once when establishing connection.

    def app_init(self, app_name, sim_flag):
        """ application initialization
        
        Called by the vt_device_backend.py when program starts.
        
        sim_flag - if True, means the backend is 'simulated', i.e. not running
                   on the real device. Because vt899 is essentially a ADC whose
                   core function is capturing waveforms, its not necessary to 
                   run the real ADC hardware everytime when debugging the
                   system, so there are pre-saved waveform data on the disk
                   which can be used to simulate the ADC's output.
        """
        # add application-specific commands to the CommandSet
        if app_name == 'fh':
            self._N_SAMPLE = 28000
            self.CommandSet['query_bin']['getdata {}'.format(self._N_SAMPLE)] = \
            'return {} symbols (Each symbol has 8bits).'.format(self._N_SAMPLE)
        elif app_name == 'pon56g':
            self._N_SAMPLE = 393600  # number of samples from .bin file
            self._N_SAMPLE_FRAME = 196608  # per frame
            self.CommandSet['query_bin']['getdata {}'.format(self._N_SAMPLE)] = \
            'return {} symbols (Each symbol has 8bits).'.format(self._N_SAMPLE)
            self.CommandSet['config']['Update prmbl'] = \
            'Update preamble sequence. Used when initializing demo. the \
            preamble data following this command should be generated by \
            `array.array(prmbl_list)` where prmble_list is a list of 500 \
            NRZ OOK symbols'
            self.CommandSet['query_bin']['getRef 2000'] = \
            'Get reference signal (preamble\'s waveform). return 500 single-\
            precision floting points data (2000 bytes in total)'
            self.CommandSet['query_bin']['getCorr 1570404'] = \
            'Get corr_result_list. For debugging use.'
        else:
            raise ValueError('Vt899.app_init() -> app_name not supported')
        
        # Locate the mmap file. For communication with vt899-get-sample.py
        self._MMAP_FILE = self._MMAP_FILE.format(app_name)

        # run the vt899-get-sample.py script.
        if sim_flag:  # for simulation, run the fake data capturing
            script_arg = ['-s', '-p']
        else:
            script_arg = []
            # shutdown firewall and initialize the amc590 ADC card.
            subprocess.run(["systemctl", "stop", "firewalld.service"])
            subprocess.run(["/root/1.2.0_R0/tool/amc590tool", "init"])
        cmd = ["python", "./labdevices/vt899-get-sample.py", app_name]
        cmd.extend(script_arg)
        subprocess.Popen(cmd)

    def handle(self, command_bytes, VT_Handler):
        result = 1 # default value is 1. 
        sock = VT_Handler.request
        config_data = b''
        if len(command_bytes) < 100:  # just a short string
            command = command_bytes.decode('utf-8')
            print(command)
        else:  # special case. currently just for 'Update prmbl'
            command = command_bytes[0:12].decode('utf-8')
            config_data = command_bytes[12:]
            print('{} command, data len = {}'.format(command, len(config_data)))
            
        # When establishing connection, the client queries 'CommSet' for once.
        if (command == 'ComSet'):
            result = sock.sendall(commandset_pack(self.CommandSet))
        elif (command == 'hello'):
            result = sock.sendall(bytes(self.hello(),'utf-8'))
        elif (command == 'CloseConnection'):
            result = -1
        elif (command[0:7] == 'getdata'):
            data_len = self._N_SAMPLE
            with open(self._MMAP_FILE, 'r') as f:
                self.handle_getdata(sock, f, int(data_len))
        elif (command == 'getRawBin'):
            with open(self._RAW_BIN, 'r') as f:
                result = sock.sendfile(f)
        elif (command[0:10] == 'UpdateRate'):
            if (command[11] in ['1' , '2' , '3']):
                self.handle_UpdateRate(sock)
            else:
                result = sock.sendall(bytes('Command {} not recognized!'.format(command),'utf-8'))
        elif (command[0:12] == 'Update prmbl'):
            if len(config_data) != 500:
                print('Warning!. command UPDATE PRMBL, len(config_data != 500)')
            prmbl = memoryview(config_data).cast('b').tolist()
            with open(self._MMAP_FILE, 'r') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                    self.handle_UpdatePrmbl(m, prmbl)
        elif (command == 'getRef 2000'):
            if hasattr(self, 'rx_p_ref_bytes'):  # make sure Prmbl was updated
                result = sock.sendall(self.rx_p_ref_bytes)
            else:
                print('Warning: pramnble is not present!')
                result = sock.sendall(bytes(b'0'*2000))
        elif (command == 'getCorr 1570404'):
            if hasattr(self, 'corr_result'):
                result = sock.sendall(self.corr_result.tobytes())
            else:
                print('Warning: corr_result is not present!')
                result = sock.sendall(b'Corr_result is not available.')
        else:
            result = sock.sendall(bytes('Command {} not recognized!'.format(command),'utf-8'))
        return result

    def hello(self):
        return 'hello, this is VT899!'

    def handle_getdata(self, sock, f, data_len):
        if (os_name == 'posix'):
            with mmap.mmap(f.fileno(), 0,  flags=mmap.MAP_SHARED,
                               access=mmap.ACCESS_READ) as m:
                data_to_send = m[0:data_len]
                sock.sendall(data_to_send)
        else:  # 'nt'
            with mmap.mmap(f.fileno(), 0,  access=mmap.ACCESS_READ) as m:
                data_to_send = m[0:data_len]
                sock.sendall(data_to_send)        

    def handle_UpdateRate(self, sock):
        sock.sendall(bytes('not supported yet', 'utf-8'))
    
    def handle_UpdatePrmbl(self, m, prmbl_int_list):
        ''' Special method for initializing procedure in PON demo
        
        When the client side calls the 'Update prmbl' command, it meas the AWG
        is sending out a frame with specific preamble. This method will read
        samples captured form AWG and fill `self.prmbl_int` & `self.prmbl_bin`.
        '''
        self.prmbl_int = prmbl_int_list
        samples_all_bin = m[0:self._N_SAMPLE]
        samples_all = np.array(memoryview(samples_all_bin).cast('b').tolist())
        (samples_frame, corr_result) = extract_frame(samples_all,
                                                     self._N_SAMPLE_FRAME,
                                                     prmbl_int_list)
        self.corr_result = np.array(corr_result, dtype = 'float32')
        rx_p_ref = array.array('f', [(samples_frame[i]+samples_frame[-500+i])/2 for i in range(500)])
        self.rx_p_ref_bytes = rx_p_ref.tobytes()


def save_ref_p(filepath, ref_to_save, dtype = 'd'): # 'd': 8byte floating point
    with open(filepath,'wb') as f:
        filedata = array.array(dtype, ref_to_save)
        filedata.tofile(f)


def my_correlation(a,b): #a and b should be one-demension ndarray
    if len(a) != len(b):
        raise ValueError('corr error, len a != len b')
    else:
        return np.matmul(a,b)


def corr_result_list(rawlist, reflist):
    if len(reflist)>len(rawlist):
        raise ValueError('corr_result_list error, len ref > len raw')
    return [np.abs(my_correlation(reflist, rawlist[i:i+len(reflist)])) for i in range(len(rawlist)-len(reflist)+1)]


def extract_frame(samples, frame_len, preamble): # regular method, correlation with preamble
    if (frame_len > len(samples)/2):
        raise ValueError('samples need to contain at least two frames!')
    preamble_dup = np.concatenate([preamble,preamble])
    corr_results = corr_result_list(samples, preamble_dup) 
#    plt.plot(corr_results)
    n_peeks = int(len(samples)/frame_len)
    peeks = []
    for i in range(n_peeks):
        peeks.append(np.argmax(corr_results))
        corr_results[peeks[-1]] = 0
    peeks.sort()
    print(peeks,peeks[1]-peeks[0])
    return (samples[peeks[0]+500:peeks[0]+500+frame_len], corr_results)