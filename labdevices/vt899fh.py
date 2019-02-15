# -*- coding: utf-8 -*-
"""
Created on Jan. 31 2019
@author: dongxucz (dongxu.c.zhang@nokia-sbell.com)

Discription:
    This module is the real work horse of the backend, a.k.a., the command executor.

    First, all available commands supported by the device should be defined in the dict named 'CommandSet'.
    The element's structure of 'CommandSet': 
        KEY = API type (corresponding to the VT_Device methods at the frontend)
        VALUE = a dict of (command_string : discription_string)
        
    Then, in the handle() function, do everything correspondingly to respond to the client's command.
    handle() returns a result value, can be used to any extended use. But returning -1 is a special
    code, which will let the VT_Handler finish the TCP session.
"""

import mmap
import subprocess
from core.vt_comm import commandset_pack
from os import name as os_name

subprocess.run(["pwd"])
_MMAP_FILE = './labdevices/vt899-fh_mmap_file.bin'
_RAW_BIN = '/tmp/chan1.bin'

_N_SAMPLE = 28000

CommandSet = {
        'query'    :{'hello'       : 'return hello this is VT899. For testing connectivity'},
                     
        'config'   :{'CloseConnection'  : 'Finish this session. Tell the backend to finish TCP session.',
                     'UpdateRate R'     : 'change sample update rate. R=1,2,3, corresponding to 1s, 0.5s, 0.1s.'},
                     
        'query_bin':{'getRawBin'     : 'send the whole .bin file (unfiltered data) to frontend',
                     'getdata '+str(_N_SAMPLE) : 'return ' + str(_N_SAMPLE) + ' symbols (Each symbol has 8bits).'}
             } # hidden item - 'ComSet' : return the CommandSet. Only used for once when establishing connection. Not visible to frontend user.

def init(sim_flag):
    """ device initialization
    Called by the vt_device_backend.py when program starts.
    """
    if sim_flag:  # for simulation, run the fake data capturing
        subprocess.Popen(["python", "./labdevices/vt899-fh-get-sample-sim.py"])
    else:
        subprocess.run(["systemctl", "stop", "firewalld.service"])  # shutdown firewall
        subprocess.run(["/root/1.2.0_R0/tool/amc590tool", "init"])
        subprocess.Popen(["python", "./labdevices/vt899-fh-get-sample.py"])
        
def handle(command, VT_Handler):
    result = 1 # default value is 1. 
    sock = VT_Handler.request
    print(command)
    if (command == 'ComSet'): # When establishing connection, the client side will query 'CommSet' for once. 
        result = sock.sendall(commandset_pack(CommandSet))
    elif (command == 'hello'):
        result = sock.sendall(bytes(hello(),'utf-8'))
    elif (command == 'CloseConnection'):
        result = -1
    elif (command[0:7] == 'getdata'):
        data_len = _N_SAMPLE
        with open(_MMAP_FILE, 'r') as f:
            handle_getdata(sock, f, int(data_len))
    elif (command == 'getRawBin'):
        with open(_RAW_BIN, 'r') as f:
            sock.sendfile(f)
    elif (command[0:10] == 'UpdateRate'):
        if (command[11] in ['1' , '2' , '3']):
            handle_UpdateRate(sock)
        else:
            sock.sendall(bytes('Command {} not recognized!'.format(command),'utf-8'))
    else:
        result = sock.sendall(bytes('Command {} not recognized!'.format(command),'utf-8'))
    return result


def hello():
    return 'hello, this is VT899!'


def handle_getdata(sock, f, data_len):
    if (os_name == 'posix'):
        with mmap.mmap(f.fileno(), 0,  flags=mmap.MAP_SHARED,
                           access=mmap.ACCESS_READ) as m:
            data_to_send = m[0:data_len]
            sock.sendall(data_to_send)
    else:  # 'nt'
        with mmap.mmap(f.fileno(), 0,  access=mmap.ACCESS_READ) as m:
            data_to_send = m[0:data_len]
            sock.sendall(data_to_send)        


def handle_UpdateRate(sock):
    sock.sendall(bytes('not supported yet', 'utf-8'))

