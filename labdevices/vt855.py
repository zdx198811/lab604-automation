# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:12:16 2018
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
_MMAP_FILE = './labdevices/vt855_mmap_file.bin'


CommandSet = {
        'query'    :{'hello'            : 'return hello back. Just for testing connectivity'},
                     
        'config'   :{'CloseConnection'  : 'Finish this session. Tell the backend to finish TCP session.',
                     'UpdateRate R'     : 'change sample update rate. R=1,2,3, corresponding to 1s, 0.5s, 0.1s.'},
                     
        'query_bin':{'getdata 24000'    : 'return 16000 symbols (Each symbol has 12bits). 24KB in total',
                     'getdata 48000'    : 'return 32000 symbols (Each symbol has 12bits). 48KB in total'}
             } # hidden item - 'ComSet' : return the CommandSet. Only used for once when establishing connection. Not visible to frontend user.

def app_init(app, sim_flag):
    if app is in ["fh"]:
        if sim_flag:
            subprocess.Popen(["python", "./labdevices/send_sample_req_sim.py"])
        else:
            subprocess.Popen(["python", "./labdevices/send_sample_req.py"])
    else:
        raise ValueError("vt855 supports application: fh")
        
def handle(command, VT_Handler):
    result = 1 # default value is 1. 
    sock = VT_Handler.request
    print(command)
    if (command == 'ComSet'): # When establishing connection, the client side will query 'CommSet' for once. 
        result = sock.sendall(commandset_pack(CommandSet))
    elif (command == 'hello'):
        result = sock.sendall(bytes(helloworld(),'utf-8'))
    elif (command == 'CloseConnection'):
        result = -1
    elif (command[0:7] == 'getdata'):
        data_len = command[8:13] # '24000' or '48000'
        if data_len in ['24000', '48000']:
            with open(_MMAP_FILE, 'r') as f:
                handle_getdata(sock, f, int(data_len))
        else:
            sock.sendall(bytes('Command {} not recognized!'.format(command),'utf-8'))
    elif (command[0:10] == 'UpdateRate'):
        if (command[11] in ['1' , '2' , '3']):
            handle_UpdateRate(sock)
        else:
            sock.sendall(bytes('Command {} not recognized!'.format(command),'utf-8'))
    else:
        result = sock.sendall(bytes('Command {} not recognized!'.format(command),'utf-8'))
    return result


def helloworld():
    return 'hello, this is VT855!'


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


