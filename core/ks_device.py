# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 17:00:19 2018

@author: dongxucz
"""

import visa # PyVisa info @ http://PyVisa.readthedocs.io/en/stable/
import numpy as np
from time import sleep
from core.msg_buf import MessageBuf

def norm_to_127(samples, remove_bias = True):
    if remove_bias:
        s_shift = (np.array(samples)-np.mean(samples))
        return np.round(127*(s_shift/np.max(np.abs(s_shift))))
    else:
        return np.round(127*(np.array(samples)/np.max(np.abs(samples))))

class M8100AWG:
    """ the base class for Keysight M8100 series AWGs"""

    def __init__(self, addr, rm_path='@py', Glb_Tout = 10000, name = ''):
        """ Initialization - connect to the device.
        
        rm_path: resource manager path. For windows, use Keysight official
        IO control lib; For linux, by default use '@py' which is pyvisa-py.
        addr: address string, e.g. "TCPIP0::172.24.145.91::inst0::INSTR"
        Glb_Tout: global timeout; 
        """
        rm = visa.ResourceManager(rm_path)
        self.msgbf = MessageBuf(msg_src = 'KST' + name)  # store messages and send message to GUI
        # Open Connection
        try:
            self.visa_obj = rm.open_resource(addr)
        except Exception:
            raise ValueError("Unable to connect to " + str(addr) + ". Abort!")
        self.visa_obj.timeout = Glb_Tout
        # get and parse IDN, not nessesary but helps to ensure the connectivity
        IDN = str(self.visa_obj.query("*IDN?"))
        IDN = IDN.split(',') # IDN parts are separated by commas, so parse on the commas
        MODEL = IDN[1]
        FIRMWARE_VERSION = IDN[3]
        self.msgbf.info('connected to '+MODEL+', firmware version '+FIRMWARE_VERSION)
        
        # reset AWG to default configuration
        self.visa_obj.write("*RST")
    
    def query(self, arg_str):
        """ Send a query via VISA protocol, then return a string from the
            remote backend. It mimics the vt_device interface.
        """
        response = self.visa_obj.query(arg_str)
        self.msgbf.info(arg_str + response)
        return response
    
    def config(self, command_str):
        """ Send a message (configuration) via VISA protocol. Mimics the
            vt_device interface. Note that it has to be a command recogonized
            by the equipment.
        """
        self.msgbf.info(command_str)
        self.visa_obj.write(command_str)
    
    def close_visa(self):
        # finish script, close connection.
        self.visa_obj.clear()
        self.visa_obj.close()
        del self.visa_obj
    
    def read_system_error(self):
        next_error = self.visa_obj.query("SYSTem:ERRor?")
        while (next_error[0] != '0'):
            self.msgbf.error(next_error)
            next_error = self.visa_obj.query("SYSTem:ERRor?")
        
    open_visa = __init__
    
    
class M8194A(M8100AWG):
    """ a wrapper for visa interface of Keysight M8194A AWG"""
    def __init__(self, addr, rm_path='@py', Glb_Tout = 10000, name = 'M8194A'):
        M8100AWG.__init__(addr, rm_path, Glb_Tout, name)


class M8195A(M8100AWG):
    """ a wrapper for visa interface of Keysight M8195A AWG"""
    
    def __init__(self, addr, rm_path='@py', Glb_Tout = 10000, name = 'M8195A'):
        M8100AWG.__init__(self, addr, rm_path, Glb_Tout, name)

        
    def set_ext_clk(self):
        self.visa_obj.write(":ROSCillator:SOURce EXTernal")
        REF_SOURCE = self.visa_obj.query(":ROSCillator:SOURce?")
        REF_OK = self.visa_obj.query(":ROSC:SOUR:CHEC? EXT")
        if ((REF_SOURCE[0:3] != 'EXT') or (int(REF_OK) == 0)):
            self.msgbf.error('Oh! forgot to inject external 100M clock?')
            #self.visa_obj.clear()
            self.visa_obj.close()
            raise ValueError('Aborting... Connect ref clock and run again!')
            
    def set_ref_out(self, div1=2, div2=5):
        # set referece out (divide the external 100M by two dividers). default 10M out
        self.visa_obj.write(":OUTP:ROSC:SOUR EXT")
        divider1_ratio = div1
        divider2_ratio = div2
        self.visa_obj.write(":OUTP:ROSC:RCD1 {0}".format(divider1_ratio))    
        while (self.visa_obj.query("*OPC?") != '1\n'):
            sleep(0.1)
        self.visa_obj.write(":OUTP:ROSC:RCD2 {0}".format(divider2_ratio))    
        while (self.visa_obj.query("*OPC?") != '1\n'):
            sleep(0.1)
            
    def set_fs_GHz(self, nGHz=64):
        # Set the sample rate of Channel 1 to 56GHz
        self.visa_obj.write(":FREQuency:RASTer {0}E+09".format(nGHz))
        
    def set_amp(self, a=0.5):
        # Set the Amplitude of Channel 1
        self.visa_obj.write(":VOLTage:AMPLitude {0}".format(a))
        
    def set_ofst(self, ofst=0):
        # Set the Offset of Channel 1
        self.visa_obj.write(":VOLTage:OFFSet {}".format(ofst))
        
    def send_binary_port1(self, samples, remove_bias=True):
        """ send waveform to port1 using Binary transfer.
        
        Note that sample length
        must be n*512, or the segment length of M8195a will not be supported.
        """
        self.msgbf.info('sending to port1. Samples will be trimmed to [-127,+127]', stdout=False)
        # ensure instrument is stopped 
        self.visa_obj.write(":ABORt")
        self.visa_obj.write(":TRACe1:DELete:ALL")
        while (self.visa_obj.query("*OPC?") != '1\n'):
            sleep(0.1)
        # define a segment
        sampleCount = len(samples)

        if (np.max(samples)>127 or np.min(samples)<-127):
            self.msgbf.warning("*** WARNING: samples' values overrange. doing nomalization")
            samples = norm_to_127(samples, remove_bias)
        samples_int = np.array(samples,dtype=int)
        self.visa_obj.write(":TRACe1:DEFine 1,{0}".format(sampleCount))
        while (self.visa_obj.query("*OPC?") != '1\n'):
            sleep(0.1)
        # download waveform, use binary transfer
        self.visa_obj.write_binary_values(":trac1:data 1,0,", samples_int, datatype=u'b')
        while (self.visa_obj.query("*OPC?") != '1\n'):
            sleep(0.1)
        # print error message
        self.read_system_error()
        
        # toggle output 1 port
        self.visa_obj.write(":OUTPut1 on")
        self.visa_obj.write(":INITiate:IMMediate") 
        
