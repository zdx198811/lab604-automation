# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:12:15 2019

@author: dongxucz
"""
import atexit
from time import asctime
import tempfile
from os import name as os_name
from PyQt5.QtCore import pyqtSignal, QObject

class SigWrapper(QObject):
    msg_sgnl = pyqtSignal(str)
    

class MessageBuf:
    """ Buffer info/warning/error generated during device operation.
    
    This class can be instantiated in any device object to act as a logger
    as well as a message channel to forwared messages to a Qt GUI.
    """
    
    def __init__(self, histlim = 50, msg_src = '', qtGui = True,
                 gui_verbos=(1,1,1), logfile = False, logpath = None):
        """ 
        Arguments:
            histlim - Storage depth. No more history logged beyond histlim.
            msg_src - The name of the device.
            qtGui - global flag to indicate whether messages should be 
                    forwarded to a PyQt based GUI app. Each specific message
                    type generated in warning/error/info function also has the
                    `gui` argument to achieve sub-level control.
            gui_verbos - a tuple with three numbers correponding to verbosity
                         level of GUI message forwarding. e.g. default (1,1,1)
                         means all messages forwarded; (0,1,1) ignores Info.
            logfile - if true, save all messages to disk
            logpath - the path of logfile on the disk (should be a text file)
        """
        self.current_msg = ''
        self.history_msgs = []
        self.hist_lim = histlim
        self._sgnl_wrapper = SigWrapper()
        self.ms = msg_src  # source of message
        self.qtGui = qtGui
        self.gui_verbos = gui_verbos
        self.logfile = logfile
        if logfile:
            if logpath:
                self.logpath = logpath
            else:
                file_name = self.ms + ''.join(asctime().split()[3].split(':')) + '.log'
                if os_name == 'nt':
                    self.logpath = tempfile.gettempdir() + '\\' + file_name
                else:
                    self.logpath = tempfile.gettempdir() + '/' + file_name
  
            self.lf = open(self.logpath, 'w')
            atexit.register(self._cleanup)
            

    def _cleanup(self):
        print("closing log file {}".format(self.lf))
        self.lf.close()
        
    def _msg_commen_op(self, stdout, gui):
        self._push_msg_hist()
        if (gui and self.qtGui):
            self._send_gui_msg()
        if stdout:
            print(self.current_msg)
        if self.logfile:
            self.lf.write(self.current_msg)
    
    def set_gui_verbos(self, x, y, z):
        self.gui_verbos = (x,y,z)
    
    def warning(self, warning_str, stdout=True, gui=True):
        self.current_msg = 'WARNING: {} ({})'.format(warning_str, self.ms)
        self._msg_commen_op(stdout, self.gui_verbos[1])
        
    def error(self, error_str, stdout=True, gui=True):
        self.current_msg = 'ERROR: {} ({})'.format(error_str, self.ms)
        self._msg_commen_op(stdout, self.gui_verbos[2])
    
    def info(self, info_str, stdout=True, gui=True):
        self.current_msg = 'INFO: {} ({})'.format(info_str, self.ms)
        self._msg_commen_op(stdout, self.gui_verbos[0])

    def _send_gui_msg(self):
        self._sgnl_wrapper.msg_sgnl.emit(self.current_msg)
    
    def _push_msg_hist(self):
        self.history_msgs.append( (asctime() + ' - ' + self.current_msg))
        if len(self.history_msgs) > 50:
            self.history_msgs.pop(0)