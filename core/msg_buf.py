# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:12:15 2019

@author: dongxucz
"""

from time import asctime
from PyQt5.QtCore import pyqtSignal, QObject

class SigWrapper(QObject):
    msg_sgnl = pyqtSignal(str)
    

class MessageBuf:
    """ Buffer info/warning/error generated during device operation """
    
    def __init__(self, histlim = 50, msg_src = ''):
        self.current_msg = ''
        self.history_msgs = []
        self.hist_lim = histlim
        self._sgnl_wrapper = SigWrapper()
        self.ms = msg_src  # source of message
        
    def _msg_commen_op(self, stdout):
        self._push_msg_hist()
        self._send_gui_msg()
        if stdout:
            print(self.current_msg)
    
    def warning(self, warning_str, stdout=True):
        self.current_msg = 'WARNING: {} ({})'.format(warning_str, self.ms)
        self._msg_commen_op(stdout)
        
    def error(self, error_str, stdout=True):
        self.current_msg = 'ERROR: {} ({})'.format(error_str, self.ms)
        self._msg_commen_op(stdout)
    
    def info(self, info_str, stdout=True):
        self.current_msg = 'INFO: {} ({})'.format(info_str, self.ms)
        self._msg_commen_op(stdout)

    def _send_gui_msg(self):
        self._sgnl_wrapper.msg_sgnl.emit(self.current_msg)
    
    def _push_msg_hist(self):
        self.history_msgs.append( (asctime() + ' - ' + self.current_msg))
        if len(self.history_msgs) > 50:
            self.history_msgs.pop(0)