# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 11:05:18 2018
@author: dongxu zhang
Discription: class definition for the VT_Device. (GUI frontend module)
Change log:
    20190227 - add pyqtSignal support to enable PyQt based GUI signal emit
"""

__version__ = '0.0.1'
from core import vt_comm
from core.msg_buf import MessageBuf

    
class VT_Device:
    """ the father class of all Vadatech devices (for GUI frontend).
    
    A VT_Device object provide open(), close(), config() and query()
    interfaces. Command supported by config() and query() are listed
    in the VT_Device.CommandSet. The CommandSet is auto-retrieved from
    the remote backend when open_device() is called.

    Attributes:
        dev_name - a string
        has_gui - decide whether messages are forwarded to a GUI app.
        open_state - a integer indicating the socket state. 0=Closed,
                     1=Connected, -1=RemoteShutdown
        CommandSet - a dict describing all supported commands.
        Comm - VT_CommClient object. the communication module.
        net_addr - network address of the remote backend.
        msgbf - stor messages and send messages to GUI

    Methods:
        set_net_addr(ipaddr, tcpport) - set the remote backend IP
        set_gui_verbos(x,y,z) - set which type of message is forwarded to GUI
        open_device() - connect to the remote backend and retrieve CommandSet
        config() - write message to the backend
        query() - retrieve data from the backend
        print_commandset() - print out all the CommandSet
    """

    # class methods:
    def __init__(self, dev_name, has_gui=False):
        self.dev_name = dev_name    # instance variable unique to each instance
        self.open_state = 0
        self.CommandSet = {}
        self.Comm = vt_comm.VT_CommClient()
        # default Communication method is Ethernet based.
        # To use other backend connection methods, define
        # new VT_Comm subclass and then overide this
        # class variable when instantiate VT_Device objects.
        self.net_addr = ("localhost", 9997)
        self.has_gui = has_gui
        self.msgbf = MessageBuf(100, dev_name, has_gui)
        self.guisgnl = self.msgbf._sgnl_wrapper.msg_sgnl

    def set_net_addr(self, addrtuple):
        """ example: ipaddr = "192.168.56.101", tcpport = 9997 """
        if addrtuple:
            (ipaddr, tcpport) = addrtuple
            self.net_addr = (ipaddr, tcpport)
    
    def set_gui_verbos(self, x, y, z):
        """ Set verbosity level of GUI message forwarding.
        
        x - Info, y - Warning, z - Error. 
        E.g. (1,1,1) means all messages forwarded; (0,1,1) ignores Info.
        """
        if self.has_gui:
            self.msgbf.set_gui_verbos(x, y, z)
        else:
            print('Operation Failed. self.has_gui = False')

    def open_device(self, addrTupe=None):
        """ starts tcp session
        
        Argument: addrTuple, which contains a IP string and an integer tcp port
            number. Eg. ('192.168.1.4', 9998). If not present, make sure to use
            set_net_addr() method first, or it tries to connect localhost.
            
        Return: a string, describing whether the connection succeeds or fails.
        """
        if self.open_state == 1:
            ConnectResult = 'opened! To re-open, run close_device() first.'
        else:
            self.Comm.__init__()
            if addrTupe:
                ConnectResult = self.Comm.connect(addrTupe)
            else:
                ConnectResult = self.Comm.connect(self.net_addr)
            if (ConnectResult[0:9] == "Connected"):
                (self.open_state, response) = self.Comm.query('ComSet')
                if (self.open_state == 1):
                    self.CommandSet = vt_comm.commandset_unpack(response)
        self.msgbf.info(ConnectResult)
        return ConnectResult

    def config(self, command_str,  databytes = b''):
        """ one-way configuration. Send a command with no response expected."""
        if (self.open_state != 1):
            # check if connected. The remote side may have shut down.
            self.msgbf.warning('Operation failed! Socket not connected.')
        else:
            # return 1 if success
            self.open_state = self.Comm.send_command(command_str, databytes)
            self.msgbf.info('{} operation succeed.'.format(command_str))

    def close_device(self):
        """shut local connection and release local socket.

        Note that sending a CloseConnection command forces the server side
        to close the session.
        """
        if self.open_state == 0:
            self.msgbf.warning('Not in open state!')
        else:
            if self.open_state == 1:
                self.config('CloseConnection')
            self.Comm.close_connection()
            self.open_state = 0
            self.msgbf.info('Device closed.')

    def query(self, arg_str):
        """ Send a command, then return a string from the remote backend. """
        response = self.query_bin(arg_str)
        response_str = response.decode()
        self.msgbf.info(response_str)
        return response_str

    def query_bin(self, arg_str):
        """ Send a command, then return a bytearray from the remote backend."""
        if (self.open_state != 1):  # check if connected.
            self.msgbf.warning('Operation failed! Socket not connected.')
            return b''
        else:
            self.msgbf.info('query_bin {}'.format(arg_str))
            len_return = self.args_str_parse(arg_str)
            (self.open_state, response) = self.Comm.query(arg_str, len_return)
            return response

    def print_commandset(self):
        self.msgbf.info(self.args_str_parse.__doc__)
        for category in self.CommandSet:
            print('\n\n{} Commands:'.format(category))
            for cmd in self.CommandSet[category]:
                print('\n    {}:\n        {}'.format(cmd, self.CommandSet[category][cmd]))
        return self.CommandSet

    def args_str_parse(self, arg_str):
        """
        VT_device's query commands may be in one of the following syntax:
            Syntax 1: 'COMMAND'
            Syntax 2: 'COMMAND VALUE'
        The first applies to scenarios that the length of response is\
        unknown, which typically is a simple status reading that returns a\
        short string. The second form has two sub-situations: one is for\
        commands with explicit expectation of returned length (bytes), such as\
        capturing (query) a certain amount of data, the other situation is to\
        send (config) a certain amount of data to the device .
        """
        arg_split = arg_str.split()
        if (len(arg_split) == 2):
            return int(arg_split[1])
        else:
            return None
