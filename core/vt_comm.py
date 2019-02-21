# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 11:07:35 2018
@author: Dongxu Zhang
Discription: class definition for the GUI communication functions

Classes:
    VT_Comm -- base class
    VT_CommServer -- used at the backend
    VT_CommClient -- used at the frontend

Utility functions:
    commandset_pack()
    commandset_unpack()
"""
__version__ = '0.0.1'
__author__ = 'Dongxu Zhang'
import socketserver
import socket
import selectors
from json import dumps, loads


def commandset_pack(commandset):
    """ Utility function. pack the device's command set into a JSON bytearray.
    Note that the commandset shoud be a dict hierarchy.
    """
    return dumps(commandset).encode('utf-8')


def commandset_unpack(commandset_bytes):
    """ Utility function. return a dict."""
    return loads(commandset_bytes.decode('utf-8'))


class VT_Comm:
    """ abstract class for VadaTech device's communication module.

    There are two types of subclass module, one is 'VT_CommServer', the other
    'VT_CommClient'. GUI frontend is the client.

    Class variables:
    RCV_CHUNK_SIZE -- (maximum) chunk size for receiving bytes from socket

    Instance variables:
    serverIP
    serverPort
    clientIP
    clientPort
    comm_type
    sock -- the socket object for TCP communication
    """

    RCV_CHUNK_SIZE = 4096

    def __init__(self):
        self.serverIP = "localhost"
        self.serverPort = 9997
        self.clientIP = "localhost"
        self.clientPort = -1
        self.comm_type = 'TCP over Ethernet'
        self.sock = None  # should be allocated in subclass instance


class VT_CommServer(VT_Comm):
    """ Backend communication module, inherited from VT_Comm.

    """
    def __init__(self, ServerAddrTuple, HandlerClass):
        # ServerAddrTuple - (HOST, PORT)
        VT_Comm.__init__(self)
        self.serverIP = ServerAddrTuple[0]
        self.serverPort = ServerAddrTuple[1]
        self.server = socketserver.TCPServer(ServerAddrTuple, HandlerClass)
        self.sock = self.server.socket


class VT_CommClient(VT_Comm):
    """ Frontend communication module, inherited from VT_Comm

    Class methods:
    connect() -- initiate tcp session.
    close_connection() -- end the tcp session
    send_command() -- send a command to remote device
    read_response() -- read command execution results
    send_binfile()
    query()
    """
    def __init__(self):
        VT_Comm.__init__(self)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_state = 0

    def connect(self, ServerAddrTuple):
        self.serverIP = ServerAddrTuple[0]
        self.serverPort = ServerAddrTuple[1]
        return_mesaage = ''
        try:
            self.sock.connect((self.serverIP, self.serverPort))
        except OSError as err:
            return_mesaage = "{}.\nTip: Check the address".format(str(err))
            self.sock.close()
            return return_mesaage
        except ConnectionRefusedError as err:
            print(type(err), str(err))
            self.sock.close()
            return "{}.\nTip: Check the address".format(str(err))
#        else:
#            self.sock.close()
#            return "time out!"
        self.clientIP, self.clientPort = self.sock.getsockname()
        self.socket_state = 1
        return_mesaage = "Connected to !! {}:{}".format(self.serverIP, str(self.serverPort))
        return return_mesaage



    def send_command(self, command_str):
        """ send command to be remotely executed and return an integer.

        Return value:
        1 -- remote execution success.
        -1 -- remote execution fail. Typically because the TCP session failure.
        """
        try:
            self.sock.sendall(bytes(command_str+"\n", "utf-8"))
            return 1
        except ConnectionAbortedError as err:
            print(type(err), str(err))
            self.sock.close()
            return -1

    def read_response(self, read_len=None):
        """ Read the remote command execution results.

        Input argument:
        read_len -- specifies the expected length in Bytes.

        Return value:
        a tuple, (CODE, bytearray). The CODE indicates socket state:
            1 -- nalmal
            -1 -- remote side has shutdown
        """
        received = bytearray()
        # use selecotor to test whether buffer is ready
        with selectors.DefaultSelector() as selector:
            selector.register(self.sock, selectors.EVENT_READ)
            ready = []
            while True:
                # returns empty [] if timeout
                ready = selector.select(timeout=10)
                if ready:
                    chunk = self.sock.recv(self.RCV_CHUNK_SIZE)
                    print('received {}B'.format(len(chunk)))
                    received.extend(chunk)
                    if read_len is not None:
                        if (len(received) == read_len):
                            break
                    else:  # read_len == None, read one-shot is enough.
                        break
                else:
                    print('VT_Comm wait for response timeout!')
                    break

            if (received or (ready == [])):
                if len(received) < 100:  # short response is typically a string
                    print("received {}B: ".format(len(received)),
                          received.decode(errors='ignore'))
                return (1, received)
            else:
                self.socket_state = 0
                self.close_connection()
                return (-1, b'Lost Connection! (server closed)')

    def read_response_old(self):
        """ Read the remote command execution results.

        Return value:
        a tuple, (CODE, BYTES). The CODE indicates socket state:
            1 -- nalmal
            -1 -- remote side has shutdown
        """
        # use selecotor to test whether buffer is ready
        with selectors.DefaultSelector() as selector:
            selector.register(self.sock, selectors.EVENT_READ)
            ready = selector.select(timeout=10)  # returns empty [] if timeout
            if ready:
                received = self.sock.recv(self.RCV_CHUNK_SIZE)
                if received:
                    print("received {} bytes: ".format(len(received)), received)
                    return (1, received)
                else:
                    self.socket_state = 0
                    self.close_connection()
                    return (-1, b'Lost Connection! (server closed)')
            else:
                return (1, b'TimeOut! nothing received.')

    def send_binfile(self, f):
        n_sent = self.sock.sendfile(f)
        return n_sent

    def query(self, command_str, len_return=None):
        """ Send a query command and return results.

        Input arguments:
        command_str -- command to be remotely executed.
        len_return -- the expected length of response.

        Return value:
        (CODE, BYTES) -- See read_response.__doc__
        """
        self.send_command(command_str)
        return self.read_response(len_return)

    def close_connection(self):
        self.sock.close()
