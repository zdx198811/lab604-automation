# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 10:51:40 2018

@author: Dongxu Zhang (dongxu.c.zhang@nokia-sbell.com)

Discription:
    The main function entrence of the vadatech device backend for remote
    control (GUI). All command handling are implemented in the vtXXX module.

"""
__version__ = '0.0.1'
import argparse
from socketserver import BaseRequestHandler
import core.vt_comm as vt_comm

HOST, PORT = "172.24.145.24", 9998         # check this line before running
# HOST, PORT = "192.168.1.4", 9998

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device", help="the name of current running chassis",
                        choices = ["vt855", "vt899"])
    parser.add_argument("app", help="the application to run")
    parser.add_argument("-s", "--sim", help="simulation mode",
                        action="store_true")
    args = parser.parse_args()

    if args.device == 'vt855':
        import labdevices.vt855 as vtXXX
    elif args.device == 'vt899':  # vt899 is wrapped as a class
        from labdevices.vt899 import Vt899
        vtXXX = Vt899()
    else:
        pass

    try:
        vtXXX.app_init(args.app, args.sim)  # init action for the specific app.
        # If the app name is not recognized, a ValueError will be raised.
        # When args.sim is True, use saved sample files.
    except ValueError as err:
        print(err)
        exit(0)
    
    class VT_Handler(BaseRequestHandler):
        """ used by the VT_Comm module when a new connection is requested.
    
        Once a connection is established, the handle() method is called.
        Finishing handle() means closing connection. Refer to socketserver
        standard library for details.
    
        Usually this class is instantiated once for each transfer (e.g.
        each command).
        But since we have no need to do multi-client threading, we can
        retain the socket and communicate repeatedly in handle(), untill
        the client closes it.
    
        setup() and finish() are optional, may be usefule in some cases.
        Refer to 'socketserver' documentation in Python standard library.
        """
        def handle(self):
            print('connected to {}'.format(self.client_address))
            # self.request is the TCP socket connected to the client
            while True:
                self.data = self.request.recv(vt_comm.VT_CommServer.RCV_CHUNK_SIZE).strip()
                # HandleResult = vtXXX.handle(self.data.decode(), self)
                HandleResult = vtXXX.handle(self.data, self)
                if HandleResult == -1:  # stop the loop, shutdown this session,
                    print('Session ended.')
                    break             # and wait for another round of connection.
    
        def setup(self):  # called in __init()__
            self.sockstate = 1
    
        def finish(self):  # called after handle() finishes
            self.sock_state = 0
            self.request.shutdown(2)  # socket.SHUT_RDWR
        
    print('listening on: {}'.format((HOST, PORT)))
    # Create the server, binding to localhost on port 9999
    CommServer = vt_comm.VT_CommServer((HOST, PORT), VT_Handler)

    # When the following line is executed, the handler class is instantiated
    # at the same time setup() -> handle() -> finish() methods are called.
    CommServer.server.serve_forever()  # handle_request()
