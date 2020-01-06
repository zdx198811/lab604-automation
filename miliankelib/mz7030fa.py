# -*- coding: utf-8 -*-
"""
Discription:
    This module provides APIs for the 米联客 MZ7030FA and MZ7XA-7010 boards.
    There are 3 different types of usage, including:
        1. direct board test; 
        2. remote pipe server;
        3. application interface.
    Refer to the readme.md file for more detailed discription.

Created on Nov. 29 2019
@author: dongxucz (dongxu.c.zhang@nokia-sbell.com)
"""

import cv2 as cv
import numpy as np
import socket
import struct
from select import select
from multiprocessing import Process, Queue, Value, Manager
from time import sleep
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod

#Manager

_BKND_UNDEFINED = 0
_BKND_LOCAL_PIPE = 1
_BKND_REMOTE_PIPE = 2
_BKND_WEB = 3
   
commandset={'terminate' :   21000010,
            'endless'   :   21000000,
            'fps8'      :   21000001,
            'fps16'     :   21000002,
            'fps24'     :   21000003}


def _send_int_to_camera(sock, n):
    ''' send an integer to the camera board.
    '''
    sock.sendall(struct.pack("L", n))

def empty_socket(sock):
    """remove the data present on the socket"""
    print('clearing residual socket bytes')
    while 1:
        inputready, _, __ = select([sock],[],[], 5)
        if len(inputready)==0: break
        for s in inputready: s.recv(1024)
    
def _recvframes(sock, n_frames, frame_bytes):
    data = b''
    frames = []
    for _ in range(n_frames):
        while len(data) < frame_bytes:
            more = sock.recv(frame_bytes - len(data))
            #print("frame %d received %d bytes." % (f+1, len(more)))
            if not more:
                raise EOFError('was expecting %d bytes but only received'
                               ' %d bytes before the socket closed'
                                % (frame_bytes, len(data)))
            data += more
            #print("total received now: %d." % (len(data)))
        frames.append(data)
        #for row in range(VSIZE):
        #   start = row*HSIZE
        #   frame.append(struct.unpack(str(HSIZE)+'B', data[start:start+HSIZE]))
        data = b''
    return frames

def _frame_buffering_process(sock, q, command_id, fsize, q_size = 1):
    ''' The child process which receives frames into a queue, and send commands
    to the camera board.
    Args:
        sock - a connected socket to the board
        q - a Queue (multiprocess FIFO) as frame buffer
        command_id - a Value (multiprocess) sending to the board
        fsize - total number of bits per frame
        q_size - size of the frame buffer
        
    Supported command_id values:
        0~20999999 : number of frames to transfer
        21000000 : endless mode
        21000001 : switch frame/second to 8
        21000002 : switch frame/second to 16
        21000003 : switch frame/second to 24
        21000010 : stop transferring
    '''
    #print('into subprocess')
    data = b''
    while True:
        if (command_id.value != 0):
            if (command_id.value == commandset['terminate']):
                _send_int_to_camera(sock, 21000010)
                command_id.value = 0
                print('exiting command processing process')
                break
            elif (command_id.value == commandset['fps8']):
                _send_int_to_camera(sock, 21000001)
            elif (command_id.value == commandset['fps16']):
                _send_int_to_camera(sock, 21000002)
            elif (command_id.value == commandset['fps24']):
                _send_int_to_camera(sock, 21000003)
            elif (command_id.value == commandset['endless']):
                _send_int_to_camera(sock, 21000000)
            else:
                print('unsuported command!')
            command_id.value = 0
        else:
            pass
        
        (ready, [], []) = select([sock],[],[],0)
        if ready:
            # print('Receiving packets')
            while (len(data) < fsize):
                if ((fsize - len(data)) >= 4096):
                    more = sock.recv(4096)
                else:
                    more = sock.recv(fsize - len(data))
                #print("frame %d received %d bytes." % (f+1, len(more)))
                if not more:
                    raise EOFError('was expecting %d bytes but only received'
                                   ' %d bytes before the socket closed'
                                    % (fsize, len(data)))
                data += more
                #print("total received now: %d." % (len(data)))
        if data:
            if not q.full():
                q.put(data)
        data = b''

class VideoCapBase(metaclass=ABCMeta):
    ''' abstract class. Mimics OpenCV's VideoCapture API
        read() method is mandatory.
    '''
    def __init__(self, src, size, fps = -1, **kwargs):
        self.src = src
        self.is_opened = False
        self.fps = fps
        self._backend_type = _BKND_UNDEFINED
        self.frame_size = size
        self._Wd = size[0] # frame width
        self._Ht = size[1] # frame hight

    @abstractmethod
    def read(self):
        pass

    def get_backend(self):
        return ['Undefined', 'LocalPipe', 'RemotePipe', 'Web'][self._backend_type]


class Mz7030faMt9v034Cap(VideoCapBase):
    def __init__(self, src=('192.168.1.10', 1069), size=(640,480), mode = 'direct', maxbuf=3, **kwargs):
        ''' 
        positional argument:

        keyword arguments:
            src - a tuple (ip, tcp), or a URL
            size - a tuple (w, h) where w and h are integers (# of pixles)
            mode - can be 'direct','server','app'
            fps - integer frame/second
        '''
        super(Mz7030faMt9v034Cap, self).__init__(src, size, **kwargs)
        if type(src)==tuple: # pipe
            assert(type(src[0])==str and type(src[1])==int)
            self._sock = None
            if (mode=='app'):
                self._backend_type = _BKND_REMOTE_PIPE
                print('remote pipe backend')
                self._Open_RP()
            elif (mode=='direct'):
                self._frame_buffering_proc = None # frame buffer process
                self._q = None # frame FIFO (multi-process interface)
                self._command = Value('i', 0)
                self._backend_type = _BKND_LOCAL_PIPE
                self._maxbuf = maxbuf
                print('local pipe backend')
                self._Open_LP()
            else:
                assert(mode=='server')
                pass
        elif type(src)==str: # url
            raise ValueError('not supported yet')
        else:
            raise ValueError('wrong src argument format')
        self.is_opened = True
    
    def _socket_connect(self):
        try:
            self._sock.connect(self.src)
        except ConnectionRefusedError as err:
            print(type(err), str(err))
            self._sock.close()
            raise ValueError("{}. Tip: Check the address2".format(str(err)))
        except OSError as err:
            msg = "{}.\nTip: Check the address1".format(str(err))
            self._sock.close()
            raise ValueError(msg)
        print("Connected to !! {}:{}".format(self.src[0], str(self.src[1])))
        return True
    
    def set_fps(self, fps):
        if not (fps in [8, 16, 24]):
            print("fps not supported!")
        else:
            self._set_command_value(commandset['fps'+str(fps)])
    
    def start_stream(self):
        self._set_command_value(commandset['endless'])
    
    def start(self):
        self.start_stream()
    
    def _set_command_value(self, val):
        if self._frame_buffering_proc==None:
            raise ValueError('frame buffering process is not running!')
        else:
            self._command.value = val
            sleep(0.1)
            while (self._command.value != 0):
                # waite for the child process to clear the command state.
                sleep(0.1)
            return None
        
    def get_n_buffed(self):
        ''' return the number of frames buffered '''
        return self._q.qsize()
    
    def stop(self):
        self._Close()
    
    def _Close(self):
        print('closing connection.....')
        self.is_opened = False
        if self._frame_buffering_proc is not None:
            self._set_command_value(commandset['terminate'])
            while not self._q.empty():
                print('clearing queue')
                self._q.get()
                print('ok')
            
            self._frame_buffering_proc.kill()
            print('child process terminated')
            self._frame_buffering_proc = None
        if self._sock is not None:
            empty_socket(self._sock)
            self._sock.shutdown(socket.SHUT_RDWR)
            self._sock.close()
            self._sock = None
        print('connection closed.')
        
    def _test_board_connection(self):
        ''' receive one frame. This function has two purposes, one is to
            verify the board is functioning correctly, the other is to update
            ARP table at both ends to ensure immidiete frame transmission
        '''
        _send_int_to_camera(self._sock, 1)
        frame = _recvframes(self._sock, 1, self._Wd*self._Ht)
        try:
            assert(self._Wd*self._Ht == len(frame[0]))
        except AssertionError:
            self._Close()
            raise ValueError("test board connection failed.\n Frame size doesn't match!")
        del frame
    
    def _Open_RP(self):
        pass
        
    def _Open_LP(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket_connect()
        self._test_board_connection()
        self._q = Queue(maxsize=self._maxbuf)
        fsize_bytes = self._Wd*self._Ht
        self._frame_buffering_proc = Process(target=_frame_buffering_process, args=(self._sock, self._q, self._command, fsize_bytes))
        self._frame_buffering_proc.start()
        
    def read(self):
        frame = []
        if (self._backend_type == _BKND_LOCAL_PIPE):
            framebytes = self._q.get()
            for row in range(self._Ht):
                start = row*self._Wd
                frame.append(struct.unpack(str(self._Wd)+'B', framebytes[start:start+self._Wd]))
            return True, np.array(frame, dtype='uint8')
        else:
            return False, None
    




if __name__ == '__main__':
    import argparse
    modechoices= {'direct', 'app', 'server'}
    parser = argparse.ArgumentParser(description='test mz7030fa board with single mt9v034 camera. ')
    parser.add_argument('-i', type=str, default='192.168.1.10',
                        help='interface the client sends to. (default 192.168.1.10)')
    parser.add_argument('-p', metavar='PORT', type=int, default=1069,
                        help='TCP port (default 1069)')
    parser.add_argument('-m', metavar='MODE', type=str, default='direct',
                        choices=modechoices, help='usage mode: direct (default), server, or app.')
    parser.add_argument('-dir', type=str, default='.',
                        help='directory to save screenshort. (default .)')
    
    parser.add_argument('-t', type=str, default='vid', choices=['vid', 'fig'],
                        help='play video or just show picture. (default vid)')

    args = parser.parse_args()
    
    ipaddr = args.i
    tcpport = args.p
    usagemode = args.m
    shotdir = args.dir
    testtype = args.t

    if (usagemode=='server'):
        # start Flask Web server
        pass

    else:
        mz7030fa = Mz7030faMt9v034Cap(src=(ipaddr,tcpport), mode=usagemode, maxbuf=2)
        mz7030fa.set_fps(16)
        mz7030fa.start_stream()
        shot_idx = 0
        print("Push Space/ESC key to save frame/exit.")
        if (testtype == 'vid'): #
            while True:
                _, img = mz7030fa.read()
                cv.imshow('capture', img)
                ch = cv.waitKey(1)
                if ch == 27:
                    mz7030fa.stop()
                    break
                if ch == ord(' '):
                    fn = '%s/shot_%03d.bmp' % (shotdir, shot_idx)
                    cv.imwrite(fn, img)
                    print(fn, 'saved')
                    shot_idx += 1
            
        else:
            while True:
                n=mz7030fa.get_n_buffed()
                for i in range(n+1):
                    _, img = mz7030fa.read()
                cv.imshow('capture', img)
                ch = cv.waitKey()
                if ch == 27:
                    mz7030fa.stop()
                    break
                if ch == ord(' '):
                    fn = '%s/shot_%03d.bmp' % (shotdir, shot_idx)
                    cv.imwrite(fn, img)
                    print(fn, 'saved')
                    shot_idx += 1
        cv.destroyAllWindows()