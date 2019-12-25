#!/usr/bin/env python

'''
This script is only intended to be used when there is no actual hardware board.
It mimic the behavior of mz7030fa hardware board, which generates constant
frame stream at 8fps/16fps/24fps speed.

'''

# Python 2/3 compatibility
from __future__ import print_function
import numpy as np
import cv2 as cv
import re
from numpy import pi, sin, cos
from os import getcwd
import socket
from time import sleep
from multiprocessing import Process, Queue, Value
from struct import unpack
from select import select

currdir = getcwd()
lena_path = currdir+'\\FogDemoData\\lena.jpg'
def anorm2(a):
    return (a*a).sum(-1)
def anorm(a):
    return np.sqrt( anorm2(a) )

def lookat(eye, target, up = (0, 0, 1)):
    fwd = np.asarray(target, np.float64) - eye
    fwd /= anorm(fwd)
    right = np.cross(fwd, up)
    right /= anorm(right)
    down = np.cross(fwd, right)
    R = np.float64([right, down, fwd])
    tvec = -np.dot(R, eye)
    return R, tvec

def mtx2rvec(R):
    w, u, vt = cv.SVDecomp(R - np.eye(3))
    p = vt[0] + u[:,0]*w[0]    # same as np.dot(R, vt[0])
    c = np.dot(vt[0], p)
    s = np.dot(vt[1], p)
    axis = np.cross(vt[0], vt[1])
    return axis * np.arctan2(s, c)

defaultSize = 512

class TestSceneRender():

    def __init__(self, bgImg = None, fgImg = None,
        deformation = False, speed = 0.25, **params):
        self.time = 0.0
        self.timeStep = 1.0 / 30.0
        self.foreground = fgImg
        self.deformation = deformation
        self.speed = speed

        if bgImg is not None:
            self.sceneBg = bgImg.copy()
        else:
            self.sceneBg = np.zeros(defaultSize, defaultSize, np.uint8)

        self.w = self.sceneBg.shape[0]
        self.h = self.sceneBg.shape[1]

        if fgImg is not None:
            self.foreground = fgImg.copy()
            self.center = self.currentCenter = (int(self.w/2 - fgImg.shape[0]/2), int(self.h/2 - fgImg.shape[1]/2))

            self.xAmpl = self.sceneBg.shape[0] - (self.center[0] + fgImg.shape[0])
            self.yAmpl = self.sceneBg.shape[1] - (self.center[1] + fgImg.shape[1])

        self.initialRect = np.array([ (self.h/2, self.w/2), (self.h/2, self.w/2 + self.w/10),
         (self.h/2 + self.h/10, self.w/2 + self.w/10), (self.h/2 + self.h/10, self.w/2)]).astype(int)
        self.currentRect = self.initialRect

    def getXOffset(self, time):
        return int( self.xAmpl*cos(time*self.speed))


    def getYOffset(self, time):
        return int(self.yAmpl*sin(time*self.speed))

    def setInitialRect(self, rect):
        self.initialRect = rect

    def getRectInTime(self, time):

        if self.foreground is not None:
            tmp = np.array(self.center) + np.array((self.getXOffset(time), self.getYOffset(time)))
            x0, y0 = tmp
            x1, y1 = tmp + self.foreground.shape[0:2]
            return np.array([y0, x0, y1, x1])
        else:
            x0, y0 = self.initialRect[0] + np.array((self.getXOffset(time), self.getYOffset(time)))
            x1, y1 = self.initialRect[2] + np.array((self.getXOffset(time), self.getYOffset(time)))
            return np.array([y0, x0, y1, x1])

    def getCurrentRect(self):

        if self.foreground is not None:

            x0 = self.currentCenter[0]
            y0 = self.currentCenter[1]
            x1 = self.currentCenter[0] + self.foreground.shape[0]
            y1 = self.currentCenter[1] + self.foreground.shape[1]
            return np.array([y0, x0, y1, x1])
        else:
            x0, y0 = self.currentRect[0]
            x1, y1 = self.currentRect[2]
            return np.array([x0, y0, x1, y1])

    def getNextFrame(self):
        img = self.sceneBg.copy()

        if self.foreground is not None:
            self.currentCenter = (self.center[0] + self.getXOffset(self.time), self.center[1] + self.getYOffset(self.time))
            img[self.currentCenter[0]:self.currentCenter[0]+self.foreground.shape[0],
             self.currentCenter[1]:self.currentCenter[1]+self.foreground.shape[1]] = self.foreground
        else:
            self.currentRect = self.initialRect + np.int( 30*cos(self.time*self.speed) + 50*sin(self.time*self.speed))
            if self.deformation:
                self.currentRect[1:3] += int(self.h/20*cos(self.time))
            cv.fillConvexPoly(img, self.currentRect, (0, 0, 255))

        self.time += self.timeStep
        return img

    def resetTime(self):
        self.time = 0.0

class VideoSynthBase(object):
    def __init__(self, size=None, noise=0.0, bg = None, **params):
        self.bg = None
        self.frame_size = (640, 480)
        if bg is not None:
            self.bg = cv.imread(cv.samples.findFile(bg))
            h, w = self.bg.shape[:2]
            self.frame_size = (w, h)

        if size is not None:
            w, h = map(int, size.split('x'))
            self.frame_size = (w, h)
            self.bg = cv.resize(self.bg, self.frame_size)

        self.noise = float(noise)

    def render(self, dst):
        pass

    def read(self, dst=None):
        w, h = self.frame_size

        if self.bg is None:
            buf = np.zeros((h, w, 3), np.uint8)
        else:
            buf = self.bg.copy()

        self.render(buf)

        if self.noise > 0.0:
            noise = np.zeros((h, w, 3), np.int8)
            cv.randn(noise, np.zeros(3), np.ones(3)*255*self.noise)
            buf = cv.add(buf, noise, dtype=cv.CV_8UC3)
        return True, buf

    def isOpened(self):
        return True

class Book(VideoSynthBase):
    def __init__(self, **kw):
        super(Book, self).__init__(**kw)
        backGr = cv.imread(cv.samples.findFile('graf1.png'))
        fgr = cv.imread(cv.samples.findFile('box.png'))
        self.render = TestSceneRender(backGr, fgr, speed = 1)

    def read(self, dst=None):
        noise = np.zeros(self.render.sceneBg.shape, np.int8)
        cv.randn(noise, np.zeros(3), np.ones(3)*255*self.noise)

        return True, cv.add(self.render.getNextFrame(), noise, dtype=cv.CV_8UC3)

class Cube(VideoSynthBase):
    def __init__(self, **kw):
        super(Cube, self).__init__(**kw)
        self.render = TestSceneRender(cv.imread(cv.samples.findFile('pca_test1.jpg')), deformation = True,  speed = 1)

    def read(self, dst=None):
        noise = np.zeros(self.render.sceneBg.shape, np.int8)
        cv.randn(noise, np.zeros(3), np.ones(3)*255*self.noise)

        return True, cv.add(self.render.getNextFrame(), noise, dtype=cv.CV_8UC3)

class Chess(VideoSynthBase):
    def __init__(self, **kw):
        super(Chess, self).__init__(**kw)

        w, h = self.frame_size

        self.grid_size = sx, sy = 10, 7
        white_quads = []
        black_quads = []
        for i, j in np.ndindex(sy, sx):
            q = [[j, i, 0], [j+1, i, 0], [j+1, i+1, 0], [j, i+1, 0]]
            [white_quads, black_quads][(i + j) % 2].append(q)
        self.white_quads = np.float32(white_quads)
        self.black_quads = np.float32(black_quads)

        fx = 0.9
        self.K = np.float64([[fx*w, 0, 0.5*(w-1)],
                        [0, fx*w, 0.5*(h-1)],
                        [0.0,0.0,      1.0]])

        self.dist_coef = np.float64([-0.2, 0.1, 0, 0])
        self.t = 0

    def draw_quads(self, img, quads, color = (0, 255, 0)):
        img_quads = cv.projectPoints(quads.reshape(-1, 3), self.rvec, self.tvec, self.K, self.dist_coef) [0]
        img_quads.shape = quads.shape[:2] + (2,)
        for q in img_quads:
            cv.fillConvexPoly(img, np.int32(q*4), color, cv.LINE_AA, shift=2)

    def get(self, propID):
    	if (propID==cv.CAP_PROP_FRAME_WIDTH):
    		return self.frame_size[0]
    	elif (propID==cv.CAP_PROP_FRAME_HEIGHT):
    		return self.frame_size[1]
    	else:
    		pass

    def render(self, dst):
        t = self.t
        self.t += 1.0/30.0

        sx, sy = self.grid_size
        center = np.array([0.5*sx, 0.5*sy, 0.0])
        phi = pi/3 + sin(t*3)*pi/8
        c, s = cos(phi), sin(phi)
        ofs = np.array([sin(1.2*t), cos(1.8*t), 0]) * sx * 0.2
        eye_pos = center + np.array([cos(t)*c, sin(t)*c, s]) * 15.0 + ofs
        target_pos = center + ofs

        R, self.tvec = lookat(eye_pos, target_pos)
        self.rvec = mtx2rvec(R)

        self.draw_quads(dst, self.white_quads, (245, 245, 245))
        self.draw_quads(dst, self.black_quads, (10, 10, 10))


classes = dict(chess=Chess, book=Book, cube=Cube)

presets = dict(
    empty = 'synth:',
    lena = 'synth:bg=lena.jpg:noise=0.1',
    chess = 'synth:class=chess:bg=lena.jpg:noise=0.1:size=640x480',
    book = 'synth:class=book:bg=graf1.png:noise=0.1:size=640x480',
    cube = 'synth:class=cube:bg=pca_test1.jpg:noise=0.0:size=640x480'
)


def create_capture(source = 0, fallback = presets['chess'], size = (640,480)):
    '''source: <int> or '<int>|<filename>|synth [:<param_name>=<value> [:...]]'
    '''
    source = str(source).strip()

    # Win32: handle drive letter ('c:', ...)
    source = re.sub(r'(^|=)([a-zA-Z]):([/\\a-zA-Z0-9])', r'\1?disk\2?\3', source)
    chunks = source.split(':')
    chunks = [re.sub(r'\?disk([a-zA-Z])\?', r'\1:', s) for s in chunks]

    source = chunks[0]
    try: source = int(source)
    except ValueError: pass
    params = dict( s.split('=') for s in chunks[1:] )
    print('creating from:%s'%source)
    cap = None
    if source == 'synth':
        Class = classes.get(params.get('class', None), VideoSynthBase)
        try: cap = Class(**params)
        except: pass
    else:
        cap = cv.VideoCapture(source)
        w,h = size
        cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
        #if 'size' in params:
            #w, h = map(int, params['size'].split('x'))
            #cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
            #cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)
        if fallback is not None:
            return create_capture(fallback, None)
    return cap

def commandProcess(sock, q, command_state, stopflag):
    ''' run as a child process that receives command from remote client.
    Communicate with the parent process by putting command into queue (q)
    and set status flag in command_state.value
    '''
    commandlen = 4 # 4 bytes
    data = b''
    while True:
        if (stopflag.value == 1):
            break
        (ready, [], []) = select([sock],[],[],0.1)
        if ready:
            try:
                newbyte = sock.recv(1)
            except OSError:
                newbyte = None
            if newbyte:
                data += newbyte
            else: # nothing received, meaning client closed
                stopflag.value = 1
                break
        if len(data)==commandlen:
            (command,) = unpack('=L',data)
            print('backendprocess : received command {}'.format(command))
            q.put(command)
            data = b''
            command_state.value = 1 # indicate parent proc to read queue
            # wait for the parent process to get command and clear flag
            if (command_state.value != 0):
                sleep(0.1)

commandset={'terminate' :   21000010,
            'endless'   :   21000000,
            'fps8'      :   21000001,
            'fps16'     :   21000002,
            'fps24'     :   21000003}

class boardSimulator:
    def __init__(self, videocap, serverip, tcpport=1069, fps=24):
        self.cap = videocap
        self.fps = fps
        self.IP = serverip
        self.Port = tcpport
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
        self.sock.bind((self.IP, self.Port))
        self.sock.listen(1)
        print('Listening at', self.sock.getsockname())
        
    def run(self):
        print('start running board simulation')
        while True:
            n_frame_to_send = 0
            sc = self.waitconnection() # blocking operation
            self.q = Queue()
            self.commandstate = Value('i', 0)
            self.stopflag = Value('i', 0)
            w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            self.command_process = Process(target=commandProcess, args=(sc, self.q, self.commandstate, self.stopflag))
            self.command_process.start()
            while True:
                if (self.commandstate.value == 1):
                    command = self.q.get()
                    if (command == commandset['terminate']):
                        n_frame_to_send = 0
                    elif (command ==  commandset['fps8']):
                        print('Set fps to 8.')
                        self.fps = 8
                    elif (command == commandset['fps16']):
                        print('Set fps to 16.')
                        self.fps = 16                        
                    elif (command == commandset['fps24']):
                        print('Set fps to 24.')
                        self.fps = 24
                    elif (command == commandset['endless']):
                        print('sending endless frames')
                        n_frame_to_send = 100000000
                    else:
                        n_frame_to_send = command
                        print('send {} frames'.format(command))
                    self.commandstate.value = 0 # clear the state flag
                
                if (self.stopflag.value==1):  # if the client closed connection
                    print('Client closed session!!')
                    self.command_process.join()
                    print('return to socket listening')
                    sc.close()
                    break
                
                if (n_frame_to_send>0):
                    ret, img = self.cap.read()
                    data_to_send = img[:,:,:1].reshape((480,640)).data.tobytes()
                    try:
                        sc.sendall(data_to_send)
                    except BrokenPipeError as err:
                        print(err, 'Maybe clinet socket has stopped.')
                        self.stopflag.value=1
                    n_frame_to_send = n_frame_to_send-1
                    sleep(1/(self.fps+1))

                    
    def waitconnection(self):
        sc, sockname = self.sock.accept()
        print('connected to:{}'.format(sc.getpeername()))
        return sc
        
if __name__ == '__main__':
    import sys
    import getopt

    print(__doc__)

    args, sources = getopt.getopt(sys.argv[1:], '', 'shotdir=')
    print('args=', args)
    print('sources=', sources)
    args = dict(args)
    shotdir = args.get('--shotdir', '.')
    if len(sources) == 0:
        sources = [ 0 ]

#    caps = list(map(create_capture, sources))
#    shot_idx = 0
#    while True:
#        imgs = []
#        for i, cap in enumerate(caps):
#            ret, img = cap.read()
#            imgs.append(img)
#            cv.imshow('capture %d' % i, img)
#        ch = cv.waitKey(1)
#        if ch == 27:
#            break
#        if ch == ord(' '):
#            for i, img in enumerate(imgs):
#                fn = '%s/shot_%d_%03d.bmp' % (shotdir, i, shot_idx)
#                cv.imwrite(fn, img)
#                print(fn, 'saved')
#            shot_idx += 1
#    cv.destroyAllWindows()
    
    cap = create_capture(fallback='synth:class=chess:bg=lena.jpg:noise=0.0:size=640x480')
    xxx = boardSimulator(cap, '10.242.13.93')
    #xxx = boardSimulator(cap, '192.168.1.3')
    xxx.run()
