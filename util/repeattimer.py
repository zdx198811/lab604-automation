# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:55:50 2020
Description:
    Repeating timer.
@author: dongxucz
"""
from threading import Thread, Timer, Lock, Event

class Periodic(object):
    """
    A utility to run some function periodically. Using threading.Timer.
    https://stackoverflow.com/questions/2398661/schedule-a-repeating-event-in-python-3
    
    Instanciate an object and call the start() method to run function periodically.
    __init__() Parameters:
        interval : int
            repeating period.
        function : callable
            the function to be repeated.
        *args, **kwargs : 
            arguments passed to function.
        a special keyword arg is 'autostart'. If True, the timer start immediately.
    """

    def __init__(self, interval, function, *args, **kwargs):
        self._lock = Lock()
        self._timer = None
        self.function = function
        self.interval = interval
        self.args = args
        self.kwargs = kwargs
        self._stopped = True
        if kwargs.pop('autostart', False):  # pop it out before passing to function
            self.start()

    def start(self, from_run=False):
        self._lock.acquire()
        if from_run or self._stopped:
            self._stopped = False
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self._lock.release()

    def _run(self):
        self.start(from_run=True)
        self.function(*self.args, **self.kwargs)

    def stop(self):
        self._lock.acquire()
        self._stopped = True
        if self._timer:
            self._timer.cancel()
        self._lock.release()

class repeatTimer(Thread):
    """Call a function after a specified number of seconds, repeatively:
            t = repeatTimer(30.0, f, args=None, kwargs=None, autostart=True)
            t.start()
            t.cancel()     # stop the timer's action
            t.stop()       # == t.cancel()
            t.pause()      # still alive but do not excecute function
            t.resume()     # resume from pause
    """

    def __init__(self, interval, function, args=None, kwargs=None, autostart=True):
        Thread.__init__(self)
        self.interval = interval
        self.function = function
        self.autostart = autostart
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}
        self.finished = Event()
        self._pause = False
        if self.autostart:
            self.start()

    def cancel(self):
        """Stop the timer."""
        self.finished.set()
    
    def stop(self):
        self.cancel()
    
    def pause(self):  # pause
        self._pause = True
        
    def resume(self):
        self._pause = False
    
    def is_runing(self):
        return (not self._pause)
    
    def is_paused(self):
        return self._pause
    
    def run(self):
        while True:
            self.finished.wait(self.interval)
            if not self.finished.is_set():
                if self._pause:
                    pass
                else:
                    self.function(*self.args, **self.kwargs)
            else:
                break


if __name__ == '__main__':

    from time import sleep
    ###################### the following lines print hello world per second
    print('the first demo:\n')
    def hello(greeting, name):
        print(f'{greeting}, {name}!')
    
    #rt = Periodic(1, hello, "World", autostart=False)  # default is autostart=True
    rt = repeatTimer(1, hello, args=["World", "world"], autostart=False)  # default is autostart=True
    rt.start()
    
    try:
        sleep(5) # your long-running job goes here...
    finally:
        rt.stop() # better in a try/finally block to make sure the program ends!
        
    ###################### the following demo operate on a thread-safe var
    print('\nThe second demo:\n')
    from random import randint
    from threadsafevar import threadsafeVar
    testvalue = threadsafeVar()
    
    def print_and_clear(v):  # this will be run in another thread periodically
        print(f"current value is {v.readvalue()}")
        v.updatevalue(0)
        print(f"cleared! \n{'_'*20}")
        
    #rt = Periodic(1, print_and_clear, testvalue, autostart=True)  # start to print and clear per second
    rt = repeatTimer(1, print_and_clear, args=[testvalue], autostart=True)  # start to print and clear per second
    
    for i in range(100):  # add to testvalue 100 times in about 15s time.
        randi = randint(1, 10)
        print(f'adding {randi}')
        testvalue.updatevalue(randi, op='add')
        if i==50:
            print("pause for 2 seconds...")
            rt.pause()
            sleep(2)
            rt.resume()
        sleep(0.15)
    
    rt.stop()