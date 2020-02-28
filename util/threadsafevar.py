# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:08:07 2020
Thread-safe value wrapper.
@author: dongxucz
"""
from threading import Lock

class threadsafeVar(object):
    """
    A variable wrapper to ensure thread-safe read/write operations.
    Same as `multiprocessing.Value` but support multiple update operations.
    """
    def __init__(self, n=0):
        self.value = n
        self._lock = Lock()
        
    def readvalue(self):
        self._lock.acquire()
        n = self.value
        self._lock.release()
        return n
    
    def updatevalue(self, n, op='set'):
        """
        `n` is an operand. `op` specifies the operation.
        supported operations are: 'set', 'add', 'minus', 'mul', 'div'.
        """
        self._lock.acquire()
        if (op=='set'):
            self.value = n
        elif (op=='add'):
            self.value += n
        elif (op=='minus'):
            self.value -= n
        elif (op=='mul'):
            self.value *= n
        elif (op=='div'):
            self.value /= n
        else:
            self._lock.release()
            raise ValueError(f"unsupported operation:'{op}'")
        self._lock.release()

    def reset(self):
        self.updatevalue(0)


if __name__ == '__main__':
    from repeattimer import repeatTimer
    from random import randint
    from time import sleep
    
    testvalue = threadsafeVar()
    
    def print_and_clear(v):  # this will be run in another thread periodically
        print(f"current value is {v.readvalue()}")
        v.updatevalue(0)
        print(f"cleared! \n{'_'*20}")
        
    rt = repeatTimer(1, print_and_clear, args=[ testvalue])  # start to print and clear per second
    
    for i in range(100):  # add to testvalue 100 times in about 15s time.
        randi = randint(1, 10)
        print(f'adding {randi}')
        testvalue.updatevalue(randi, op='add')
        sleep(0.15)
    
    rt.stop()
    