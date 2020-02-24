# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 10:46:43 2020
Description:
    A simple implementation of state machine.
    ————————————————
    inspired by：https://blog.csdn.net/StoryMonster/article/details/99443480
    
    Based on the original code, I've changed the transaction table's data
    structure from list to dict, which improves readability and probably state
    switching efficiency for large state machines.
    Also, added a print_transaction_table() interface.
    And finally, some other trival robustness improvements.
    
@author: Zhang Dongxu
"""

from abc import ABCMeta, abstractmethod
from collections import namedtuple

class FsmState(metaclass=ABCMeta):
    @abstractmethod
    def enter(self, event, fsm):  # event trigers state transition. fsm is a statemachine
        pass
    
    def exit(self, fsm):
        pass

    def __repr__(self):
        return self.__class__.__name__

class FsmFinalState(FsmState):
    def enter(self, event, fsm):
        pass
    def exit(self, fsm):
        pass

class FsmEvent(metaclass=ABCMeta):
    def __repr__(self):
        return self.__class__.__name__

class FsmException(Exception):
    def __init__(self, description):
        super().__init__(description)

# 一次状态转换的所有要素：上一个状态--事件-->下一个状态

class finiteSM:
    '''
    State machine.
    '''
    def __init__(self, context=None, initState=None):       ## context：状态机上下文
        self.context = context
        self.state_transaction_table = dict()             ## 常规状态转换表
        self.global_transaction_table = dict()            ## 存若干可能的event，遇到后就某种最终状态
        self.current_state = initState() if initState else None
        self.initState = initState
        self.working_state = FsmState
    
    def set_init_state(self, initState):
        self.initState = initState
    
    def reset(self):
        self.current_state.exit(self)
        self.run()
        
    def add_global_transaction(self, event, end_state):     # 全局转换，直接进入到结束状态
        if not issubclass(end_state, FsmFinalState):
            raise FsmException("The state should be FsmFinalState subclass")
        if not issubclass(event, FsmEvent):
            raise FsmException("The envet should be FsmEvent subclass")
        self.global_transaction_table.update({event.__name__ : end_state})
        
    def add_transaction(self, prev_state, event, next_state):
        '''
        Parameters
        ----------
        prev_state : an FsmState subclass.
        event : an FsmEvent subclass.
        next_state : an FsmState subclass.
        
        NOTE: the arguments should be classes instead of instances.
        '''
        if (issubclass(prev_state, FsmState) and issubclass(next_state, FsmState) and issubclass(event, FsmEvent)):
            pass
        else:
            raise FsmException('argument error.')
        
        if issubclass(prev_state, FsmFinalState):
            raise FsmException("It's not allowed to add transaction after Final State Node")
            
        if prev_state.__name__ in self.state_transaction_table:
            self.state_transaction_table[prev_state.__name__].update({event.__name__:next_state})
        else:
            self.state_transaction_table[prev_state.__name__] = {event.__name__:next_state}

    def process_event(self, event_obj):
        """ event is an instance of FsmEvent
        """
        event = str(event_obj)
        if event in self.global_transaction_table:
            self.current_state = self.global_transaction_table[event]()
            self.current_state.enter(event_obj, self)
            self.clear_transaction_table()
            return
        
        try:
            self.current_state.exit(self.context)
            self.current_state = self.state_transaction_table[str(self.current_state)][event]()
            self.current_state.enter(event_obj, self)
            if isinstance(self.current_state, FsmFinalState):
                self.clear_transaction_table()
            return
        except KeyError:
            raise FsmException(f"Transaction not found. state={self.current_state}, event={event}")
        except Exception:
            raise 

    def clear_transaction_table(self):
        self.global_transaction_table = dict()
        self.state_transaction_table = dict()
        self.current_state = None

    def run(self):
        if len(self.state_transaction_table) == 0: raise FsmException("Empty table.")
        if self.initState is not None:
            self.current_state = self.initState()
        else:  # this seems strange but just randomly sets the first state found in transaction table
            self.current_state = list(self.state_transaction_table.values())[0].values()[0]()
        self.current_state.enter(None, self)

    def isRunning(self):
        return self.current_state is not None

    def next_state(self, event_obj):
        ''' check the next state if event happens.
        The parameter event_obj is an instance of FsmEvent
        '''
        event = str(event_obj)
        if event in self.global_transaction_table:
            return self.global_transaction_table[event]
        if event in self.state_transaction_table[str(self.current_state)]:
            return self.state_transaction_table[str(self.current_state)][event]
        else:
            return None
    
    def print_transaction_table(self):
        ''' returns a print() friendly string.'''
        ret = ''
        if self.global_transaction_table :
            ret += 'Global Transaction Table:\nfrom Any working state:\n    '
            
        for event in self.global_transaction_table:
            ret += (event + ' -> ' + self.global_transaction_table[event].__name__ + '\n')
        
        ret += '\nState Transaction Table:\n'
        for currentstateitem in self.state_transaction_table:
            ret += (currentstateitem + ':\n')
            for event in self.state_transaction_table[currentstateitem]:
                ret += ('    ' + event + ' -> ' + self.state_transaction_table[currentstateitem][event].__name__ + '\n')
        return ret
    
if __name__ == '__main__':
    
    # Define your state machine class to suport any customized featrues. This is optional.
    class DVD(finiteSM):
        def __init__(self, *args, **kwargs):
            super().__init__(None, *args, **kwargs)

    # The following define states and describe behaviours when entering or 
    # exiting a state by subclassing `FsmState` and overloading its `enter()` 
    # and `exit()` methods.
            
    class DvdPowerOn(FsmState):
        def enter(self, event, fsm):
            print("dvd is power on")
            
        def exit(self, fsm):
            pass
    
    class DvdPlaying(FsmState):
        def enter(self, event, fsm):
            print("dvd is going to play")
            if hasattr(event, 'replay'):
                while (event.replay>0):
                    print('replaytimes = %d'%event.replay)
                    self.replay()
                    event.replay -= 1
                    
        def exit(self, fsm):
            print("dvd stoped play")
    
        def replay(self):
            print('auto replaying.')
            
    class DvdPausing(FsmState):
        def enter(self, event, fsm):
            print("dvd is going to pause")
    
        def exit(self, fsm):
            print("dvd stopped pause")
    
    class DvdPowerOff(FsmState):
        def enter(self, event, fsm):
            print("dvd is power off")
            
        def exit(self, fsm):
            pass
    
    class DvdExplode(FsmFinalState):
        def enter(self, event, fsm):
            print("dvd exploded!")
            
    
    # The following lines describes events by subclassing `FsmEvent`. You may
    # carry variables with events by defining member variables here
    class PowerOnEvent(FsmEvent):
        pass
    
    class PowerOffEvent(FsmEvent):
        pass
    
    class PlayEvent(FsmEvent):
        def __init__(self, replay = -1):
            self.replay = replay
        pass
    
    class PauseEvent(FsmEvent):
        pass

    class ErrorEvent(FsmEvent):
        pass

    # create a state machine instance
    dvd = DVD(initState=DvdPowerOff);
    
    # add transaction table
    dvd.add_transaction(DvdPowerOff, PowerOnEvent, DvdPowerOn)
    dvd.add_transaction(DvdPowerOn, PowerOffEvent, DvdPowerOff)
    dvd.add_transaction(DvdPowerOn, PlayEvent, DvdPlaying)
    dvd.add_transaction(DvdPlaying, PowerOffEvent, DvdPowerOff)
    dvd.add_transaction(DvdPlaying, PauseEvent, DvdPausing)
    dvd.add_transaction(DvdPausing, PowerOffEvent, DvdPowerOff)
    dvd.add_transaction(DvdPausing, PlayEvent, DvdPlaying)
    dvd.add_global_transaction(ErrorEvent, DvdExplode)
    
   
    # print state transaction table
    print(dvd.print_transaction_table())
    
    # run state machine
    dvd.run()
    
    # now test event triger behaviours
    dvd.process_event(PowerOnEvent())
    dvd.process_event(PlayEvent())
    dvd.process_event(PauseEvent())
    dvd.process_event(PlayEvent(3))
    dvd.process_event(PowerOffEvent())
    dvd.process_event(PowerOnEvent())
    dvd.process_event(ErrorEvent())
    