#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Author: Dongxu Zhang 
Date createed: 2019-Feb
Description:
    Show 75 led bulbs, indicating 75 fronthaul channels.
"""

import sys
from PyQt5.QtWidgets import (QWidget, QGridLayout, QApplication)
from guiunits.pyqt_led import Led as QLed

class LedPannel(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        
        grid = QGridLayout()
        self.setLayout(grid)
 
        names = list(range(1,76))
        
        positions = [(i,j) for i in range(5) for j in range(15)]
        self.leds = []
        for position, name in zip(positions, names):
            led = QLed(self, on_color=QLed.green, off_color=QLed.red,
                       warning_color=QLed.orange, build='debug')  # set to 'debug' = enabled.
            led.set_shape(QLed.circle)
            led.setText(str(name))
            # led.setFixedSize(80, 50)
            grid.addWidget(led, *position)
            self.leds.append(led)
            
        # self.move(300, 150)
        # self.setWindowTitle('Led Pannel')
        self.show()
        
    def turn_all_on(self):
        for led in self.leds:
            led.turn_on()
    
    def turn_all_off(self):
        for led in self.leds:
            led.turn_off()

    def turn_all_warning(self):
        for led in self.leds:
            led.turn_warning()

    def revert_all(self):
        for led in self.leds:
            led.revert_status()
            

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = LedPannel()
    sys.exit(app.exec_())
    
    
    
    