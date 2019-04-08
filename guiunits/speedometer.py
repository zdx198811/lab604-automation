# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:47:43 2019

@author: dongxucz
"""

from PyQt5.QtCore import Qt, QPoint, QRectF, QPointF, QSize, pyqtProperty, QPropertyAnimation
from PyQt5.QtGui import (QColor, QConicalGradient, QPainterPath,
        QPainter, QFont, QFontMetrics)
from PyQt5.QtWidgets import QApplication, QWidget, QSizePolicy

class Speedometer(QWidget):
    """QWidget of a Speedometer
    Use `setSpeed(speed)` method to update display.
    Use `reset()` to reset back to 0 with an animation effect.
    """
    def __init__(self, title, unit, min_value, max_value, init_value=None, parent=None):
        QWidget.__init__(self, parent)
        self.min_value = min_value
        self.max_value = max_value
        initv = 0.0
        if init_value:
            if init_value < 0:
                initv = 0
            elif init_value > max_value:
                initv = max_value
            else:
                initv = init_value
        self.speed = initv
        self.displayPowerPath = True
        self.title = title
        self.power = 100.0 * (self.speed-self.min_value)/(self.max_value-self.min_value)
        self.powerGradient =  QConicalGradient(0, 0, 180)
        self.powerGradient.setColorAt(0, Qt.red)
        self.powerGradient.setColorAt(0.375, Qt.yellow)
        self.powerGradient.setColorAt(0.75, Qt.green)
        self.unitTextColor = QColor(Qt.gray)
        self.speedTextColor = QColor(Qt.black)
        self.powerPathColor = QColor(Qt.gray)
        self.unit = unit
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        
        self.anim_reset = QPropertyAnimation(self, b"value")
        self.anim_reset.setDuration(500)
        self.anim_reset.setEndValue(0)
    
    def setSpeed(self, speed):
        self.speed = speed
        self.power = 100.0 * (self.speed-self.min_value)/(self.max_value-self.min_value)
        self.update()

    def setUnit(self, unit):
        self.unit = unit

    def setPowerGradient(self, gradient):
        self.powerGradient = gradient

    def setDisplayPowerPath(self, displayPowerPath):
        self.displayPowerPath = displayPowerPath

    def setUnitTextColor(self, color):
        self.unitTextColor = color

    def setSpeedTextColor(self, color):
        self.speedTextColor = color

    def setPowerPathColor(self, color):
        self.powerPathColor = color

    def sizeHint(self):
        return QSize(100,100)
    
    def reset(self):
        self.anim_reset.setStartValue(self.speed)
        self.anim_reset.start()
        
    def paintEvent(self, evt):
        x1 = QPoint(0, -70)
        x2 = QPoint(0, -90)
        x3 = QPoint(-90,0)
        x4 = QPoint(-70,0)
        extRect = QRectF(-90,-90,180,180)
        intRect = QRectF(-70,-70,140,140)
        midRect = QRectF(-44,-80,160,160)
        unitRect = QRectF(-50,60,110,50)

        speedInt = self.speed
        #speedDec = (self.speed * 10.0) - (speedInt * 10)
        s_SpeedInt = speedInt.__str__()[0:4]

        powerAngle = self.power * 270.0 / 100.0

        dummyPath = QPainterPath()
        dummyPath.moveTo(x1)
        dummyPath.arcMoveTo(intRect, 90 - powerAngle)
        powerPath = QPainterPath()
        powerPath.moveTo(x1)
        powerPath.lineTo(x2)
        powerPath.arcTo(extRect, 90, -1 * powerAngle)
        powerPath.lineTo(dummyPath.currentPosition())
        powerPath.arcTo(intRect, 90 - powerAngle, powerAngle)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        side = min(self.width(), self.height())
        painter.scale(side / 200.0, side / 200.0)

        painter.save()
        painter.rotate(-135)

        if self.displayPowerPath:
            externalPath = QPainterPath()
            externalPath.moveTo(x1)
            externalPath.lineTo(x2)
            externalPath.arcTo(extRect, 90, -270)
            externalPath.lineTo(x4)
            externalPath.arcTo(intRect, 180, 270)

            painter.setPen(self.powerPathColor)
            painter.drawPath(externalPath)

        painter.setBrush(self.powerGradient)
        painter.setPen(Qt.NoPen)
        painter.drawPath(powerPath)
        painter.restore()
        painter.save()

        painter.translate(QPointF(0, -50))

        painter.setPen(self.unitTextColor)
        fontFamily = self.font().family()
        unitFont = QFont(fontFamily, 9)
        painter.setFont(unitFont)
        painter.drawText(unitRect, Qt.AlignCenter, "{}".format(self.unit))

        painter.restore()

        painter.setPen(self.unitTextColor)
        fontFamily = self.font().family()
        unitFont = QFont(fontFamily, 12)
        painter.setFont(unitFont)
        painter.drawText(unitRect, Qt.AlignCenter, "{}".format(self.title))

        speedColor = QColor(0,0,0)
        speedFont = QFont(fontFamily, 30)
        fm1 = QFontMetrics(speedFont)
        speedWidth = fm1.width(s_SpeedInt)

        #speedDecFont = QFont(fontFamily, 23)
        #fm2 = QFontMetrics(speedDecFont)
        #speedDecWidth = fm2.width(s_SpeedDec)

        leftPos = -1 * speedWidth + 40
        leftDecPos = leftPos + speedWidth
        topPos = 10
        topDecPos = 10
        painter.setPen(self.speedTextColor)
        painter.setFont(speedFont)
        painter.drawText(leftPos, topPos, s_SpeedInt)

    value = pyqtProperty(float, fset=setSpeed)

if __name__ == '__main__':
    import sys
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QVBoxLayout, QPushButton
    from numpy.random import randn
    app = QApplication(sys.argv)
    meter = Speedometer('Speedometer', 'Km/h', 0, 100)
    steps = 60
    coef = 90/sum([1/i for i in range(1,steps+1)])
    global timer
    timer = QTimer()
    timer.setInterval(20)
    global status
    status = 'off'
    startbutton = QPushButton('Start')
    startbutton.resize(150 ,  80)
    anim_start = QPropertyAnimation(meter, b"value")
    anim_start.setDuration(500)
    anim_start.setStartValue(0)
    anim_start.setEndValue(80)
    anim_start.finished.connect(timer.start)

    def random_speed_jitter():
        # print('hello',meter.speed)
        step = randn()/10
        if ((meter.speed + step) >= 100) or ((meter.speed + step) <=80):
            step = step * (-1)
        meter.setSpeed(meter.speed + step)

    def swap_status():
        global status
        global timer
        if (status == 'off'):
            status = 'on'
            startbutton.setText('Stop')
            anim_start.start()  # timer will start after anim_start finishes
        else:
            status = 'off'
            timer.stop()
            startbutton.setText('Start')
            meter.reset()
    
    timer.timeout.connect(random_speed_jitter)
    startbutton.clicked.connect(swap_status)
    
    window = QWidget()
    window.resize(460,  500)
    layout = QVBoxLayout()
    layout.addWidget(startbutton)
    layout.addWidget(meter)
    window.setLayout(layout)

    #window.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
    window.show()
    sys.exit(app.exec_()) 
    
    
    