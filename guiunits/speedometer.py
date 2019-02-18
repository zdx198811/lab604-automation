# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 10:47:43 2019

@author: dongxucz
"""

from qtpy.QtCore import Qt, QPoint, QRectF, QPointF, QSize
from qtpy.QtGui import (QColor, QConicalGradient, QPainterPath,
        QPainter, QFont, QFontMetrics)
from qtpy.QtWidgets import QApplication, QWidget, QSizePolicy

class Speedometer(QWidget):
    """QWidget of a Speedometer
    Use `setSpeed(speed)` method to update display.
    """
    def __init__(self, title, unit, min_value, max_value, init_value=None, parent=None):
        QWidget.__init__(self, parent)
        self.min_value = min_value
        self.max_value = max_value
        initv = 0
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
    
    def paintEvent(self, evt):
        x1 = QPoint(0, -70)
        x2 = QPoint(0, -90)
        x3 = QPoint(-90,0)
        x4 = QPoint(-70,0)
        extRect = QRectF(-90,-90,180,180)
        intRect = QRectF(-70,-70,140,140)
        midRect = QRectF(-44,-80,160,160)
        unitRect = QRectF(-44,60,110,50)

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

        leftPos = -1 * speedWidth + 50
        leftDecPos = leftPos + speedWidth
        topPos = 10
        topDecPos = 10
        painter.setPen(self.speedTextColor)
        painter.setFont(speedFont)
        painter.drawText(leftPos, topPos, s_SpeedInt)

if __name__ == '__main__':
    import sys
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QVBoxLayout, QPushButton
    from numpy.random import randn
    app = QApplication(sys.argv)
    meter = Speedometer('Speedometer', 'Km/h', 0, 100)
    steps = 60
    coef = 90/sum([1/i for i in range(1,steps+1)])
    
    def initSpeed(): # count=1, interval=100
        current_speed = 0
        counter = 1
        timer = QTimer()
        def helloworld():
            print('hello',meter.speed)
            step = randn()/10
            if ((meter.speed + step) >= 100) or ((meter.speed + step) <=80):
                step = step * (-1)
            meter.setSpeed(meter.speed + step)
        def handler():
            nonlocal current_speed
            nonlocal counter
            meter.setSpeed(current_speed)
            current_speed = current_speed + 19.231301736672293/counter
            print(counter)
            counter = counter + 1
            if counter >= 61:
                #timer.stop()
                #timer.deleteLater()
                timer.timeout.disconnect(handler)
                timer.timeout.connect(helloworld)
        timer.timeout.connect(handler)
        timer.start(10)

    topbutton = QPushButton('start')
    topbutton.resize(150 ,  80)
    topbutton.clicked.connect(initSpeed)
    
    window = QWidget()
    window.resize(460,  500)
    layout = QVBoxLayout()
    layout.addWidget(topbutton)
    layout.addWidget(meter)
    window.setLayout(layout)

    #window.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
    window.show()
    sys.exit(app.exec_()) 
    
    
    