from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import pyqtSignal, QObject

class SigWrapper(QObject):
    sgnl = pyqtSignal(str)

class ConnectBtn(QPushButton):
    
    def __init__(self):
        super(ConnectBtn, self).__init__("Connect")
        # Added a signal
        self.signal_wraper = SigWrapper()
        self.clicked.connect(self.send_str_on_clicked)
        
    def send_str_on_clicked(self):
        self.signal_wraper.sgnl.emit('clicked\n')
