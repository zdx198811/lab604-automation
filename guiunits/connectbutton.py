from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import pyqtSignal, QObject

class SigWrapper(QObject):
    click_connect_sgnl = pyqtSignal(str)

class ConnectBtn(QPushButton):
    
    def __init__(self, addrQLineEdit):
        super(ConnectBtn, self).__init__("Connect")
        # Added a signal
        self.addrQLineEdit = addrQLineEdit
        self.signal_wraper = SigWrapper()
        self.clicked.connect(self.send_str_on_clicked)
        
    def send_str_on_clicked_test(self):
        self.signal_wraper.click_connect_sgnl.emit('clicked\n')
    
    def send_str_on_clicked(self):
        '''
        addrline should be the QLineEdit object that contains user inputed
        backend IP address.
        '''
        txt = 'connect to' + self.addrQLineEdit.text() + '\n'
        self.signal_wraper.click_connect_sgnl.emit(txt)
