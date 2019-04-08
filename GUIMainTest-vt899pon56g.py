# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:53:44 2019
Description:
    The GUI app for 56G PON demo. For more information, refer to the 
    corresponding application note in the `lab604-automation` documentation.
@author: dongxucz
"""

import array
import sys
from os import getcwd
from time import sleep
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, \
                QGraphicsPixmapItem, QGraphicsView, QGraphicsItem, \
                QPushButton, QLabel, QWidget, QGraphicsOpacityEffect, \
                QGraphicsTextItem, QTextBrowser, QLineEdit, QGroupBox, \
                QVBoxLayout, QGridLayout
from PyQt5.QtGui import QBrush, QPen, QPainter, QPixmap, QFont, QColor
from PyQt5.QtCore import (Qt, QObject, QPointF, 
        QPropertyAnimation, pyqtProperty)
from vtbackendlib.vt899 import extract_frame, resample_symbols
import numpy as np
import core.vt_device as vt_device
from core.ook_lib import OOK_signal
from core.ks_device import M8195A
from guiunits.connectbutton import ConnectBtn
from guiunits.speedometer import Speedometer
from guiunits.mlpplotwidget import pon56gDemoPlot
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

############## Debugging ##################
_SIM = True
###########################################

VT899Addr = "10.242.13.34", 9998
M8195Addr = "10.242.13.77"
cwd = getcwd()
sample_folder = cwd+'\\vtbackendlib\\0726vt899pon56g\\'

class vtdev(vt_device.VT_Device):
    def __init__(self, devname, addr, frame_len, symbol_rate):
        vt_device.VT_Device.__init__(self, devname)
        self.set_net_addr(addr)
        self.frame_len = frame_len
        self.n_prmbl = 500  # n_payload=195608. see work notebook-2 18-07-05
        self.n_symbol_test = 10000
        self.symbol_rate = symbol_rate
        
        # neural network algorithm related attributes
        self.trainset = OOK_signal()
        self.trainset_rx = np.array([])
        self.neuralnet = self.init_nn(n_tap=15)
        self.label_pos = self.n_tap - 4 # n_tap-1 means the label is the last symbol
        self.max_epoch = 5
        
    def set_preamble(self, preamble_int):
        self.preamble_int = preamble_int
        self.preamble_wave = None
        self.preamble_len = len(preamble_int)

    def prepare_trainloader(self, trainset_rx):
        self.trainset_rx = trainset_rx
        trainset_ref = self.trainset.nrz()
        trainsymbols_x = self.trainset_rx[:-1*self.n_symbol_test]
        trainsymbols_y = trainset_ref[:-1*self.n_symbol_test]
        testsymbols_x = self.trainset_rx[len(trainsymbols_x):]
        testsymbols_y = trainset_ref[len(trainsymbols_x):]
        self.trainloader, self.testset = \
            self.init_dataloader(self.n_tap, trainsymbols_x, trainsymbols_y,
            testsymbols_x, testsymbols_y, self.label_pos, bs=50)
        
    def train_nn(self, trainset_rx):
        """ Train the self.neuralnet using trainset_rx.
        
        Argument:
            trainset_rx - a numpy array containing all data for both training
                and validation. The seperation of training and validation data
                is determined by self.n_symbol_test attribute (the last
                n_symbol_test samples are for validation).
        """
        self.prepare_trainloader(trainset_rx)
        criterion = nn.MSELoss() #criterion = nn.CrossEntropyLoss()
        criterion = criterion.double()
        optimizer = optim.SGD(self.neuralnet.parameters(), lr=0.1, momentum=0.6)
        accuracy_histbest = 0
        accuracy_waiting_cnt = 0
        for epoch in range(self.max_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data_batched in enumerate(self.trainloader):
                # get the inputs
                inputs = data_batched['features']
                labels = data_batched['labels'].unsqueeze(1).double()
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = self.neuralnet(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # print statistics
                if (i % 299 == 0):    # print every 300 mini-batches
                    print('epoch %d-%d, loss: %.3f' %
                          (epoch + 1, i+1, running_loss / 299))
                    running_loss = 0.0
            correct = 0
            total = 0
            output_dfnn_buf = torch.Tensor([[0]])
            testset_outputs_dfnn = []
            with torch.no_grad():
                #for i, data in enumerate(testloader):
                for i in range(self.testset.n_sample()):
                    ####### extract data from dataset #######
                    data = self.testset.getitem(i)
                    inputs = torch.tensor(data['features']).unsqueeze(0)
                    labels = data['labels']
                    #inputs = data['features']
                    inputs[0][-1] = output_dfnn_buf[0][0]
                    outputs = self.neuralnet(inputs)
                    testset_outputs_dfnn.append(outputs.item())
                    predicted = np.round(outputs.item())
                    #output_dfnn_buf = outputs.clone()    # can achieve 0 BER with 0 noise
                    output_dfnn_buf = torch.Tensor([[predicted]])  # can achieve 0 BER with 0 noise
                    total += 1
                    if predicted == labels.item():
                        correct += 1
                    else:
                        print(i,
                              '\n\tinput=',inputs,'\n', 
                              '\tlabel=', labels, '\n',
                              '\toutput=',outputs.item(), predicted)
                        pass
            print('Accuracy on %d test data: %.4f %%' % (total, 100*correct/total))
        #    plt.hist(testset_outputs_dfnn,bins=100)
        #    plt.show()
            if accuracy_waiting_cnt>1:
                if accuracy_histbest <= correct/total:
                    break
            if accuracy_histbest < correct/total:
                accuracy_histbest = correct/total
                accuracy_waiting_cnt = 0
            else:
                accuracy_waiting_cnt+=1

    def run_inference(self, testset_nrz, rx_symbol):
        """ Run inference with the trained neural network.
        
        Arguments:
            testset_nrz - list of ±1 as the test labels
            rx_symbol - the received signal via ADC.
        """
        testsymbols_x = rx_symbol
        testsymbols_y = testset_nrz
        (test_x, test_y) = self.lineup(testsymbols_x, testsymbols_y,
                           n_tap=self.n_tap, label_pos=self.label_pos,
                           for_test=True)
        testset = nn_ndarray_dataset(test_x, test_y)
        correct = 0
        total = 0
        output_dfnn_buf = torch.Tensor([[0]])
        testset_outputs_dfnn = []
        with torch.no_grad():
            for i in range(testset.n_sample()):
                ####### extract data from dataset #######
                data = testset.getitem(i)
                inputs = torch.tensor(data['features']).unsqueeze(0)
                labels = data['labels']
                #inputs = data['features']
                inputs[0][-1] = output_dfnn_buf[0][0]
                outputs = self.neuralnet(inputs)
                testset_outputs_dfnn.append(outputs.item())
                predicted = np.round(outputs.item())
                #output_dfnn_buf = outputs.clone()    # 0 BER with 0 noise
                output_dfnn_buf = torch.Tensor([[predicted]])
                total += 1
                if predicted == labels.item():
                    correct += 1
                else:
                    print(i,
                          '\n\tinput=',inputs,'\n', 
                          '\tlabel=', labels, '\n',
                          '\toutput=',outputs.item(),'-->', predicted)
                    pass
        print('BER on %d test data: %.4f %%' % (total, 100*(1-correct/total)))

    def hd_decision(self, testset_bits, rx_symbol):
        """ recover payload bits using hard decision to comapre BER. """
        rx_hard = OOK_signal(data_ref=rx_symbol) #
        n_error = 0
        for i in range(rx_hard.data_len):
            if (rx_hard.data_bits[i]!=testset_bits[i]):
                n_error+=1
                #print(i,rx_symbol[i],'->',trainset_nrz[i])
        BER_hard = n_error/rx_hard.data_len
        print('BER=%.3f, accurate=%.1f %%' % (BER_hard,100*(1-BER_hard)))

    def save_trained_nn(self, nn_save_path):
        torch.save(self.neuralnet, nn_save_path)
        
    def init_dataloader(self, n_tap, train_x, train_y, test_x, test_y, label_pos, bs):
        (_x, _y) = self.lineup(train_x, train_y, n_tap=n_tap,
                               label_pos=label_pos, framelen=196608)
        (_x_t, _y_t) = self.lineup(test_x, test_y, n_tap=n_tap, 
                                   label_pos=label_pos, for_test=True)
        trainset = nn_ndarray_dataset(_x, _y)
        trainloader = DataLoader(trainset, batch_size=bs,
                                 shuffle=True,drop_last=True) # num_workers=1
        testset = nn_ndarray_dataset(_x_t, _y_t)
        # testloader = DataLoader(testset)
        # return (trainloader, testloader)
        return (trainloader, testset)
        
    def init_nn(self, n_tap):
        self.n_tap = n_tap
        D_in = n_tap+1
        H_1 =  n_tap+1
        H_2 = 2
        D_out = 1
        df_nn = nn.Sequential(
            nn.Linear(D_in, H_1, bias=False),
            nn.Tanh(),
            nn.Linear(H_1, H_2, bias=False),
            nn.Tanh(),
            nn.Linear(H_2, D_out),
        )
        df_nn = df_nn.double()
        return df_nn

    def lineup(self, x, y, n_tap, label_pos, for_test=False, framelen=None):
        """ lineup feature and duo-binary labels for the decision-feedback NN"""
        if framelen==None:
            framelen = len(x)
        if label_pos>n_tap-1:
            raise ValueError('invalid label_pos')
        else:
            features = []
            labels = []
            n_frames = int(len(x)/framelen)
            for frame_i in range(n_frames):
                for i in range(framelen-n_tap+1):
                    temp_x = x[frame_i*framelen+i:frame_i*framelen+i+n_tap]
                    if for_test:
                        features.append(np.append(temp_x, 10)) 
                    else:
                        features.append(np.append(temp_x, (y[frame_i*framelen+i+label_pos-1]+y[frame_i*framelen+i+label_pos-2])/2))
                    labels.append((y[frame_i*framelen+i+label_pos]+y[frame_i*framelen+i+label_pos-1])/2)
            for i in range(n_frames*framelen, len(x)-n_tap):
                temp_x = x[i:i+n_tap]
                if for_test:
                    features.append(np.append(temp_x, 10))
                else:
                    features.append(np.append(temp_x, (y[i+label_pos-1]+y[i+label_pos-2])/2))
                labels.append((y[i+label_pos]+y[i+label_pos-1])/2)
            return(np.array(features),np.array(labels))


class awg(M8195A):
    def __init__(self, addr):
        M8195A.__init__("TCPIP0::{}::inst0::INSTR".format(addr))
        self.set_ext_clk()
        self.set_ref_out()  # set ref_out frequency by configuring two dividers 
                            # default parameters: div1=2, div2=5
        self.set_fs_GHz(nGHz=56)
        self.set_amp(0.6)   # set output amplitude (range between 0 and 1)
        self.set_ofst(0)    # set output offset

class nn_ndarray_dataset(Dataset):
    """the customized data reader for Pytorch's DataLoader utility.
       Inherited from the abstract class 'Dataset' with overided __len__()
       and __getitem__() methods. Takes ndarray as inputs.
    """
    def __init__(self, dataframe_x, dataframe_y, transform=None):
        """
        Args:
            pd_dataframe_x/y, pandas dataframe of feature/label.
        """
        self.dataframe_x = dataframe_x
        self.dataframe_y = dataframe_y
        self.transform = transform
        
    def __len__(self):
        return self.dataframe_x.shape[0]

    def __getitem__(self, idx):
        sample = {'features':self.dataframe_x[idx], 
                  'labels':self.dataframe_y[idx]}
        return sample
    getitem = __getitem__
    n_sample = __len__

def read_sample_bin_file(filepath, dtype = 'b'): # default formating 'b'=int8
    with open(filepath,'rb') as fo:
        filedata = array.array(dtype,fo.read())
    return np.array(filedata.tolist())

def normalize_rxsymbols(rx_raw):
    rx_shift = (np.array(rx_raw)-np.mean(rx_raw))
    return rx_shift/np.max(np.abs(rx_shift))

class Fan(QObject):
    """ Define a class to show a picture of a fan, for animation.
    
    To define a pyqtProperty for animation, the base class should be a QObject
    or any other inherited classes, like QWidget.
    Then, add a QGraphicsPixmapItem to accormadate the picture.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap_item = QGraphicsPixmapItem(QPixmap(cwd+'\\guiunits\\imags\\pon56gdemo\\fan.png'))
        self.pixmap_item.setTransformOriginPoint(self.pixmap_item.boundingRect().center())
        #self.pixmap_item.show()
        
    def _set_rotation_dgr(self, dgr):
        self.pixmap_item.setRotation(dgr)
        
    def fanAnimation(self):
        anim = QPropertyAnimation(self, b'rotation')
        return anim
    
    rotation = pyqtProperty(float, fset=_set_rotation_dgr)

        
class AppWindow(QMainWindow):
    def __init__(self, datadevice):
        super().__init__()
        self.datadevice = datadevice
        self.nokia_blue = QColor(18, 65, 145)
        self.title = "High-speed PON demo"
        self.geo = {
            'top'   : 30,
            'left'  : 0,
            'width' : 1920,
            'height': 1080 }
        self.setStyleSheet("background-color: white;")
        self.initWindow()
        
    def initWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.geo['left'], self.geo['top'], 
                         self.geo['width'], self.geo['height'])

        self.titleBar = self.inittitlebar()
        self.bkgrndGroup = self.initbkgrndgroup()
        self.detailGroup = self.initdetailgroup()
        self.prototypeGroup = self.initprototypeGroup()
        
        self.initGeometries()
        self.initConnections()
        self.show()
        
    def initGeometries(self):
        self.titleBar.setGeometry(147 ,59, 1625, 69)
        self.bkgrndGroup.setGeometry(20 ,178, 570, 826)
        self.detailGroup.setGeometry(610 ,178, 1285, 420)
        self.prototypeGroup.setGeometry(610, 613, 1285, 391)

    def inittitlebar(self):
        wdgt = QWidget(parent=self)
        mainTitle = QLabel(parent=wdgt)
        mainTitle.setText("Ultra-Fast Fiber Access with Intelligent PHY")
        font = QFont("Nokia Pure Text Light", 30, QFont.Bold)
        mainTitle.setFont(font)
        # mainTitle.setFrameStyle(22)  # show border
        mainTitle.setAlignment(Qt.AlignRight | Qt.AlignCenter)
        palette = self.palette()
        palette.setColor(self.foregroundRole(), self.nokia_blue)
        mainTitle.setPalette(palette)
        mainTitle.setGeometry(0,0,880, 69)
        
        subTitle = QLabel(parent=wdgt)
        subTitle.setText("—— Enabling 50Gbps over 10G-class devices")
        font = QFont("Nokia Pure Text Light", 20)
        subTitle.setFont(font)
        # subTitle.setFrameStyle(22)  # show border
        subTitle.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        palette = self.palette()
        palette.setColor(self.foregroundRole(), self.nokia_blue)
        subTitle.setPalette(palette)
        subTitle.setGeometry(900,16,600, 40)
        return wdgt

    def initbkgrndgroup(self):
        wdgt = QWidget(parent=self)
        
        title = QLabel(parent=wdgt)
        title.setText("Growing Demand for Access")
        font = QFont("Nokia Pure Text Light", 25)
        title.setFont(font)
        # title.setFrameStyle(22)  # show border
        title.setAlignment(Qt.AlignHCenter | Qt.AlignCenter)
        palette = self.palette()
        palette.setColor(self.foregroundRole(), self.nokia_blue)
        title.setPalette(palette)
        title.setGeometry(20, 10, 490, 69)
        
        bkgrndYear = QLabel(parent=wdgt)
        bkgrndYear.setPixmap(QPixmap(cwd+'\\guiunits\\imags\\pon56gdemo\\bkgrndyear.png'))
        bkgrndYear.move(25,110)
        
        bkgrndSlider = QLabel(parent=wdgt)
        bkgrndSlider.setPixmap(QPixmap(cwd+'\\guiunits\\imags\\pon56gdemo\\bkgrndslider.png'))
        bkgrndSlider.move(35,490)
        
        bkgrnd2015 = QLabel(parent=wdgt)
        bkgrnd2015.setPixmap(QPixmap(cwd+'\\guiunits\\imags\\pon56gdemo\\bkgrnd2015.png'))
        bkgrnd2015.move(280, 600)
        # anim2015 = QPropertyAnimation(bkgrnd2015, b"windowOpacity")
        
        bkgrnd2020 = QLabel(parent=wdgt)
        bkgrnd2020.setPixmap(QPixmap(cwd+'\\guiunits\\imags\\pon56gdemo\\bkgrnd2020.png'))
        bkgrnd2020.move(270, 340)
        mask2020 = QGraphicsOpacityEffect(parent=bkgrnd2020)
        bkgrnd2020.setGraphicsEffect(mask2020)
        mask2020.setOpacity(0.5)

        print(mask2020.isEnabled() )
        
        bkgrnd2025 = QLabel(parent=wdgt)
        bkgrnd2025.setPixmap(QPixmap(cwd+'\\guiunits\\imags\\pon56gdemo\\bkgrnd2025.png'))
        bkgrnd2025.move(280, 103)
        
        
        wdgt.setStyleSheet("background-color: rgb(242, 242, 242);")
        return wdgt
        
    def initdetailgroup(self):
        view = QGraphicsView(parent=self)
        brush=QBrush(QColor(242, 242, 242))
        view.setBackgroundBrush(brush)
        view.setFrameStyle(16)  # QFrame.Plain
        
        detailFigure = QGraphicsPixmapItem(QPixmap(cwd+'\\guiunits\\imags\\pon56gdemo\\detailfigure.png'))
        title = QGraphicsTextItem("Our Innovation/Contribution")
        font = QFont("Nokia Pure Text Light", 25)
        title.setFont(font)
        title.setDefaultTextColor(self.nokia_blue)
        fan = Fan()
        
        scene = QGraphicsScene()
        scene.setSceneRect(0, 0, 1285, 420)
        scene.addItem(detailFigure)
        scene.addItem(title)
        scene.addItem(fan.pixmap_item)
        
        detailFigure.setPos(QPointF(28, 52))
        title.setPos(QPointF(50,20))
        fan.pixmap_item.setPos(QPointF(457, 139))
        self.fanAnim = fan.fanAnimation()
        
        view.setScene(scene)
        view.setSceneRect(0, 0, 1285, 420)
        view.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setRenderHint(QPainter.Antialiasing)      

        return view
    
    
    def initprototypeGroup(self):
        wdgt = QWidget(parent=self)
        
        title = QLabel(parent=wdgt)
        title.setText("Prototype Real-time Monitor")
        font = QFont("Nokia Pure Text Light", 25)
        title.setFont(font)
        # title.setFrameStyle(22)  # show border
        title.setAlignment(Qt.AlignLeft | Qt.AlignCenter)
        palette = self.palette()
        palette.setColor(self.foregroundRole(), self.nokia_blue)
        title.setPalette(palette)
        title.setGeometry(20, 10, 490, 69)
        
        meter = Speedometer(title='Data Rate', unit = 'Gbps',
                            min_value=0, max_value=60, parent=wdgt)
        meter.setGeometry(490, 60, 320, 280)
        
        berPlot = pon56gDemoPlot(parent=wdgt, width=5, height=4, tight_layout=True,
                                 dpi=100, datadevice=self.datadevice)
        berPlot.setGeometry(40, 70, 440, 280)

        ConsoleGroupBox = QGroupBox("Device Control Panel", parent=wdgt)
        ConsoleGroupBox.setGeometry(820, 18, 420, 340)
        Console = QTextBrowser()
        AddrEdit = QLineEdit()
        ConnectButton = ConnectBtn(AddrEdit)
        TestConnectionButton = QPushButton("Test Connection")
        StartStopButton = QPushButton("Train NN")
        QuitButton = QPushButton("Quit")
        layout = QVBoxLayout()
        sublayout = QGridLayout()
        sublayout_widget = QWidget()
        sublayout.addWidget(AddrEdit, 1, 0, 1, 2)
        sublayout.addWidget(ConnectButton, 1, 2)
        sublayout.addWidget(TestConnectionButton, 2, 0)
        sublayout.addWidget(StartStopButton, 2, 1)
        sublayout.addWidget(QuitButton, 2, 2)
        sublayout_widget.setLayout(sublayout)
        layout.addWidget(Console)
        layout.addWidget(sublayout_widget)
        ConsoleGroupBox.setLayout(layout)
        AddrEdit.setStyleSheet("background-color: rgb(255, 255, 255);")
        Console.setStyleSheet("background-color: rgb(255, 255, 255);")
        
        wdgt.setStyleSheet("background-color: rgb(242, 242, 242);")
        return wdgt
        
    def initConnections(self):
        pass


if __name__ == '__main__':
    csvpath = 'D:\\PythonScripts\\lab604-automation\\vtbackendlib\\0726vt899pon56g\\'
#    ook_preamble = OOK_signal(load_file= csvpath+'Jul 6_1741preamble.csv')
    frame_len = 196608
#    trainset = OOK_signal()
#    trainset.append(OOK_signal(load_file=csvpath+'Jul 9_0841train.csv'))
#    trainset.append(OOK_signal(load_file=csvpath+'Jul 9_0842train.csv'))
#    trainset.append(OOK_signal(load_file=csvpath+'Jul 9_0843train.csv'))
#    trainset.append(OOK_signal(load_file=csvpath+'Jul 9_0845train.csv'))

    if not _SIM:
        # initiate AWG
        m8195a = awg(M8195Addr)
    
    vt899 = vtdev("vt899pondemo", VT899Addr, frame_len, 56)
    
    pon56Gdemo = QApplication(sys.argv)
    window = AppWindow(datadevice=vt899)
    sys.exit(pon56Gdemo.exec())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    vt899.open_device()
#    vt899.print_commandset()
#    vt899.query('hello')
#    
#    if not _SIM:
#        # send a frame containing preamble
#        data_for_prmbl_sync = trainset.take(slice(frame_len))
#        awg.send_binary_port1(250*(data_for_prmbl_sync-0.5))
#        sleep(1)  # make sure the ADC has captured the new waveform
#        
#    ook_prmlb = ook_preamble.nrz(dtype = 'int8')
#    vt899.set_preamble(ook_prmlb)
#    vt899.trainset = trainset
#    vt899.config('prmbl500', ook_prmlb.tobytes())
#    sleep(2)  # the backend need several seconds to do resample & correlation
#    ref_bin = vt899.query_bin('getRef 2000')
#    vt899.preamble_wave = np.array(memoryview(ref_bin).cast('f').tolist())
#   
##    corr_result = np.array(memoryview(vt899.query_bin('getCorr 1570404')).cast('f').tolist())
##    plt.plot(corr_result)
#    
#    trainset_rx = []
#    if not _SIM:
#        for i in range(4):
#            print(slice(i*frame_len, (i+1)*frame_len))
#            data_for_train = trainset.take(slice(i*frame_len, (i+1)*frame_len))
#            awg.send_binary_port1(250*(data_for_train-0.5))
#            sleep(2)  # make sure the ADC has captured the new waveform
#            frame_bin = vt899.query_bin('getFrame 786432')
#            frame = list(memoryview(frame_bin).cast('f').tolist())
#            trainset_rx.extend(frame)
#    else:
#        for i in range(4):
#            path_str = sample_folder+'chan1-{0}.bin'.format(i)
#            samples_all = normalize_rxsymbols(read_sample_bin_file(path_str))
#            (samples_frame, cl) = extract_frame(samples_all, 196608, ook_preamble.nrz())
#            trainset_rx.extend(resample_symbols(samples_frame, vt899.preamble_wave))
#       
#    vt899.train_nn(np.array(trainset_rx))
#    
#    if not _SIM:
#        ookrand = OOK_signal()
#        ookrand.append(ook_preamble)
#        ookrand.append(np.random.randint(2,size=frame_len-2*ook_preamble.data_len))
#        ookrand.append(ook_preamble)
#        awg.send_binary_port1(126*ookrand.nrz(), rescale = False)
#        rx_bin = vt899.query_bin('getFrame 786432')
#        rx_frame = list(memoryview(rx_bin).cast('f').tolist())
#    else:
#        ook_rand = OOK_signal(data_ref=trainset.take(slice(0*frame_len, 1*frame_len)))
#        rx_all = normalize_rxsymbols(read_sample_bin_file(csvpath+'chan1-0.bin'))
#        (rx_orign, cl) = extract_frame(rx_all, 196608, ook_preamble.nrz())
#        rx_frame = resample_symbols(rx_orign, vt899.preamble_wave)
#        
#    vt899.hd_decision(ook_rand.nrz(), rx_frame)
#    vt899.run_inference(ook_rand.nrz(), rx_frame)
#    
#    vt899.close_device()
#    
#    





