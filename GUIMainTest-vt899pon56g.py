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
                QVBoxLayout, QGridLayout, QSlider
from PyQt5.QtGui import ( QBrush, QPen, QPainter, QPixmap, QFont, QColor,
                         QIcon, QTextDocument)
from PyQt5.QtCore import (Qt, QObject, QPointF, QSize, QRect, QEasingCurve,
        QPropertyAnimation, pyqtProperty, pyqtSignal, QEvent, QStateMachine, 
        QSignalTransition, QState, QTimer)
from vtbackendlib.vt899 import extract_frame, resample_symbols
import numpy as np
import core.vt_device as vt_device
from core.ook_lib import OOK_signal
from core.ks_device import M8195A
from guiunits.connectbutton import ConnectBtn
from guiunits.speedometer import Speedometer
from guiunits.mlpplotwidget import pon56gDemoBerPlot, pon56gDemoMsePlot
from guiunits.pon56gDemoNNTrainingOutput_s import training_console_output
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

############## for Debugging ################
_SIM = True


############ global variables ###############
VT899Addr = "10.242.13.34", 9998
M8195Addr = "10.242.13.77"

cwd = getcwd()
sample_folder = cwd+'\\vtbackendlib\\0726vt899pon56g\\'

csvpath = 'D:\\PythonScripts\\lab604-automation\\vtbackendlib\\0726vt899pon56g\\'
frame_len = 196608
ook_preamble = OOK_signal(load_file= csvpath+'Jul 6_1741preamble.csv')
ook_prmlb = ook_preamble.nrz(dtype = 'int8')
globalTrainset = OOK_signal()
if not _SIM:  #
    print('Loading data ..........')
    globalTrainset.append(OOK_signal(load_file=csvpath+'Jul 9_0841train.csv'))
    print('25% done ...')
    globalTrainset.append(OOK_signal(load_file=csvpath+'Jul 9_0842train.csv'))
    print('50% done ...')
    globalTrainset.append(OOK_signal(load_file=csvpath+'Jul 9_0843train.csv'))
    print('75% done ...')
    globalTrainset.append(OOK_signal(load_file=csvpath+'Jul 9_0845train.csv'))
    print('OK!\n')


#    vt899.trainset = globalTrainset
#    vt899.config('prmbl500', ook_prmlb.tobytes())

class vtdev(vt_device.VT_Device):
    
    # algorithm state coding for self.algo_state:
    Init = 0     # before setting preamble (cannot extract frame)
    NoNN = 1     # can extract frame, but NN not trained
    YesNN = 2    # NN trained
    TranSit = 3  # A intermediate state: just after NN trained, but speedometer animation not done
    
    def __init__(self, devname, frame_len=0, symbol_rate=0, addr=None, gui=True):
        vt_device.VT_Device.__init__(self, devname, gui)
        self.set_net_addr(addr)
        self.frame_len = frame_len
        self.n_prmbl = 500  # n_payload=195608. see work notebook-2 18-07-05
        self.n_symbol_test = 10000
        self.symbol_rate = symbol_rate
        self.algo_state = vtdev.Init
        self.set_gui_verbos(1,1,1)
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
                    self.msgbf.info('epoch %d-%d, loss: %.3f' %
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
                        self.msgbf.info('{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}'.format(i,
                              '\n\tinput=',inputs,'\n', 
                              '\tlabel=', labels, '\n',
                              '\toutput=',outputs.item(), predicted))
                        pass
            self.msgbf.info('Accuracy on %d test data: %.4f %%' % (total, 100*correct/total))
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
        
        self.algo_state = vtdev.YesNN
        
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
                    self.msgbf.info(i,
                          '\n\tinput=',inputs,'\n', 
                          '\tlabel=', labels, '\n',
                          '\toutput=',outputs.item(),'-->', predicted)
                    pass
        self.msgbf.info('BER on %d test data: %.4f %%' % (total, 100*(1-correct/total)))

    def hd_decision(self, testset_bits, rx_symbol):
        """ recover payload bits using hard decision to comapre BER. """
        rx_hard = OOK_signal(data_ref=rx_symbol) #
        n_error = 0
        for i in range(rx_hard.data_len):
            if (rx_hard.data_bits[i]!=testset_bits[i]):
                n_error+=1
                #print(i,rx_symbol[i],'->',trainset_nrz[i])
        BER_hard = n_error/rx_hard.data_len
        self.msgbf.info('BER=%.3f, accurate=%.1f %%' % (BER_hard,100*(1-BER_hard)))
    
    def calcEexpectedGbps(self, ber, maxGbps=60):
        """ calculate an 'achievable bit rate' based on a BER value"""
        if ber>0.8:  # abnormal
            expectedGbps = 0
        elif ber>0.009:  # NoNN
            expectedGbps = 12.5  # 14-ber*10
        else:  # YesNN
            expectedGbps = -152.073*ber + 51.018
        return expectedGbps
    
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

class itemClickedSgnlWrapper(QObject):
    sgnl = pyqtSignal()

class Fan(QObject):
    """ Define a class to show a picture of a fan, for animation.
    
    To define a pyqtProperty for animation, the base class should be a QObject
    or any other inherited classes, like QWidget.
    Then, add a QGraphicsPixmapItem to host the picture.
    """
       
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap_item = QGraphicsPixmapItem(QPixmap(cwd+'\\guiunits\\imags\\pon56gdemo\\fan.png'))
        self.pixmap_item.setTransformOriginPoint(self.pixmap_item.boundingRect().center())
        #self.clickedSgnlWrapper = itemClickedSgnlWrapper()
        #self.clicked = self.clickedSgnlWrapper.sgnl
        #self.pixmap_item.mousePressEvent = self.clickEventHandler
        
    def clickEventHandler(self, event):
        print('emitting signal')
        self.clicked.emit()
        
    def _set_rotation_dgr(self, dgr):
        self.pixmap_item.setRotation(dgr)
        
    def fanAnimation(self):
        anim = QPropertyAnimation(self, b'rotation')
        anim.setDuration(1000)
        anim.setStartValue(0)
        anim.setEndValue(360)
        anim.setLoopCount(-1)
        return anim
    
    # define a property named as 'rotation', and designate a setter function.
    rotation = pyqtProperty(float, fset=_set_rotation_dgr)


class fadingPic(QObject):
    """ Wrap a QGraphicsPixmapItem and impliment the fade in/out animation"""
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
    
    def _set_opacity(self, opc):
        self.pixmap_item.setOpacity(opc)
    
    def fadeIn(self):
        anim = QPropertyAnimation(self, b'opacity')
        anim.setDuration(800)
        anim.setStartValue(0)
        anim.setEndValue(1)
        #anim.setLoopCount(1)
        return anim
    
    def fadeOut(self):
        anim = QPropertyAnimation(self, b'opacity')
        anim.setDuration(800)
        anim.setStartValue(1)
        anim.setEndValue(0)
        #anim.setLoopCount(1)
        return anim    
    
    opacity = pyqtProperty(float, fset=_set_opacity)
    
class AppWindow(QMainWindow):
    def __init__(self, datadevice, awg):
        super().__init__()
        self.datadevice = datadevice
        self.awg = awg
        self.nokia_blue = QColor(18, 65, 145)
        self.title = "High-speed PON demo"  # "超高速光接入网" 
        self.geo = {
            'top'   : 30,
            'left'  : 0,
            'width' : 1920,
            'height': 1080 }
        self.setStyleSheet("background-color: white;")
        self._detailFigure_2ClickedSigWrapper = itemClickedSgnlWrapper()
        self.detailFigure_2Clicked = self._detailFigure_2ClickedSigWrapper.sgnl
        self.setFocusPolicy(Qt.StrongFocus)
        self.initWindow()
        self._lang = 'en'  # language. Changed on pressing "L" key
        
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
        # self.fanAnim.start()
        self.show()
        
    def initGeometries(self):
        self.titleBar.setGeometry(147 ,59, 1625, 69)
        self.bkgrndGroup.setGeometry(20 ,178, 570, 826)
        self.detailGroup.setGeometry(610 ,178, 1285, 420)
        self.prototypeGroup.setGeometry(610, 613, 1285, 391)

    def inittitlebar(self):
        wdgt = QWidget(parent=self)
        mainTitle = QLabel(parent=wdgt)
        mainTitle.setText("Ultra-Fast Fiber Access with Intelligent PHY") # 
        font = QFont("Nokia Pure Text Light", 35, QFont.Bold)
        mainTitle.setFont(font)
        # mainTitle.setFrameStyle(22)  # show border
        mainTitle.setAlignment(Qt.AlignHCenter | Qt.AlignCenter) # Qt.AlignRight
        palette = self.palette()
        palette.setColor(self.foregroundRole(), self.nokia_blue)
        mainTitle.setPalette(palette)
        mainTitle.setGeometry(0,0,950, 69)
        
        subTitle = QLabel(parent=wdgt)
        subTitle.setText("—— Enabling 50Gbps over 10G-class devices")
        font = QFont("Nokia Pure Text Light", 20)
        subTitle.setFont(font)
        # subTitle.setFrameStyle(22)  # show border
        subTitle.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        palette = self.palette()
        palette.setColor(self.foregroundRole(), self.nokia_blue)
        subTitle.setPalette(palette)
        subTitle.setGeometry(960,16,600, 40)
        
        self.mainTitle = mainTitle
        self.subTitle = subTitle
        return wdgt

    def initbkgrndgroup(self):
        wdgt = QWidget(parent=self)
        
        title = QLabel(parent=wdgt)
        title.setText("Growing Demand for Access")
        font = QFont("Nokia Pure Text Light", 25, QFont.Bold)
        title.setFont(font)
        # title.setFrameStyle(22)  # show border
        title.setAlignment(Qt.AlignLeft | Qt.AlignCenter)  # Qt.AlignHCenter
        palette = self.palette()
        palette.setColor(self.foregroundRole(), self.nokia_blue)
        title.setPalette(palette)
        title.setGeometry(20, 10, 490, 69)
        
        bkgrndYear = QLabel(parent=wdgt)
        bkgrndYear.setPixmap(QPixmap(cwd+'\\guiunits\\imags\\pon56gdemo\\bkgrndyear.png'))
        bkgrndYear.move(25,110)
        
        bkgrndSlider = QPushButton(parent=wdgt)
        bkgrndSlider.setFixedSize(40,60)
        bkgrndSlider.setStyleSheet("QPushButton { background : transparent }")
        bkgrndSlider.setIcon(QIcon(cwd+'\\guiunits\\imags\\pon56gdemo\\bkgrndslider.png'))
        bkgrndSlider.setIconSize(QSize(50,63))
        bkgrndSlider.setFlat(True)
        bkgrndSlider.move(38,640)
        
        sliderAnim_1 = QPropertyAnimation(bkgrndSlider, b"geometry")
        sliderAnim_1.setStartValue(QRect(38, 640, 40, 60))
        sliderAnim_1.setEndValue(QRect(38, 400, 40, 60))
        sliderAnim_1.setDuration(1000)
        sliderAnim_1.setEasingCurve(QEasingCurve.OutQuad)
        sliderAnim_2 = QPropertyAnimation(bkgrndSlider, b"geometry")
        sliderAnim_2.setStartValue(QRect(38, 400, 40, 60))
        sliderAnim_2.setEndValue(QRect(38, 160, 40, 60))
        sliderAnim_2.setDuration(1000)
        sliderAnim_2.setEasingCurve(QEasingCurve.OutQuad)
        
        bkgrnd2015 = QLabel(parent=wdgt)
        bkgrnd2015.setPixmap(QPixmap(cwd+'\\guiunits\\imags\\pon56gdemo\\bkgrnd2015.png'))
        bkgrnd2015.move(280, 600)
        # anim2015 = QPropertyAnimation(bkgrnd2015, b"windowOpacity")
        
        bkgrnd2020 = QLabel(parent=wdgt)
        bkgrnd2020.setPixmap(QPixmap(cwd+'\\guiunits\\imags\\pon56gdemo\\bkgrnd2020.png'))
        bkgrnd2020.move(270, 340)
        mask2020 = QGraphicsOpacityEffect(parent=bkgrnd2020)
        bkgrnd2020.setGraphicsEffect(mask2020)
        mask2020.setOpacity(0)
        bkgrnd2020FadeIn = QPropertyAnimation(mask2020, b"opacity")
        bkgrnd2020FadeIn.setDuration(1000)
        bkgrnd2020FadeIn.setStartValue(0)
        bkgrnd2020FadeIn.setEndValue(1)
        bkgrnd2020FadeIn.setEasingCurve(QEasingCurve.InQuad)
        
        bkgrnd2025 = QLabel(parent=wdgt)
        bkgrnd2025.setPixmap(QPixmap(cwd+'\\guiunits\\imags\\pon56gdemo\\bkgrnd2025.png'))
        bkgrnd2025.move(275, 110)
        mask2025 = QGraphicsOpacityEffect(parent=bkgrnd2025)
        bkgrnd2025.setGraphicsEffect(mask2025)
        mask2025.setOpacity(0)
        bkgrnd2025FadeIn = QPropertyAnimation(mask2025, b"opacity")
        bkgrnd2025FadeIn.setDuration(1000)
        bkgrnd2025FadeIn.setStartValue(0)
        bkgrnd2025FadeIn.setEndValue(1)
        bkgrnd2025FadeIn.setEasingCurve(QEasingCurve.InQuad)
        
        wdgt.setStyleSheet("background-color: rgb(242, 242, 242);")
        
        self.bkgrndTitle = title
        self.bkgrndSlider = bkgrndSlider
        self.sliderAnim_1 = sliderAnim_1
        self.sliderAnim_2 = sliderAnim_2
        self.mask2020 = mask2020
        self.mask2025 = mask2025
        self.bkgrnd2020FadeIn = bkgrnd2020FadeIn
        self.bkgrnd2025FadeIn = bkgrnd2025FadeIn
        self._bkgrndSliderStatus = 0  # 0 - @2015; 1 - @2020; 2 - @2025
        return wdgt
        
    def initdetailgroup(self):
        view = QGraphicsView(parent=self)
        brush=QBrush(QColor(242, 242, 242))
        view.setBackgroundBrush(brush)
        view.setFrameStyle(16)  # QFrame.Plain
        
        def clickEventHandler(event):
            self.detailFigure_2Clicked.emit()
        
        detailFigure_1 = QGraphicsPixmapItem(QPixmap(cwd+'\\guiunits\\imags\\pon56gdemo\\detailfigure_1.png'))
        detailFigure_2_Qobj = fadingPic(QPixmap(cwd+'\\guiunits\\imags\\pon56gdemo\\detailfigure_2_en.png'))
        detailFigure_2 = detailFigure_2_Qobj.pixmap_item
        detailFigure_1.mousePressEvent = clickEventHandler
        title = QGraphicsTextItem("Our Innovation/Contribution")
        font = QFont("Nokia Pure Text Light", 25, QFont.Bold)
        title.setFont(font)
        title.setDefaultTextColor(self.nokia_blue)

        textItem1 = QGraphicsTextItem()
        textItem1.setHtml('''<body style="font-family:Nokia Pure Text Light;color:#124191;font-size:23px;">
                          <div >10GHz</div>
                          <div > Optics </div>
                          </body>''')
        textItem1.setTextWidth(80)
        textItem2 = QGraphicsTextItem()
        textItem2.setHtml('''<body style="font-family:Nokia Pure Text Light;color:#124191;font-size:23px;">
                          <div > 10GHz</div>
                          <div > Optics </div>
                          </body>''')
        textItem2.setTextWidth(100)
        
        fan = Fan()  # a QObject which wraps a QGraphicsItem inside
        
        scene = QGraphicsScene()
        scene.setSceneRect(0, 0, 1285, 420)
        scene.addItem(detailFigure_2)
        scene.addItem(detailFigure_1)
        scene.addItem(textItem1)
        scene.addItem(textItem2)
        scene.addItem(title)
        scene.addItem(fan.pixmap_item)
        
        detailFigure_1.setPos(QPointF(35, 88))
        detailFigure_2.setPos(QPointF(570, 40))
        detailFigure_2.setOpacity(0)  # hided at first
        title.setPos(QPointF(50,20))
        textItem1.setPos(QPointF(40, 168))
        textItem2.setPos(QPointF(361, 168))
        fan.pixmap_item.setPos(QPointF(456.5, 138))
        self.fanAnim = fan.fanAnimation()
        
        view.setScene(scene)
        view.setSceneRect(0, 0, 1285, 420)
        view.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setRenderHint(QPainter.Antialiasing)      
        
        self.detailGrpTextItem1 = textItem1
        self.detailGrpTextItem2 = textItem2
        self.detailFigTitle = title
        self.turbofan = fan
        self.NNfigure_fadeIn = detailFigure_2_Qobj.fadeIn()
        self.NNfigure_fadeOut = detailFigure_2_Qobj.fadeOut()
        self._detailFigure_2_state = 0  # 0-hided, 1-showed
        return view
    
    
    def initprototypeGroup(self):
        wdgt = QWidget(parent=self)
        
        title = QLabel(parent=wdgt)
        title.setText("Prototype Monitor")
        font = QFont("Nokia Pure Text Light", 25, QFont.Bold)
        title.setFont(font)
        # title.setFrameStyle(22)  # show border
        title.setAlignment(Qt.AlignLeft | Qt.AlignCenter)
        palette = self.palette()
        palette.setColor(self.foregroundRole(), self.nokia_blue)
        title.setPalette(palette)
        title.setGeometry(50, 10, 300, 69)
        
        meter = Speedometer(title='Data Rate', unit = 'Gbps',
                            min_value=0, max_value=55, parent=wdgt)
        meter.setGeometry(40, 80, 320, 270)
        
        initMeterAnim = QPropertyAnimation(meter, b"value")
        initMeterAnim.setStartValue(0)
        initMeterAnim.setEndValue(12)
        initMeterAnim.setDuration(500)

        boostMeterAnim = QPropertyAnimation(meter, b"value")
        boostMeterAnim.setStartValue(12)
        boostMeterAnim.setEndValue(50)
        boostMeterAnim.setDuration(3000)
        boostMeterAnim.setEasingCurve(QEasingCurve.InQuint)

        berPlot = pon56gDemoBerPlot(parent=wdgt, width=3.5, height=2, tight_layout=True,
                                 dpi=100, datadevice=self.datadevice)
        berPlot.setGeometry(405, 25, 420, 170)
        
        msePlot = pon56gDemoMsePlot(parent=wdgt, width=3.5, height=2, tight_layout=True,
                                 dpi=100)
        msePlot.setGeometry(405, 195, 420, 170)
        
        self.updateTimer = QTimer()
        self.updateTimer.setInterval(1100)

        ConsoleGroupBox = QGroupBox("Device Control Panel", parent=wdgt)
        ConsoleGroupBox.setGeometry(870, 22, 370, 340)
        Console = QTextBrowser()
        AddrEdit = QLineEdit()
        # AddrEdit.setText('10.242.13.34')
        AddrEdit.setText('192.168.1.199')
        ConnectButton = ConnectBtn(AddrEdit)
        ResetNNButton = QPushButton("Reset")
        TrainButton = QPushButton("Train NN")
        QuitButton = QPushButton("Quit")
        layout = QVBoxLayout()
        sublayout = QGridLayout()
        sublayout_widget = QWidget()
        sublayout.addWidget(AddrEdit, 1, 0, 1, 2)
        sublayout.addWidget(ConnectButton, 1, 2)
        sublayout.addWidget(ResetNNButton, 2, 0)
        sublayout.addWidget(TrainButton, 2, 1)
        sublayout.addWidget(QuitButton, 2, 2)
        sublayout_widget.setLayout(sublayout)
        layout.addWidget(Console)
        layout.addWidget(sublayout_widget)
        ConsoleGroupBox.setLayout(layout)
        AddrEdit.setStyleSheet("background-color: rgb(255, 255, 255);")
        Console.setStyleSheet("background-color: rgb(255, 255, 255);")
        
        wdgt.setStyleSheet("background-color: rgb(242, 242, 242);")
        
        self.prototypeTitle = title
        self.AddrEdit = AddrEdit
        self.Console = Console
        self.meter = meter
        self.initMeterAnim = initMeterAnim
        self.boostMeterAnim = boostMeterAnim
        self.ConnectButton = ConnectButton
        self.TrainButton = TrainButton
        self.QuitButton = QuitButton
        self.ResetNNButton = ResetNNButton
        self.berPlot = berPlot
        self.msePlot = msePlot
        return wdgt
        
    def bkgrndGroupSM(self):
        """ State machine of animations """
        if self._bkgrndSliderStatus==0:  # move from 2015 to 2020
            self.sliderAnim_1.start()
            self.bkgrnd2020FadeIn.start()
            self._bkgrndSliderStatus = 1
        elif self._bkgrndSliderStatus==1:  # move from 2020 to 2025
            self.sliderAnim_2.start()
            self.bkgrnd2025FadeIn.start()
            self._bkgrndSliderStatus = 2
        elif self._bkgrndSliderStatus==2:  # move back to 2015
            self.bkgrndSlider.move(38,640)
            self.mask2020.setOpacity(0)
            self.mask2025.setOpacity(0)
            self._bkgrndSliderStatus = 0

    def detailFigSM(self):
        """ State machine of animations """
        if self._detailFigure_2_state == 0:
            self.NNfigure_fadeIn.start()
            self._detailFigure_2_state = 1
            self.fanAnim.start()
        else:
            self.NNfigure_fadeOut.start()
            self._detailFigure_2_state = 0
            self.fanAnim.stop()
    
    def initConnections(self):
        self.bkgrndSlider.clicked.connect(self.bkgrndGroupSM)
        self.detailFigure_2Clicked.connect(self.detailFigSM)
        self.ConnectButton.clicked.connect(self.openVTdevice)
        self.AddrEdit.returnPressed.connect(self.openVTdevice)
        self.QuitButton.clicked.connect(self.closeEvent)
        self.datadevice.guisgnl.connect(self.Console.append)
        self.updateTimer.timeout.connect(self.berPlot.update_figure)
        self.berPlot.plot2Console.connect(self.Console.append)
        self.berPlot.plot2Meter.connect(self.meter.setSpeed)
        # self.TrainButton.clicked.connect(self.trainNN)  # train NN
        self.TrainButton.clicked.connect(self.trainNN_temp)  # train NN, just GUI effect
        self.ResetNNButton.clicked.connect(self.resetPlot)
        
    def openVTdevice(self):
        ipaddr = self.AddrEdit.text()
        print((ipaddr, 9998))
        self.datadevice.set_net_addr((ipaddr,9998))
        self.Console.append('connecting to'+ ipaddr)
        if 'Connected' in self.datadevice.open_device().split():
            self.datadevice.query('hello')
            self.Console.append('Set preamble ...')
            self.setPreamble()
    
    def setPreamble(self):
        self.datadevice.set_preamble(ook_prmlb)  # local int preamble record
        #self.datadevice.config('prmbl500', ook_prmlb.tobytes())  # write the same to remote backend
        self.datadevice.trainset = globalTrainset
        sleep(2)  # the backend need several seconds to do resample & correlation
        #ref_bin = self.datadevice.query_bin('getRef 2000')
        #vt899.preamble_wave = np.array(memoryview(ref_bin).cast('f').tolist())
        self.Console.append('Preamble synchronization Done! ')
        self.datadevice.algo_state = self.datadevice.NoNN
        self.initMeterAnim.finished.connect(self.updateTimer.start)
        self.initMeterAnim.start()

        
    def trainNN(self):
        # self.fanAnim.start()  # start fan animation to indicate neural network
        if ((self.datadevice.open_state == 0) or (self.datadevice.algo_state==self.datadevice.Init)):
            self.Console.append('Device not opend, or preamble not set. Cannot procced.')
        else:
            frame_len = self.datadevice.frame_len
            trainset_rx = []
            if not _SIM:
                for i in range(4):
                    print(slice(i*frame_len, (i+1)*frame_len))
                    data_for_train = globalTrainset.take(slice(i*frame_len, (i+1)*frame_len))
                    self.awg.send_binary_port1(250*(data_for_train-0.5))
                    sleep(2)  # make sure the ADC has captured the new waveform
                    frame_bin = self.datadevice.query_bin('getFrame 786432')
                    frame = list(memoryview(frame_bin).cast('f').tolist())
                    trainset_rx.extend(frame)
            else:
                for i in range(4):
                    path_str = sample_folder+'chan1-{0}.bin'.format(i)
                    samples_all = normalize_rxsymbols(read_sample_bin_file(path_str))
                    (samples_frame, cl) = extract_frame(samples_all, 196608, ook_preamble.nrz())
                    trainset_rx.extend(resample_symbols(samples_frame, self.datadevice.preamble_wave))
            self.datadevice.train_nn(np.array(trainset_rx))

    def trainNN_temp(self):
        # not really running NN, in order to test pure GUI functions
        self.updateTimer.stop()
        self.msePlot.reset()
        if ((self.datadevice.open_state == 0) or (self.datadevice.algo_state==self.datadevice.Init)):
            self.Console.append('Device not opend, or preamble not set. Cannot procced.')
        else:
            texts = training_console_output[:]
            tempTimer = QTimer()
            tempTimer.setInterval(30)
            def printTrainingOutput():
                if texts:
                    text_item = texts.pop(0)
                    self.Console.append(text_item)
                    if text_item[-5:]=='demo)':
                        mse = float(text_item[-20:-15])
                        self.msePlot.update_figure(mse)
                else:
                    tempTimer.stop()
                    self.datadevice.algo_state = self.datadevice.TranSit
                    self.updateTimer.start()
                    self.boostMeterAnim.finished.connect(self.changeAlgoState)
                    self.boostMeterAnim.start()
                    #self.updateTimer.start()
            tempTimer.timeout.connect(printTrainingOutput)
            tempTimer.start()

    def resetPlot(self):
        # clear the MSE plot, and turn the BER & speedometer state back to 12.5G
        self.updateTimer.stop()
        self.datadevice.algo_state = self.datadevice.NoNN
        self.msePlot.reset()
        self.updateTimer.start()
        
    def changeAlgoState(self):
        self.datadevice.algo_state = self.datadevice.YesNN
        
    def cleanUpAndQuit(self):
        # close the VT_Device to inform the backend ending the TCP session.
        self.datadevice.close_device()
        self.close()
    
    def keyPressEvent(self, KEvent):
        k = KEvent.key()
        print(k,'  pressed')
        if k==Qt.Key_Q:
            self.bkgrndSlider.click()
        elif k==Qt.Key_W:
            self.detailFigure_2Clicked.emit()
        elif k==Qt.Key_T:
            self.TrainButton.click()
        elif k==Qt.Key_R:
            self.ResetNNButton.click()
        elif k==Qt.Key_L:
            self.switchLang()
        else:
            pass
    
    def switchLang(self):
        if (self._lang == 'en'):
            print("Switching language form EN to CN.")
            self._lang = 'cn'
            self.mainTitle.setText('''<div style="font-family:微软雅黑;margin-left:250px;">基于  <i>人工智能</i>  的‘超高速光接入’ </div>''')
            self.subTitle.setText('''<div style="font-family:微软雅黑;"> ——50G光接入 </div> ''')
            self.bkgrndTitle.setText("光接入的 ‘方案 vs 需求’")
            self.detailFigTitle.setPlainText("颠覆式创新")
            self.prototypeTitle.setText("硬件平台实时监控")
            self.detailGrpTextItem1.setHtml('''
                 <body style="font-family:Nokia Pure Text Light;color:#124191;font-size:23px;">
                 <div >10GHz</div>
                 <div > <b>光器件</b> </div>
                 </body>''')
            self.detailGrpTextItem2.setHtml('''
                 <body style="font-family:Nokia Pure Text Light;color:#124191;font-size:23px;">
                 <div > 10GHz</div>
                 <div > <b>光器件</b> </div>
                 </body>''')
        else:
            print("Switching language form CN to EN.")
            self._lang = 'en'
            self.mainTitle.setText("Ultra-Fast Fiber Access with Intelligent PHY")
            self.subTitle.setText('''<div style="font-family:微软雅黑;"> ——50G光接入 </div> ''')
            self.subTitle.setText("—— Enabling 50Gbps over 10G-class devices")
            self.bkgrndTitle.setText("Growing Demand for Access")
            self.prototypeTitle.setText("Prototype Monitor")
            self.detailFigTitle.setPlainText("Our Innovation/Contribution")
            self.detailGrpTextItem1.setHtml('''
                 <body style="font-family:Nokia Pure Text Light;color:#124191;font-size:23px;">
                 <div >10GHz</div>
                 <div > Optics </div>
                 </body>''')
            self.detailGrpTextItem2.setHtml('''
                 <body style="font-family:Nokia Pure Text Light;color:#124191;font-size:23px;">
                 <div > 10GHz</div>
                 <div > Optics </div>
                 </body>''')
                
    def closeEvent(self, ce):
        self.cleanUpAndQuit()


if __name__ == '__main__':

    if not _SIM:
        print('Initializing AWG ...')
        m8195a = awg(M8195Addr)
        # send a frame containing preamble
        data_for_prmbl_sync = globalTrainset.take(slice(frame_len))
        awg.send_binary_port1(250*(data_for_prmbl_sync-0.5))
        sleep(1)  # make sure the ADC has captured the new waveform
    else:
        m8195a = None
    
    vt899 = vtdev("vt899pondemo", frame_len=frame_len, symbol_rate=56, gui=True)
    
    pon56Gdemo = QApplication(sys.argv)
    window = AppWindow(datadevice=vt899, awg=m8195a)
    sys.exit(pon56Gdemo.exec())
    print("close device")
    vt899.close_device()
    

    ook_preamble = OOK_signal(load_file= csvpath+'Jul 6_1741preamble.csv')
    ook_prmlb = ook_preamble.nrz(dtype = 'int8')


#   
##    corr_result = np.array(memoryview(vt899.query_bin('getCorr 1570404')).cast('f').tolist())
##    plt.plot(corr_result)


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
#        ook_rand = OOK_signal(data_ref=globalTrainset.take(slice(0*frame_len, 1*frame_len)))
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




