# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:16:16 2018

@author: dongxucz
"""

import numpy as np
import csv as csvlib
from locale import atoi
from os.path import exists
from torch.utils.data import Dataset

class OOK_signal:
    """Create (randomly generate|load from file) or save an OOK signal.

       NOTE: Default OOK formating is RZ code, therefore, when external ref
       is given (data_ref or load_file), any sample beyond {0, 1} will be 
       converted to 0 or 1 following the principle:
            x = 0 if sample_value <= 0
                1 if sample_value > 0
       In other words, given a ref signal of NRZ code, this automatically
       converts to default RZ code, stored in OOK_signal.data_bits
       Use OOK_signal.nrz() to get an NRZ code of data_bits.
    """
    def __init__(self, data_len = 0, data_ref=[], load_file = ''):
        '''
        Arguments:
            data_len - number of symbols to generate/load
            data_ref - a list containing reference symbols to copy from
            load_file - data file's name to copy from (should be xx.csv)
        '''
        if (load_file!=''): # load data from disk
            if exists(load_file):
                if (load_file[-3:]!='csv'):
                    raise ValueError('OOK_signal: only .csv is supported')
                else:
                    f = open(load_file, 'r')
                    #self.data_bits=np.array([atoi(item[0]) for item in csvlib.reader(f)])
                    self.data_bits=np.sign([np.max((0,atoi(item[0]))) for item in csvlib.reader(f)])
                    f.close()
                    if (data_len==0)|((data_len!=0)&(data_len>len(self.data_bits))):
                        self.data_len = len(self.data_bits)
                        if data_len!=0:
                            print('WARNING: load_file has less samples ({0}) than required ({1})'.format(self.data_len, data_len))
                    else:
                        self.data_bits=self.data_bits[0:data_len]
                        self.data_len = data_len
            else:
                raise ValueError('Class OOK_signal: {0} does not exist'.format(load_file))
        else: # copy from reference or generate randomly
            if (len(data_ref) == 0):
                self.data_len = data_len
                self.data_bits = np.random.randint(2,size=data_len)
            else:
                if (data_len==0)|((data_len!=0)&(data_len>len(data_ref))):
                    self.data_bits = np.sign([np.max((0,item)) for item in data_ref])
                    self.data_len = len(data_ref)
                    if (data_len != 0):
                        print('WARNING: data_ref has less samples ({0}) than required ({1})'.format(self.data_len, data_len))
                else:
                    self.data_bits = np.sign([np.max((0,item)) for item in data_ref[0:data_len]])
                    self.data_len = data_len
    def append(self, a):
        #print(a.__class__, self.__class__)
        if (a.__class__ == np.ndarray or a.__class__ == list):
            a_ook = np.sign([np.max((0,item)) for item in a])
        #elif (a.__class__== self.__class__):
        else:
            a_ook = a.data_bits
        self.data_bits = np.concatenate((self.data_bits, a_ook),axis=0)
        self.data_len = len(self.data_bits)
        
    def nrz(self, dtype = int):
        temp_array = np.sign( self.data_bits - 0.5 )
        return temp_array.astype(dtype)
    
    def data_bits_astype(self, dtype):
        return self.data_bits.astype(dtype)
    
    
    def save_to_csv(self, csv_name=None):
        if csv_name==None:
            raise ValueError('OOK_signal::save_to_csv() - please provide a file name')
        else:
            with open(csv_name,'w', newline='') as f:
                writer = csvlib.writer(f)
                for item in self.data_bits:
                    writer.writerow([item])
            f.close()
            
class nn_pd_dataset(Dataset):
    """the customized data reader for Pytorch's DataLoader utility.
       Inherited from the abstract class 'Dataset' with overided __len__()
       and __getitem__() methods. Takes padas dataset as inputs.
    """
    def __init__(self, pd_dataframe_x, pd_dataframe_y, transform=None):
        """
        Args:
            pd_dataframe_x/y, pandas dataframe of feature/label.
        """
        self.pd_dataframe_x = pd_dataframe_x
        self.pd_dataframe_y = pd_dataframe_y
        self.transform = transform
        
    def __len__(self):
        return self.pd_dataframe_x.shape[0]

    def __getitem__(self, idx):
        sample = {'features':self.pd_dataframe_x.iloc[idx].get_values(), 
                  'labels':self.pd_dataframe_y.iloc[idx]}
        return sample
    getitem = __getitem__
    n_sample = __len__
    
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