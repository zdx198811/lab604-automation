# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 14:29:44 2019

@author: dongxucz
"""

from random import randint
import numpy as np
import csv as csvlib
from locale import atoi, atof
from core.dmt_lib import DmtMod, DmtDeMod
from bitstring import BitArray
import matplotlib.pyplot as plt

def extract_samples(bytes_received):
    alldata = bytes_received.hex()
    samples_int = []
    nsample = int(len(bytes_received)/1.5)
    for i in range(nsample):
        sample_bitarray = BitArray('0x'+ alldata[i*3:(i+1)*3])
        samples_int.append(sample_bitarray.int)
    return samples_int


def load_preamble_data():
    preamble_file_dir = 'D:/PythonScripts/vadatech/vt898/qam16_dmt_Apr26.csv'
    f_pre = open(preamble_file_dir, 'r')
    preamble384 = [atof(item[0]) for item in csvlib.reader(f_pre)]
    f_pre.close()
    preamble_file_dir = 'D:/PythonScripts/vadatech/vt898/qam16_Apr26.csv'
    f_pre_int = open(preamble_file_dir, 'r')
    preamble_int192 = [atoi(item[0]) for item in csvlib.reader(f_pre_int)]
    f_pre_int.close()
    qam_ref = [qam16_to_complex(i) for i in preamble_int192]
    qam_ref_shaped = np.reshape(qam_ref[0:],(int(frame_size/2),2), order='F')
    qam_ref_shaped_extend = DMT_conj_map(qam_ref_shaped)
    qam_ref_extend = np.reshape(qam_ref_shaped_extend,(frame_size*2,),order='F')
    return (preamble384,qam_ref_extend)


qam_dec=[]
Mod = 16
for i in range(1920):
    qam_dec.append(randint(0,(Mod-1)))

dmt = DmtMod(symbols_dec=qam_dec, frame_len=192, qam_level=Mod)

dmt._interp_kind = 'linear' # 'quadratic'
new_sample_rate = 4
dmt.change_sample_rate(new_sample_rate)

N = 25

plt.plot(np.arange(N), dmt.wvfm_real[:N], 'ro-')
postinterp = dmt.samples[0:int(N*dmt.over_sample_ratio)-(new_sample_rate-1)]
plt.plot(np.linspace(0, N-1, num=len(postinterp)), postinterp, 'b*-')
plt.show()

plt.plot(np.arange(N), dmt.wvfm_real[-1*N:], 'ro-')
postinterp = dmt.samples[int(-1*N*dmt.over_sample_ratio):-1*(new_sample_rate-1)]
plt.plot(np.linspace(0, N-1, num=len(postinterp)), postinterp, 'b*-')

dmt_prmbl = DmtMod(symbols_dec = preamble_int192, frame_len = 192, qam_level=16)
dmt_prmbl_wvfm = dmt_prmbl.samples
dmt_prmbl_symbols_iq = dmt_prmbl.symbols_iq

preamble192 = preamble384[:192]
preamble_int96 = preamble_int192[:96]

from dmt_lib import DmtMod, DmtDeMod

prmbl = DmtDeMod(samples = dmt_prmbl_wvfm, frame_len=192, qam_level=16,
                 preamble = preamble_int96)

prmbl = DmtDeMod(samples = dmt.samples, frame_len=192, qam_level=16,
                 preamble = preamble_int96)

asdf=DmtDeMod()

wvfm0 = np.concatenate( (dmt_prmbl_wvfm[50:], dmt_prmbl_wvfm[:50] ) )
wvfm = np.concatenate( (wvfm0, wvfm0) )

prmbl.update(wvfm, re_calibrate = True)


N = 25
plt.plot(np.arange(N), dmt_prmbl_wvfm[:N], 'ro-')
plt.plot(np.arange(N), preamble384[:N], 'b*-')
plt.show()



asdf0 = dmt_prmbl.symbols_iq

asdf = prmbl.symbols_iq
plt.scatter(asdf.real,asdf.imag, s=5)

qamfigure,[txqam, rxqam] = plt.subplots(nrows=1,ncols=2)
qamfigure.set_size_inches((8,2.4))
txqam.scatter(np.array(asdf0).real, np.array(asdf0).imag, s=5)
txqam.set_position([0.125,0.125,0.26,0.80])
rxqam.scatter(asdf.real, asdf.imag, s=5)
rxqam.set_position([0.52,0.125,0.26,0.80])