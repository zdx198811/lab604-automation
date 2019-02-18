# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 18:30:50 2019
Description:
    A class wrapper for descrete-multi-tone (DMT). Created from the scripts
    used in the fronthaul ptototype.
    
@author: Dongxu ZHANG (dongxu.c.zhang@nokia-sbell.com)
"""

from math import gcd
from scipy import interpolate
import numpy as np
# from bitstring import BitArray
# import matplotlib.pyplot as plt

def qam4_to_complex(qamdata_dec):
    comp_set = np.array([-1+1j, -1-1j, 1+1j, 1-1j])
    return comp_set[qamdata_dec]


def qam16_to_complex(qamdata_dec):
    comp_set = np.array([-3+3j, -3+1j, -3-3j, -3-1j, -1+3j, -1+1j, -1-3j,
                         -1-1j, 3+3j, 3+1j, 3-3j, 3-1j, 1+3j, 1+1j, 1-3j,
                         1-1j])/3
    return comp_set[qamdata_dec]


def qam64_to_complex(qamdata_dec):
    comp_set_base = np.array([-3+3j, -3+1j, -3-3j, -3-1j, -1+3j, -1+1j, -1-3j,
                              -1-1j, 3+3j, 3+1j, 3-3j, 3-1j, 1+3j, 1+1j, 1-3j,
                              1-1j])
    comp_set_1 = [item+(4+4j) for item in comp_set_base]
    comp_set_2 = [item+(4-4j) for item in comp_set_base]
    comp_set_3 = [item+(-4-4j) for item in comp_set_base]
    comp_set_4 = [item+(-4+4j) for item in comp_set_base]
    comp_set = np.array(comp_set_1+comp_set_2+comp_set_3+comp_set_4)/7
    return comp_set[qamdata_dec]

    
class DmtCommon:
    """ base class for DMT modulation/demodulation tools 
    
    Variables:
        samples -- raw tx or rx sample data (to DAC, or from ADC)
        nsample -- len(samples)
        symbols_dec -- an array of intger numbers (payload)
        symbols_iq -- an array of complex numbers (payload)
        symbols -- post-ifft symbols, complex value, doubled size of payload
        wvfm_real -- = symbols.real; samples = interpolated wvfm_real
        nsymbol -- = len(wvfm_real) = 2*len(symbols_iq)
        sample_rate -- GHz (working speed of DAC or ADC)
        symbol_rate -- GHz (DMT time-domain symbol rate, a.k.a. 2*payload_rate)
        over_sample_ratio -- sample_rate/symbol_rate
        frame_len -- number of symbols per frame. Note that DMT's frame size
                     equals 2*n_effecitve_subcarrier
        nframe -- int(nsymbol/frame_len)

    methods:
        qam_dec_to_complex(m, qam_dec) -- convert int to complex
        _interp_ratio_calc(origin_fs, target_fs)
    
    """
    def __init__(self, sample_rate = 1, symbol_rate = 1, frame_len = 2,
                 qam_level = 4):
        np.fft.restore_all()
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.over_sample_ratio = sample_rate/symbol_rate
        if ((frame_len % 2) != 0):
            raise ValueError("frame_len MUST BE EVEN: 2*effecitve_subcarriers")
        self.frame_len = frame_len
        self.samples = []
        self.symbols_dec = []
        self.symbols_iq = []
        self.symbols = []
        self.nsample = 0
        self.nsymbol = 0
        self.qam_level = qam_level
        
    def _correlation(self, a, b):
        if len(a) != len(b):
            raise ValueError('corr error, len a != len b')
        else:
            return np.matmul(a,b) #sum([a[i]*b[i] for i in range(len(a))])

    def _corr_result_list(self, rawlist, reflist):
        if len(reflist)>len(rawlist):
            raise ValueError('_corr_result_list error, len ref > len raw')
        else:
            reflen = len(reflist)
            rawlen = len(rawlist)
            return [np.abs(self._correlation(reflist, rawlist[i:i+reflen])) 
                    for i in range(rawlen-reflen+1)]

    def _qam_conversion_func(self):
        """ return a function that converts QAMm symbol form int to (I+Qj)"""
        if self.qam_level == 4:
            return qam4_to_complex
        elif self.qam_level == 16:
            return qam16_to_complex
        elif self.qam_level == 64:
            return qam64_to_complex
        else:
            raise ValueError('Only support qam 4/16/64.')
    
    def DMT_conj_map(self, qam_shaped):
        qam_conj_extend = qam_shaped.tolist()
        N = len(qam_conj_extend)
        for i in range(N):
            if i==0:
                qam_conj_extend.append((qam_shaped[0].imag).tolist())
                for j in range(len(qam_shaped[0])):
                    qam_conj_extend[0][j]=qam_conj_extend[0][j].real
            else:
                qam_conj_extend.append(np.conj(qam_shaped[N-i]).tolist())
        return np.array(qam_conj_extend)

    def DMT_conj_demap(self, qam_extend):
        qam_demap = qam_extend.tolist()
        (N,L) = np.shape(qam_demap)
        for i in range(L):
            qam_demap[0][i] = qam_extend[0,i].real + 1j*qam_extend[int(N/2),i].real
        return np.array(qam_demap[:int(N/2)])

    def _interp_ratio_calc(self, origin_fs, target_fs):
        if (origin_fs != int(origin_fs)):
            origin_fs_str = str(origin_fs)
            pow_idx = len(origin_fs_str) - origin_fs_str.find('.') - 1 
            origin_fs = origin_fs*pow(10,pow_idx)
            target_fs = target_fs*pow(10,pow_idx)
        if (target_fs != int(target_fs)):
            target_fs_str = str(target_fs)
            pow_idx = len(target_fs_str) - target_fs_str.find('.') - 1
            origin_fs = origin_fs*pow(10,pow_idx)
            target_fs = target_fs*pow(10,pow_idx)
        origin_fs = int(origin_fs)
        target_fs = int(target_fs)
        fs_gcd = gcd(origin_fs,target_fs)
        origin_fs = int(origin_fs/fs_gcd)
        target_fs = int(target_fs/fs_gcd)
        fs_lcm = origin_fs * target_fs / gcd(origin_fs, target_fs)
        return (int(fs_lcm/origin_fs),int(fs_lcm/target_fs))


class DmtMod(DmtCommon):
    """ DMT modulator

    Store dmt symbols and calculate waveforms for DAC.

    Variables:
        _interp_kind -- interpolation method. Refer to scipy.interolate
                        default to 'nearest'. Others include: 'linear',
                        'zero', 'slinear', 'quadratic', 'cubic'
        wvfm_real -- time domain waveform @ symbol_rate

    Methods:
        _tx_interp(sample_rate, interp_kind) -- interpolate wvfm_real.
        change_sample_rate(new_sample_rate) -- update and return self.samples
    """
    def __init__(self, symbols_dec = [0,0], sample_rate = 1,
                 symbol_rate = 1, frame_len = 2, qam_level = 4):
        """ generate waveform based on the given list of decimal numbers
        
        Input Arguments:
            symbols_dec -- Decimal int list. Valid value range 0 ~ qam_level-1
                           The tailing symbols that cannot compose a whole 
                           frame will be discarded.
            sample_rate -- GHz. Default = 1
            symbol_rate -- GHz. Default = 1
            frame_len -- an even integer, = 2*effective_subcarriers
            qam_level -- default = 4
        """
        DmtCommon.__init__(self, sample_rate, symbol_rate, frame_len, qam_level)
        self._interp_kind = 'nearest'
        self.nframe = int(len(symbols_dec)/(frame_len/2))
        self.nsymbol = self.nframe*self.frame_len
        nsymbol_effective = int(self.nsymbol/2)  # DMT feature
        self.symbols_dec = symbols_dec[:nsymbol_effective]  # discard tail
        
        try:  # map decimal qam symbols into (I+Qj) form
            self.symbols_iq = list(map(self._qam_conversion_func(),
                                    self.symbols_dec))
        except IndexError as err:
            err_str = ("Decimal qam symbol index INVALID. " + str(err))
            raise ValueError(err_str)

        # get the DMT signal using ifft algorithm
        symbols_iq_shaped = np.reshape(self.symbols_iq[0:],
                                (int(self.frame_len/2), self.nframe),
                                order='F')
        self.symbols = self.DMT_conj_map(symbols_iq_shaped)
        wvfm_shaped = np.fft.ifft(self.symbols, self.frame_len, axis = 0)
        wvfm = np.reshape(wvfm_shaped,(self.nsymbol,),order='F')
        self.wvfm_real = wvfm.real
        # wvfm_imag = wvfm.imag
        
        # get the over-sampled DMT samples using interpolation
        samples_ = self._tx_interp(sample_rate, self._interp_kind)
        
        # normalize the amplitude.
        self.samples = self._amplitude_norm(samples_)
        self.nsample = len(self.samples)

    def _tx_interp(self, sample_rate, interp_kind):
        """ oversample self.wvfm_real to match sample_rate
        
        Be careful with the interpolate function which may cut-off the last
        data point (depend on specific arguments, see docs of interpolate
        module). Here we assume the waveform repeat itself, and concatenate
        the first point of the frame to the end of the frame itself when
        interpolating. In other cases where different frames are to be joint,
        this self-concatenating scheme may introduce a little inaccuracy.
        """
        self.sample_rate = sample_rate
        x_origin = np.arange(0, self.nsymbol+1)
        x_interp = np.arange(0, self.nsymbol, 
                            (self.nsymbol+1)/((self.nsymbol+1)*self.over_sample_ratio))
        f_interp = interpolate.interp1d(x_origin,
                                        np.concatenate((self.wvfm_real, [self.wvfm_real[0]])),
                                        kind = interp_kind)
        return f_interp(x_interp)

    def _amplitude_norm(self, x):
        """ Normalize the amplitude to 1. """
        return np.array(x)/max(np.array(x))
    
    def change_sample_rate(self, new_sample_rate):
        self.over_sample_ratio = new_sample_rate/self.symbol_rate
        self.samples = self._tx_interp(new_sample_rate, self._interp_kind)
        self.nsample = len(self.samples)
        return self.samples


class DmtDeMod(DmtCommon):
    """ DMT De-modulator

    Store waveform (from ADC) and calculate corresponding qam symbols. The
    basic processing flow is:
    Resample(opt.) -> Correlation -> Downsample (to symbol_rate) -> De-Mod ->
    Equalization(opt.) -> De-mapping
    
    Variables:
        _resample_kind -- interpolation method. Refer to scipy.interolate docs.
            default to 'slinear'.
        preamble -- decimal value (payloads), derive _p_symbols and _p_samples.
            Used for correlation and equalization. If it is not given when
            initializing, the update() function simply tries to start from the
            first sample, and equalization is skipped.
        samples -- data array directly received from ADC
        nsample -- len(samples), not that it does not necessarily have relation
            to the frame_len, because the data chunk from ADC may be arbitrary.
        wvfm --
        nsymbol -- number of DMT symbols. = frame_len*nframe
        equ_coef -- ndarray, equalization coefficiency
        best_groupid -- the best relative position when down-sampling
        best_pos -- the best position to truncate samples
        prmbl_frm_ofst -- the offset (# of frame) where the preamble starts. It
            is used to extract preamble samples after stripping off the heading
            garbage from the raw samples.
        
    Methods:
        __init__ -- 
        update(samples, re_calibrate) -- do the demodulation process. If 
            re_calibrate is set to True, redo correlation @ update equ_coef.
        _load_preamble(preamble)
    """    
    def __init__(self, samples = [0,0], sample_rate = 1, symbol_rate = 1, 
                 frame_len = 2, qam_level = 4, preamble = None):
        DmtCommon.__init__(self, sample_rate, symbol_rate, frame_len, qam_level)
        self._resample_kind = 'slinear'
        self.nsample = len(samples)
        self.preamble = preamble
        self.equ_coef = np.ones(self.frame_len)
        if preamble:
            if (len(preamble) % int(frame_len/2)) != 0 :
                raise ValueError("preamble length error!")
            else:
                self.set_preamble(preamble)
            self.update(samples, re_calibrate = True)
        else:  # no preamble, no equalization
            self.best_groupid = 0
            self.best_pos = 0
            self.update(samples, re_calibrate = False)
    
    def set_preamble(self, preamble):
        print('set new preamble')
        self._p_symbols, self._p_samples = self._load_preamble(preamble)
    
    def _load_preamble(self, preamble):
        p_dmt = DmtMod(preamble, frame_len = self.frame_len,
                       qam_level = self.qam_level)
        p_samples = p_dmt.samples
        p_symbols = np.reshape(p_dmt.symbols, (p_dmt.symbols.size, ),
                               order = 'F')
        return (p_symbols, p_samples)
    
    def update(self, samples, re_calibrate = False):
        """ Update samples and demodulate based on the new samples.
        
        If re_calibrate is set to True, the preamble is used to re-align frame
        head and re-calculate equalization coefficient. Otherwise (default = 
        False), use the previous head and coefficient.
        """
        self.samples = samples
        self.nsample = len(samples)
        # Resample the signal to match symbol_rate.
        # First, upsample the signal to make sure an apropriate length
        x_origin = np.arange(0, self.nsample + 1)
        (ratio_up, ratio_down) = self._interp_ratio_calc(self.sample_rate,
                                                         self.symbol_rate)
        x_interp = np.arange(0, self.nsample,
                             (self.nsample+1)/((self.nsample+1)*ratio_up))
        f_interp = interpolate.interp1d(x_origin,
                                        np.concatenate((samples, [0])),
                                        self._resample_kind)
        wvfm_interp = f_interp(x_interp)

        # Then, downsample. (extract best_wvfm and find self.prmbl_frm_ofst)
        extract_len = int(len(wvfm_interp)/ratio_down)
        if re_calibrate:
            print('Realign frames.')
            best_corr_value  = 0
            for i in range(ratio_down):
                # for each candidate down_sample group, calculate correlation
                wvfm_i = wvfm_interp.take(
                        [ratio_down * j + i for j in range(extract_len)])
                corr_results = np.round(
                        self._corr_result_list(wvfm_i, self._p_samples))
                # plt.plot(corr_results)
                best_pos_i = np.argmax(corr_results)
                max_corr_value = corr_results[best_pos_i]
                print(best_pos_i, best_pos_i % self.frame_len, max_corr_value)
                if (max_corr_value > best_corr_value):
                    best_corr_value = max_corr_value
                    self.best_groupid = i
                    self.best_pos = best_pos_i % self.frame_len
                    preamble_frame_pos = int(best_pos_i/self.frame_len)
                    # print(preamble_frame_pos, best_pos_i)
                    self.prmbl_frm_ofst = preamble_frame_pos
        else:  # does not need to re-calibrate
            print("use previous frame offset")
            wvfm_i = wvfm_interp.take(
                [ratio_down * j + self.best_groupid for j in range(extract_len)])
        wvfm_best = wvfm_i.take(list(range(self.best_pos, len(wvfm_i))))
        # de-modulate the extracted DMT signal
        # first, do fft 
        self.nframe = int(len(wvfm_best)/self.frame_len)
        self.nsymbol = self.nframe * self.frame_len
        wvfm_shaped = np.reshape(wvfm_best[:self.nsymbol],
                                 (self.frame_len, self.nframe),
                                 order='F')
        symbols_shaped = np.fft.fft(wvfm_shaped, self.frame_len, axis = 0)
        rx_symbols = np.reshape(symbols_shaped, (self.nsymbol,), order='F')
        # then, do equalization
        if (re_calibrate):   #False: 
            # extract preamble (I+Qj) symbols
            prmbl_pos = self.frame_len * self.prmbl_frm_ofst
            rx_p_symbol = rx_symbols[
                      prmbl_pos : prmbl_pos + len(self._p_symbols)]
            l = self.frame_len
            p_nf = int(len(self._p_symbols) / self.frame_len)  # frames in preamble
            self.equ_coef = np.array(
                [np.mean([rx_p_symbol[i + l * j] / self._p_symbols[i + l * j] for j in range(p_nf)]) for i in range(l)])
    
        self.symbols = np.array([rx_symbols[i] / self.equ_coef[i % self.frame_len]
                                 for i in range(self.nsymbol)])
        symbols_shaped_postequ = np.reshape(self.symbols,
                                            (self.frame_len, self.nframe),
                                            order = 'F')
        # at last, dmt de-mapping
        self.symbols_iq_shaped = self.DMT_conj_demap(symbols_shaped_postequ)
        (N,L) = np.shape(self.symbols_iq_shaped)
        self.symbols_iq = np.reshape(self.symbols_iq_shaped, (N*L,), order='F')

        





