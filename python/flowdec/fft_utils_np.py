""" Numpy implementations of various FFT utilities used for testing and validation """
import numpy as np

OPM_FFTP = 'FFTPACK'
OPM_LOG2 = 'LOG2'
OPM_NONE = 'NONE'
OPTIMAL_PAD_MODES = [OPM_NONE, OPM_LOG2, OPM_FFTP]


def optimize_dims(dims, mode):
    """Computes FFT Length for data padded out to optimize FFT implementations"""
    from scipy.signal.signaltools import fftpack
    mode = mode.upper()

    # Round FFT Length up to next nearest optimal value based on mode given
    if mode == OPM_LOG2:
        return 2 ** np.ceil(np.log2(dims)).astype(int)
    elif mode == OPM_FFTP:
        return np.array([fftpack.helper.next_fast_len(int(sz)) for sz in dims])
    elif mode != OPM_NONE:
        raise ValueError('Padding mode "{}" invalid'.format(mode))
    return dims


def get_fft_pad_dims(data, kernel):
    """Compute "FFT Length" necessary for arrays to be convolved linearly"""
    return np.array(data.shape) + np.array(kernel.shape) - 1


def convolve(data, kernel):
    """ TF convolution operator via FFT """
    from scipy.signal import fftconvolve
    return fftconvolve(data, kernel, mode='same')


def extract(data, base_dims, pad_dims):
    """Determine tensor slices used to cut off padding added to FFT inputs"""
    ind_start = (pad_dims - base_dims) // 2
    ind_end = ind_start + base_dims
    pad_slice = tuple([slice(ind_start[i], ind_end[i]) for i in range(len(ind_end))])
    pad_slice_optim = tuple([slice(0, int(sz)) for sz in pad_dims])

    return data[pad_slice_optim][pad_slice]







