""" Utilities for performing data validation and analysis """
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from flowdec import exec as fd_exec
from flowdec.fft_utils_tf import OPTIMAL_PAD_MODES, OPM_LOG2
from skimage.transform import resize
from scipy.ndimage.interpolation import shift as scipy_shift
from skimage.measure import compare_ssim
from scipy.signal import fftconvolve
import numpy as np


def mutate(d, data_fn=None, kern_fn=None):
    """Apply functions data and/or kernel function to acquisition"""
    return fd_data.Acquisition(
        data=data_fn(d.data) if data_fn else d.data,
        actual=data_fn(d.actual) if data_fn else d.actual,
        kernel=kern_fn(d.kernel) if kern_fn else d.kernel,
    )


def shift(acq, data_shift=None, kern_shift=None):
    """Apply translation to acquisition data"""
    return mutate(
        acq, data_fn=None if not data_shift else lambda d: scipy_shift(d, data_shift),
        kern_fn=None if not kern_shift else lambda k: scipy_shift(k, kern_shift)
    )


def subset(acq, data_slice=None, kern_slice=None):
    """Apply slice operation to acquisition data"""
    return mutate(acq,
        data_fn=None if not data_slice else lambda d: d[data_slice],
        kern_fn=None if not kern_slice else lambda k: k[kern_slice]
    )


def downsample(acq, data_factor=None, kern_factor=None):
    """Downsample acquisition data by the given factors"""
    def resize_fn(img, factor):
        return resize(
            img, [int(sz * factor) for sz in img.shape], mode='constant',
            anti_aliasing=True, order=1, preserve_range=True
        ).astype(img.dtype)
    return mutate(acq,
        data_fn=None if not data_factor else lambda d: resize_fn(d, data_factor),
        kern_fn=None if not kern_factor else lambda k: resize_fn(k, kern_factor)
    )


def decon_tf(acq, n_iter, **kwargs):
    return fd_restoration.richardson_lucy(acq, n_iter, **kwargs)


def decon_dl2(acq, n_iter, pad_mode):
    return fd_exec.run_dl2(acq, n_iter, pad_mode)


def binarize(img):
    """Convert image to binary based on mean-threshold"""
    return (img > img.mean()).astype(np.float32)


def score(img_pred, img_true):
    """Convert similarity score between images to validate"""
    return compare_ssim(img_pred, img_true, data_range=img_true.max() - img_true.min())


def reblur(acq, scale=.05, seed=1):
    """Apply blurring operation to the ground-truth data in an acquisition

    This operation works by convolving the ground-truth image with the configured kernel and then
    adding poisson noise

    Args:
        acq: Acquisition to blur
        scale: Fraction of min/max value range of acquisition ground-truth image to use as standard deviation in
            poisson noise
        seed: Seed for poisson noise generation
    Result:
        New acquisition object with same ground-truth and kernel, but newly assigned blurred data
    """
    sd = scale * (acq.actual.max() - acq.actual.min())
    np.random.seed(seed)
    noise = np.random.poisson(sd, size=acq.actual.shape)
    kernel = acq.kernel / acq.kernel.sum()  # Normalize to 0-1
    data = fftconvolve(acq.actual, kernel, 'same') + noise
    return fd_data.Acquisition(
        data=data.astype(acq.data.dtype),
        kernel=acq.kernel,
        actual=acq.actual
    )


def run_deconvolutions(acq, n_iter, dl2=False, dtype=None):
    """ Perform deconvolution using several different implementations

    Args:
        acq: Acquisition to deconvolve
        n_iter: Number of iterations to use
        dl2: Whether or not to include DeconvolutionLab2 implementation
        dtype: Data type of original image (used to determine value ranges)
    """
    res = {'data': {}, 'scores': {}, 'acquisition': acq}

    if dtype is None:
        dtype = acq.data.dtype
    clip_range = np.iinfo(dtype).min, np.iinfo(dtype).max

    # Create result for each padding mode
    for pad_mode in OPTIMAL_PAD_MODES:
        res['data']['tf_' + pad_mode] = decon_tf(acq, n_iter, pad_mode=pad_mode).clip(*clip_range)

    if dl2:
        res['data']['dl2'] = decon_dl2(acq, n_iter, pad_mode=OPM_LOG2)

    # Compute similarity score between blurred image and ground-truth
    res['scores']['original'] = score(acq.data, acq.actual)

    # Compute similarity scores between deconvolved results and ground-truth
    for k in res['data'].keys():
        res['scores'][k] = score(res['data'][k], acq.actual)

    return res