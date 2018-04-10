""" Utilities for performing data validation and analysis """
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from flowdec import exec as fd_exec
from flowdec.fft_utils_tf import OPM_LOG2
from skimage import restoration as sk_restoration
from skimage.transform import resize
from skimage.exposure import rescale_intensity
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
        img = rescale_intensity(img, out_range=(0., 1.))
        return resize(img, [int(sz * factor) for sz in img.shape], mode='constant')
    return mutate(acq,
        data_fn=None if not data_factor else lambda d: resize_fn(d, data_factor),
        kern_fn=None if not kern_factor else lambda k: resize_fn(k, kern_factor)
    )


def decon_tf(acq, n_iter, **kwargs):
    return fd_restoration.richardson_lucy(acq, n_iter, **kwargs)


def decon_sk(acq, n_iter):
    return sk_restoration.richardson_lucy(acq.data, acq.kernel, iterations=n_iter, clip=False)


def decon_dl2(acq, n_iter, pad_mode):
    return fd_exec.run_dl2(acq, n_iter, pad_mode)


def binarize(img):
    """Convert image to binary based on mean-threshold"""
    return (img > img.mean()).astype(np.float32)


def score(img_pred, img_true):
    """Convert similarity score between images to validate"""
    return compare_ssim(img_pred.max(axis=0), img_true.max(axis=0))


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
    data = fftconvolve(acq.actual, acq.kernel, 'same') + noise
    return fd_data.Acquisition(
        data=data.astype(acq.data.dtype),
        kernel=acq.kernel,
        actual=acq.actual
    )


def run_deconvolutions(acq, n_iter, dl2=False):
    """ Perform deconvolution using several different implementations

    Args:
        acq: Acquisition to deconvolve
        n_iter: Number of iterations to use
        dl2: Whether or not to include DeconvolutionLab2 implementation
    """
    res = {'data': {}, 'scores': {}, 'acquisition': acq}
    res['data']['tf'] = decon_tf(acq, n_iter, pad_mode=OPM_LOG2)
    res['data']['sk'] = decon_sk(acq, n_iter)
    if dl2:
        res['data']['dl2'] = decon_dl2(acq, n_iter, pad_mode=OPM_LOG2)

    # Compute similarity score between blurred image and ground-truth
    res['scores']['original'] = score(binarize(acq.data), binarize(acq.actual))

    # Compute similarity scores between deconvolved results and ground-truth
    for k in res['data'].keys():
        res['scores'][k] = score(binarize(res['data'][k]), binarize(acq.actual))

    return res