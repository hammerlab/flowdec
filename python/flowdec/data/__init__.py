""" Dataset manager for fetching and representing pre-defined images shipping with this package """
import os
import tfdecon
import numpy as np
from skimage import io


class Acquisition(object):
    """ Data model for measured quantities to be deconvolved """

    def __init__(self, data, kernel, actual=None):
        """New acquisition instance
        Args:
            data: Observed data; e.g. a measured image from a microscope
            kernel: Kernel assumed to have been used in generating the observed data (i.e. a point spread function)
            actual: Optional ground-truth data useful for synthetic tests and validation
        """
        if data.ndim != kernel.ndim:
            raise ValueError('Dimensions of data and kernel must match (i.e. both 1D, 2D, or 3D)')
        if not data.ndim in [1, 2, 3]:
            raise ValueError('Number of data and kernel dimensions must be 1, 2, or 3')
        self.data = data
        self.kernel = kernel
        self.actual = actual

    def to_feed_dict(self):
        return {'data': self.data, 'kernel': self.kernel}

    def shape(self):
        return self.transform(lambda d: d.shape)

    def stats(self):
        from scipy.stats import describe
        return self.transform(lambda v: describe(v.ravel()))

    def apply(self, fn, **kwargs):
        return Acquisition(
            data=fn(self.data, **kwargs),
            kernel=fn(self.kernel, **kwargs),
            actual=None if self.actual is None else fn(self.actual, **kwargs)
        )

    def transform(self, fn, **kwargs):
         return {
            'data': fn(self.data, **kwargs),
            'kernel': fn(self.kernel, **kwargs),
            'actual': None if self.actual is None else fn(self.actual, **kwargs)
        }

    def copy(self):
        return self.apply(np.copy)


def downsample_acquisition(acquisition, factor, **kwargs):
    """Downsample dataset by a factor of `factor`"""
    if not 0 < factor <= 1:
        raise ValueError('Downsampling factor must be in (0, 1] (given "{}")'.format(factor))

    from skimage.transform import resize

    # Force setting of "mode" parameter to avoid UserWarning
    if 'mode' not in kwargs:
        kwargs['mode'] = 'constant'

    _rescale = lambda img: resize(img, [int(sz * factor) for sz in img.shape], **kwargs)
    return acquisition.apply(_rescale)


def load_img_stack(path):
    """Load multiple single-channel image files matching the given path expression into a concatenated array

    Note that this is intended for use with single channel images; if more channels are present use skimage.io
    functions directly.

    Args:
        path: Path expression compatible with ```skimage.io.imread_collection```
    Returns:
        3 dimensional numpy array with axes [z, x, y] where z is z-coordinate of images and
        x, y are pixel locations
    """
    img = io.imread_collection(path)
    return io.concatenate_images(img)


def _load_dataset(name):
    """Get dataset by name"""

    data_dir = os.path.join(tfdecon.data_dir, name)
    if not os.path.exists(data_dir):
        raise ValueError('Dataset "{}" not found (path "{}" does not exist)'.format(name, data_dir))

    actp = os.path.join(data_dir, 'actual.tif')
    return Acquisition(
        data=io.imread(os.path.join(data_dir, 'data.tif')),
        kernel=io.imread(os.path.join(data_dir, 'kernel.tif')),
        actual=io.imread(actp) if os.path.exists(actp) else None
    )


def bars_25pct():
    """Load "Hollow Bars" dataset downsampled to 25% of original"""
    return _load_dataset('bars-25pct')


def bead_25pct():
    """Load "Bead" dataset downsampled to 25% of original"""
    return _load_dataset('bead-25pct')


def bead_18pct():
    """Load "Bead" dataset downsampled to 18% of original"""
    return _load_dataset('bead-18pct')
