""" Dataset manager for fetching and representing pre-defined images shipping with this package """
import os
import os.path as osp
import zipfile
import flowdec
import requests
import numpy as np
import logging
from skimage import io
logger = logging.getLogger(__name__)


class Acquisition(object):
    """ Data model for measured quantities to be deconvolved """

    def __init__(self, data, kernel=None, actual=None):
        """New acquisition instance
        Args:
            data: Observed data; e.g. a measured image from a microscope
            kernel: Optional kernel assumed to have been used in generating the observed data (i.e. a PSF)
            actual: Optional ground-truth data useful for synthetic tests and validation
        """
        if data.ndim not in [1, 2, 3]:
            raise ValueError('Number of data and kernel dimensions must be 1, 2, or 3')
        self.data = data
        self.kernel = kernel
        self.actual = actual

    def to_feed_dict(self):
        return {'data': self.data, 'kernel': self.kernel}

    def shape(self):
        return self.transform(lambda d: d.shape)

    def dtype(self):
        return self.transform(lambda d: d.dtype)

    def stats(self):
        from scipy.stats import describe
        return self.transform(lambda v: describe(v.ravel()))

    def apply(self, fn, **kwargs):
        return Acquisition(
            data=fn(self.data, **kwargs),
            kernel=None if self.kernel is None else fn(self.kernel, **kwargs),
            actual=None if self.actual is None else fn(self.actual, **kwargs)
        )

    def transform(self, fn, **kwargs):
        return {
            'data': fn(self.data, **kwargs),
            'kernel': None if self.kernel is None else fn(self.kernel, **kwargs),
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



################################
#  Reference Dataset Utilities #
################################


# External Datasets

DATA_DIR_DEFAULT = '~/.flowdec/datasets'
_DATA_DIR = os.getenv('FLOWDEC_DATA_DIR', DATA_DIR_DEFAULT)


def set_external_data_dir(path):
    """Assign data directory manually

    Otherwise, this will be inferred from the environment variable "FLOWDEC_DATA_DIR" and if
    that is not set will default to `DATA_DIR_DEFAULT`
    Args:
        path: Path containing image data to use for validation and experimentation
    """
    global _DATA_DIR
    _DATA_DIR = path


def get_external_data_dir():
    global _DATA_DIR
    return _DATA_DIR


def _download_google_drive_file(id, destination):
    """ Utility function for downloading Google Drive files

    This is necessary to deal with the fact that larger files cannot be linked
    to directly as a redirect prompt about virus scanning is returning instead.

    This function will detect when that happens and generate a new request based
    on the redirect in order to execute the proper download
    Args:
        id: Google Drive file id
        destination: File path to download data to
    """

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def _load_external_dataset(name, gdrive_id, img_dirs=None):
    """Get external dataset by name, fetching it if not already present"""

    # Initialize location in which to store external data
    data_dir = os.path.expanduser(_DATA_DIR)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=False)

    # Fetch zip archive containing data if not already present
    if not os.path.exists(osp.join(data_dir, name)):
        zip_file = osp.join(data_dir, name + '.zip')
        logger.info('Downloading archive for dataset "{}" (will only occur on first reference) ...'.format(name))
        _download_google_drive_file(gdrive_id, zip_file)
        logger.info('Download for dataset "{}" complete'.format(name))
        with zipfile.ZipFile(zip_file, 'r') as zipf:
            zipf.extractall(data_dir)
        os.unlink(zip_file)

    # Load tif stacks into acquisition instance
    data_dir = osp.join(data_dir, name)

    def get_image(k):
        # Load image for acquisition part based on per-directory separation as individual tif
        # stacks if such a separation is defined
        if img_dirs is not None:
            if k not in img_dirs:
                return None
            path = osp.join(data_dir, img_dirs[k], '*.tif')
            return load_img_stack(path)
        # If no subdirectories are given, assume the dataset has already been consolidated
        # into individual tif stacks
        else:
            path = osp.join(data_dir, k + '.tif')
            return io.imread(path) if osp.exists(path) else None

    return Acquisition(**{k: get_image(k) for k in ['data', 'kernel', 'actual']})


def load_bars():
    """Get data for "Hollow Bars" dataset"""
    gdrive_id = '1Av3h3hefgSYnL_bhI0_omNX1wcihtQDc'
    img_dirs = {'data': 'Bars-G10-P30', 'kernel': 'PSF-Bars', 'actual': 'Bars'}
    return _load_external_dataset('bars', gdrive_id, img_dirs)


def load_bead():
    """Get data for "Fluorescent Bead" dataset"""
    gdrive_id = '1kMZB-VDl46PLLn41t81vvTNDNoxrL2dd'
    img_dirs = {'data': 'Bead', 'kernel': 'PSF-Bead'}
    return _load_external_dataset('bead', gdrive_id, img_dirs)


def load_microtubules():
    """Get data for "Microtubules" dataset"""
    gdrive_id = '1YFH2Fugii-owyR1YVkeSmR0Z0izp7EwS'
    return _load_external_dataset('microtubules', gdrive_id)


def load_neuron():
    """Get data for "Purkinje Neuron" dataset"""
    gdrive_id = '1QDrXTtkybKfKuaAzl7XmRWfOSrLVQR_H'
    return _load_external_dataset('neuron', gdrive_id)


CELEGANS_CHANNELS = ['CY3', 'DAPI', 'FITC']


def load_celegans():
    """ Fetch C. Elegans dataset as 3 separate Acquisition instances, in a dict keyed by channel name

    Example:
        data = load_celegans()
        data.keys() -> ['CY3', 'FITC', 'DAPI']
        data['CY3'].shape() -> {'actual': None, 'data': (104, 712, 672), 'kernel': (104, 712, 672)}
    Returns:
        3 acquisitions in a dict keyed by channel name
    """
    from collections import OrderedDict
    return OrderedDict({ch: load_celegans_channel(ch) for ch in CELEGANS_CHANNELS})


def load_celegans_channel(channel):
    """Get single channel data for "C. Elegans" dataset
    Args:
        channel: Name of channel to download tiff stacks for (one of ['CY3', 'DAPI', 'FITC'])
    """
    if not channel in CELEGANS_CHANNELS:
        raise ValueError("Channel name must be one of {} (given = {})".format(CELEGANS_CHANNELS, channel))
    gdrive_id = '19bdJA7oRvFAW2NMM937uAdViI0O6QtQ5'
    img_dirs = {'data': 'CElegans-{}'.format(channel), 'kernel': 'PSF-CElegans-{}'.format(channel)}
    return _load_external_dataset('celegans', gdrive_id, img_dirs)


# Repository Datasets

def _get_dataset_path(path):
    """Create absolute path from relative path starting in datasets folder"""
    return os.path.join(flowdec.data_dir, path)


def _load_repo_dataset(name):
    """Get dataset by name"""

    data_dir = _get_dataset_path(name)
    if not os.path.exists(data_dir):
        raise ValueError('Dataset "{}" not found (path "{}" does not exist)'.format(name, data_dir))

    return Acquisition(**{
        k: (io.imread(osp.join(data_dir, k + '.tif')) if osp.exists(osp.join(data_dir, k + '.tif')) else None)
        for k in ['data', 'kernel', 'actual']
    })


def bars_25pct():
    """Load "Hollow Bars" dataset downsampled to 25% of original"""
    return _load_repo_dataset('bars-25pct')


def bead_25pct():
    """Load "Bead" dataset downsampled to 25% of original"""
    return _load_repo_dataset('bead-25pct')

def neuron_25pct():
    """Load "Purkinje Neuron" dataset downsampled to 25% of original"""
    return _load_repo_dataset('neuron-25pct')
