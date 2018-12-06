""" TensorFlow utilities related to FFT padding and function selection """
import tensorflow as tf

OPM_LOG2 = 'LOG2'
OPM_NONE = 'NONE'
OPTIMAL_PAD_MODES = [OPM_NONE, OPM_LOG2]

PADF_REFLECT = 'REFLECT'
PADF_SYMMETRIC = 'SYMMETRIC'
PADF_ZERO = 'CONSTANT'
PAD_FILL_MODES = [PADF_REFLECT, PADF_SYMMETRIC, PADF_ZERO]


def get_fft_tf_fns(n_dims, real_domain_only=True):
    """Determine which TF functions should be used for FFT based on number of dimensions in data.
    
    Currently, TF only supports 1, 2, or 3 dimensional FFT operations
    """
    if n_dims == 1:
        if real_domain_only:
            return tf.spectral.rfft, tf.spectral.irfft
        else:
            return tf.spectral.fft, tf.spectral.ifft
    elif n_dims == 2:
        if real_domain_only:
            return tf.spectral.rfft2d, tf.spectral.irfft2d
        else:
            return tf.spectral.fft2d, tf.spectral.ifft2d
    elif n_dims == 3:
        if real_domain_only:
            return tf.spectral.rfft3d, tf.spectral.irfft3d
        else:
            return tf.spectral.fft3d, tf.spectral.ifft3d
    else:
        raise ValueError('Number of data dimensions must be <= 3')


def optimize_dims(dims, mode):
    """Compute FFT Length for data padding necessary to optimize FFT implementations.

    Since the primary GPU FFT implementation used by TensorFlow is cuFFT, optimal
    dimensions for arguments should be constructed following the guidelines
    provided by [Nvidia](http://docs.nvidia.com/cuda/cufft/index.html#accuracy-and-performance).

    It seems that like many FFT implementations, cuFFT will will attempt to use
    the more efficient Cooley-Tukey algorithm when the input dimensions are composite
    numbers constructed as products of the primes 2, 3, 5, and 7.  If dimensions to meet
    this criteria, then the slower Bluestein algorithm is used.

    In this case, support is added only for dimensions rounded up to the nearest power of 2
    (or no extra padding at all) though it may make sense in the future to add support
    for more complicated "products-of-primes" ala ```scipy.signal.signaltools.fftpack```.

    Args:
        dims: A 1-D integer tensor with dimensions to be optimized (e.g. results from `get_fft_pad_dims`)
        mode: Determines methodology used to extend dimensions to optimal values; one of:
            - "log2" - Will round dimensions up to next power of 2
            - "none" - Will leave dimensions unaltered
    Returns:

    """
    mode = mode.upper()

    # Round FFT Length up to next nearest optimal value based on mode given
    if mode == OPM_LOG2:
        # See ```_enclosing_power_of_two``` as used in
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/signal/python/ops/spectral_ops.py
        bases = tf.ceil(tf.log(tf.cast(dims, tf.float32)) / tf.log(2.0))
        return tf.cast(tf.pow(2.0, bases), dims.dtype)
    elif mode != OPM_NONE:
        raise ValueError('Padding mode "{}" invalid'.format(mode))
    return dims


def get_fft_pad_dims(data, kernel):
    """Compute "FFT Length" necessary for arrays to be convolved linearly.

    In each dimension associated with the given data, the FFT Length
    is computed as S1 + S2 - 1 where each size "S*" is the length
    along that dimension.  For example:
    
    ```python
    get_fft_pad_dims( np.ones((5, 10)), np.ones((2, 3)) ) = array([5 + 2 - 1, 10 + 3 - 1])
    ```

    Some motivation for this can be found [here](https://ccrma.stanford.edu/~jos/ReviewFourier/FFT_Convolution.html)
    which briefly explains why FFT based convolutions require padding dimensions in this way.  Otherwise,
    in the case of images at least, the convolved results will wrap edges around to the other side of the
    same image (i.e. a circular or cyclic convolution rather than a linear one).
    """
    return tf.shape(data) + tf.shape(kernel) - 1


def extract(data, base_dims, pad_dims):
    """Extracts the region of a convolution result matching dimensions of original input data array.
    
    This will account for padding added as part of the summation of dimensions as well as the rounding up
    nearest optimal size.  These are treated separately here to potentially allow for for different output
    "mode" settings as in scipy.signal.fftconvolve.

    Args:
        data: N-dimensional tensor resulting from convolution operation
        base_dims: 1-D tensor (of length N) containing shape of original data matrix sent through convolution
        pad_dims: 1-D tensor (of length N) with lengths along axes after padding for circular convolution
            (e.g. result from `get_fft_pad_dims`)
    Returns:
        N-dimensional tensor similar to data, but with zero-padded elements removed
    """
    ind_start = (pad_dims - base_dims) // 2

    data = tf.slice(data, tf.zeros_like(base_dims), pad_dims)

    data = tf.slice(data, ind_start, base_dims)

    return data


def convolve(data, kernel_fft, dims, fft_fwd, fft_rev):
    """ TF convolution operator via FFT """

    # Ensure all tensors are not empty (this can lead to python process
    # crashing instead of raising an error)
    with tf.control_dependencies([
        tf.assert_positive(tf.size(data)),
        tf.assert_positive(tf.size(kernel_fft)),
        tf.assert_positive(tf.size(dims)),
    ]):
        data_fft = fft_fwd(data, fft_length=dims)
        data_conv = fft_rev(tf.multiply(data_fft, kernel_fft), fft_length=dims)
        return data_conv


def fftshift(arr, ndims=None):
    """fftshift operator ala np.fft.fftshift

    Args:
        arr: Array to swap half-spaces for
        ndims: Number of dimensions to expect in tensor; defaults to None implying that an attempt will
        be made to infer this from the tensor (which does not always work for dynamic placeholders)
    Returns:
        Array of same dimensions (see np.fft.fftshift for more details)
    """
    res = arr
    for k in range(arr.shape.ndims):
        n = tf.shape(arr)[k]
        m = (n + 1) // 2
        idx = tf.concat([tf.range(m, n), tf.range(0, m)], axis=0)
        res = tf.gather(res, idx, axis=k)
    return res


def ifftshift(arr, ndims=None):
    """fftshift operator ala np.fft.ifftshift

    Args:
        arr: Array to swap half-spaces for
        ndims: Number of dimensions to expect in tensor; defaults to None implying that an attempt will
        be made to infer this from the tensor (which does not always work for dynamic placeholders)
    Returns:
        Array of same dimensions (see np.fft.ifftshift for more details)
    """
    res = arr
    for k in range(arr.shape.ndims):
        n = tf.shape(arr)[k]
        m = n - (n + 1) // 2
        idx = tf.concat([tf.range(m, n), tf.range(0, m)], axis=0)
        res = tf.gather(res, idx, axis=k)
    return res
