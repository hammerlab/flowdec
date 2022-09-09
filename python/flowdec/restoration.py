""" Deconvolution algorithm implementations """
import abc
import tensorflow as tf
from flowdec import fft_utils_tf
from flowdec.fft_utils_tf import OPM_LOG2, OPM_2357, OPM_NONE, OPTIMAL_PAD_MODES, PADF_REFLECT, PAD_FILL_MODES
from flowdec.fft_utils_tf import optimize_dims, ifftshift
from flowdec.tf_ops import pad_around_center, unpad_around_center, tf_observer

SMODE_CONSTANT = 'CONSTANT'
SMODE_INPUT = 'INPUT'
START_MODES = [SMODE_CONSTANT, SMODE_INPUT]

DEFAULT_PAD_MODE = OPM_LOG2
DEFAULT_PAD_FILL = PADF_REFLECT
DEFAULT_START_MODE = SMODE_CONSTANT


class DeconvolutionResult(object):

    def __init__(self, data, info):
        self.data = data
        self.info = info


class DeconvolutionGraph(object):

    def __init__(self, tf_graph, inputs, outputs):
        self.tf_graph = tf_graph
        self.inputs = inputs
        self.outputs = outputs

    def save(self, export_dir, save_as_text=True):
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={k: tf.saved_model.utils.build_tensor_info(v) for k, v in self.inputs.items()},
            outputs={k: tf.saved_model.utils.build_tensor_info(v) for k, v in self.outputs.items()}
        )

        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

        with tf.Session(graph=self.tf_graph) as sess:
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
            )
        builder.save(as_text=save_as_text)
        return self


class Deconvolver(metaclass=abc.ABCMeta):

    def __init__(self, device):
        self.device = device

    def _get_tf_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.device(self.device):
                inputs, outputs = self._build_tf_graph()
        return DeconvolutionGraph(graph, inputs, outputs)

    def initialize(self):
        self.graph = self._get_tf_graph()
        return self

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError()

    def _run(self, acquisition, input_kwargs, session_config=None):

        if not hasattr(self, 'graph'):
            raise ValueError('Must initialize deconvolver before running (via `.initialize` function)')

        if self.device is not None and session_config and not session_config.allow_soft_placement:
            raise AssertionError(
                'When explicitly setting a device, you must also set '
                '"allow_soft_placement" to true in TF session configuration'
            )

        with tf.compat.v1.Session(config=session_config, graph=self.graph.tf_graph) as sess:
            data_dict = {self.graph.inputs[k]: v for k, v in acquisition.to_feed_dict().items()}
            args_dict = {self.graph.inputs[k]: v for k, v in input_kwargs.items() if v is not None}
            res = sess.run(self.graph.outputs, feed_dict={**data_dict, **args_dict})
            return res

    def _run_batch(self, acquisition_batch, **kwargs):
        return [self._run(acq, **kwargs) for acq in acquisition_batch]


def default_input_prep_fn(tensor_name, tensor):
    """ Prepare Kernel/PSF by normalizing sum to one
    Args:
        tensor_name: Name of tensor to apply function to
        tensor: Tensor value
    Returns:
        Normalized PSF tensor
    """
    if tensor_name.startswith('kernel:'):
        return tensor / tf.reduce_sum(tensor)
    return tensor


class FFTDeconvolver(Deconvolver):

    def __init__(self, n_dims, pad_mode, pad_min, pad_fill, start_mode,
        input_prep_fn, output_prep_fn,
        real_domain_fft, device):
        super(FFTDeconvolver, self).__init__(device)
        self.n_dims = n_dims
        self.pad_min = pad_min
        self.pad_mode = pad_mode.upper()
        self.pad_fill = pad_fill.upper()
        self.start_mode = start_mode.upper()
        self.input_prep_fn = input_prep_fn
        self.output_prep_fn = output_prep_fn
        self.real_domain_fft = real_domain_fft
        self.fft_dtype = tf.float32 if real_domain_fft else tf.complex64

        # Validate arguments
        if self.pad_mode not in OPTIMAL_PAD_MODES:
            raise ValueError('Pad mode "{}" invalid; Should be one of {}'.format(self.pad_mode, OPTIMAL_PAD_MODES))
        if self.pad_fill not in PAD_FILL_MODES:
            raise ValueError('Pad fill "{}" invalid; Should be one of {}'.format(self.pad_fill, PAD_FILL_MODES))
        if self.start_mode not in START_MODES:
            raise ValueError('Start mode "{}" invalid; Should be one of {}'.format(self.start_mode, START_MODES))
        if self.pad_min is not None and len(self.pad_min) != self.n_dims:
            raise ValueError('Pad minimum "{}" invalid; Expecting exactly {} values'.format(self.pad_min, self.n_dims))

        # Because TF FFT implementations all only work with 32-bit floats the spatial inputs/outputs in the
        # constructed graph are constrained to this type for now (but it could change in the future)
        self.dtype = tf.float32
        self.device = device

    def _wrap_input(self, tensor):
        return self.input_prep_fn(tensor.name, tensor) if self.input_prep_fn else tensor

    def _wrap_output(self, tensor, inputs=None):
        return self.output_prep_fn(tensor.name, tensor, inputs=inputs) if self.output_prep_fn else tensor


class FFTIterativeDeconvolver(FFTDeconvolver):

    def _get_niter(self):
        return tf.compat.v1.placeholder(tf.int32, shape=(), name='niter')


def richardson_lucy(acquisition, niter=10, pad_mode=DEFAULT_PAD_MODE, session_config=None, **kwargs):
    algo = RichardsonLucyDeconvolver(acquisition.data.ndim, pad_mode=pad_mode, **kwargs)
    return algo.initialize().run(acquisition, niter, session_config=session_config).data


class RichardsonLucyDeconvolver(FFTIterativeDeconvolver):
    """Richardson Lucy Deconvolution Algorithm

    Implementation of iterative Richardson Lucy algorithm on a TensorFlow computational graph.

    Some highlights of this implementation include:
    - Circulant convolution (no linear convolution yet)
    - Automatic, but configurable, padding along dimensions to next highest power of 2 for performance gains (
        experiments show that deconvoling realistic volumes can take up to 4 times longer without this)
    - Configurable domain for FFT operations (either real or complex); Real domain is faster/more memory
        efficient but complex domain is typically more accurate (so it is the default setting)

    Note: Comments throughout are in reference to the following implementations:

    Reference Implementations:
        - Matlab: https://svn.ecdf.ed.ac.uk/repo/ph/IGM/matlab/generic/images/deconvlucy.m
        - Basic Matlab: https://en.wikipedia.org/wiki/Talk:Richardson%E2%80%93Lucy_deconvolution
        - Scikit-Image: https://github.com/scikit-image/scikit-image/blob/master/skimage/restoration/deconvolution.py
        - DeconvolutionLab2: https://github.com/hadim/DeconvolutionLab2/blob/jcufft/src/main/java/deconvolution/
            algorithm/RichardsonLucy.java

    Args:
        n_dims: Rank of tensors to be used as inputs (i.e. number of dimensions); Note that the order of the dimensions and
            their interpretation (e.g. x vs y vs z) is up to the user and any convention can be adopted so long as
            all data and kernel matrices use the same convention
        pad_mode: Padding mode for optimal FFT performance; One of ['log2', '2357' ,'none'] (case-insensitive, default 'log2')
        pad_min: Minimum padding to add to each dimension; Should by array or list of numbers equal
            to extension in each dimension;  For example, "np.array([0, 0, 5])" would do nothing to x and
            y padding but would force padding in z-direction to be at least 5 if using the xyz convention
        pad_fill: Type of fill to use when padding images; One of ['reflect', 'symmetric', 'constant']
            (case-insensitive, default 'reflect'); see https://www.tensorflow.org/api_docs/python/tf/pad for more details
        start_mode: Initial image mode; One of ['constant', 'input'] (case-insensitive, default 'constant') where:
            - constant: Use a constant value of .5 as starting image
            - input: Use image to deconvolve as starting image
        input_prep_fn: Data preparation function to inject within computation graph; Default is PSF
            normalization function used to ensure PSF tensor sums to one; Signature is fn(tensor_name, tensor)
        output_prep_fn: Output preparation function to inject within computation graph (e.g.
            Clipping values in deconvolved results); Signature is fn(tensor_name, tensor, inputs=None) where
            input placeholders may be provided as a way to make transformations of results dependent on
            input data (inputs is a dictionary keyed by tensor input name)
        observer_fn: Function to inject into tensorflow graph causing passage of current image estimation
            and iteration number (useful for setting number of iterations); signature should be:
            `fn(img, i, *args)` where img is the current state of the deconvolution result and i is the
            current 1-based iteration number (*args contains debugging data like the padded, reflected
            version of the image) -- this function should return nothing but if it does, results are ignored
        real_domain_fft: Flag indicating whether or not to use the real or complex TF FFT functions
        epsilon: Minimum value below which interemdiate results will become 0 to avoid division by
            small numbers
        device: TensorFlow format device name onto which the majority of the operations should be
            placed (e.g. '/cpu:0', '/gpu:1'); If providing this, you must also *not* override the
            default setting of "allow_soft_placement=True" in TF session configs
    """
    def __init__(self, n_dims, pad_mode=DEFAULT_PAD_MODE, pad_min=None, pad_fill=DEFAULT_PAD_FILL, start_mode=DEFAULT_START_MODE,
        input_prep_fn=default_input_prep_fn, output_prep_fn=None, observer_fn=None,
        real_domain_fft=False, epsilon=1e-6, device=None):
        super(RichardsonLucyDeconvolver, self).__init__(
            n_dims, pad_mode, pad_min, pad_fill, start_mode, input_prep_fn,
            output_prep_fn, real_domain_fft, device
        )
        self.observer_fn = observer_fn
        self.epsilon = epsilon

    def run(self, acquisition, niter, session_config=None):
        input_kwargs = dict(niter=niter, pad_mode=self.pad_mode, pad_min=self.pad_min, start_mode=self.start_mode)
        res = self._run(acquisition, input_kwargs, session_config=session_config)
        return DeconvolutionResult(res['result'], info={k: v for k, v in res.items() if k != 'result'})

    def _build_tf_graph(self):
        niter = self._get_niter()

        # Create argument placeholders with same defaults as those used at graph construction time
        padmodh = tf.compat.v1.placeholder_with_default(DEFAULT_PAD_MODE, (), name='pad_mode')
        smodeh = tf.compat.v1.placeholder_with_default(DEFAULT_START_MODE, (), name='start_mode')
        padminh = tf.compat.v1.placeholder_with_default(tf.zeros(self.n_dims, dtype=tf.int32), self.n_dims, name='pad_min')

        # Data and kernel should have shapes (z, height, width)
        dataph = tf.compat.v1.placeholder(self.dtype, shape=[None] * self.n_dims, name='data')
        kernph = tf.compat.v1.placeholder(self.dtype, shape=[None] * self.n_dims, name='kernel')
        datah, kernh = self._wrap_input(dataph), self._wrap_input(kernph)

        # Add assertion operations to validate padding mode, start mode, and data/kernel dimensions
        flag_pad_mode = tf.stack([tf.equal(padmodh, OPM_LOG2), tf.equal(padmodh, OPM_2357), tf.equal(padmodh, OPM_NONE)], axis=0)
        assert_pad_mode = tf.compat.v1.assert_greater(
                tf.reduce_sum(tf.cast(flag_pad_mode, tf.int32)), 0,
                message='Pad mode not valid', data=[padmodh])

        flag_start_mode = tf.stack([tf.equal(smodeh, SMODE_CONSTANT), tf.equal(smodeh, SMODE_INPUT)], axis=0)
        assert_start_mode = tf.compat.v1.assert_greater(
                tf.reduce_sum(tf.cast(flag_start_mode, tf.int32)), 0,
                message='Start mode not valid', data=[smodeh])

        flag_shapes = tf.shape(datah) - tf.shape(kernh)
        assert_shapes = tf.compat.v1.assert_greater_equal(
                tf.reduce_sum(flag_shapes), 0,
                message='Data shape must be >= kernel shape', data=[tf.shape(datah), tf.shape(kernh)])

        with tf.control_dependencies([assert_pad_mode, assert_start_mode, assert_shapes]):

            # If configured to do so, expand dimensions of data array to power of 2 or
            # prime factor multiples (after adding a minimum padding as well, if given)
            # to avoid use of Bluestein algorithm in favor of significantly faster Cooley-Tukey FFT
            pad_shape = tf.shape(datah) + padminh
            datat = tf.cond(tf.equal(padmodh, OPM_2357),
                lambda: pad_around_center(datah, optimize_dims(pad_shape, OPM_2357), mode=self.pad_fill),
                lambda: tf.cond(tf.equal(padmodh, OPM_LOG2),
                lambda: pad_around_center(datah, optimize_dims(pad_shape, OPM_LOG2), mode=self.pad_fill),
                lambda: pad_around_center(datah, pad_shape, mode=self.pad_fill)
            ))

            # Pad kernel (with zeros only) to equal dimensions of data tensor and run "circular"
            # transformation as this algorithm is based on circular convolutions and the results
            # will have half spaces swapped otherwise
            kernt = tf.cast(ifftshift(pad_around_center(kernh, tf.shape(datat))), self.fft_dtype)

        # Infer available TF FFT functions based on predefined number of data dimensions
        # TODO: Find a way to determine dimensionality of images separately from batch dimension and
        # update the rank used to get fft fns excluding batch dim
        fft_fwd, fft_rev = fft_utils_tf.get_fft_tf_fns(min(self.n_dims, 3), real_domain_only=self.real_domain_fft)

        # Determine intermediate kernel representation necessary based on domain specified to
        # carry out computations
        kern_fft = fft_fwd(kernt)
        if self.real_domain_fft:
            kern_fft_conj = fft_fwd(tf.reverse(kernt, axis=tf.range(0, self.n_dims)))
        else:
            kern_fft_conj = tf.math.conj(kern_fft)

        # Initialize resulting deconvolved image -- there are several sensible choices for this like the
        # original image or constant arrays, but some experiments show this to be better, and other
        # implementations doing the same are "Basic Matlab" and "Scikit-Image" (see class notes for links)
        decon = tf.cond(
            tf.equal(smodeh, SMODE_CONSTANT),
            lambda: tf.identity(.5 * tf.ones_like(datat, dtype=self.dtype), name='deconvolution'),
            # Multiplication used here to avoid https://github.com/tensorflow/tensorflow/issues/11186
            lambda: tf.identity(datat * tf.ones_like(datat, dtype=self.dtype), name='deconvolution')
        )

        def cond(i, decon):
            return i <= niter

        def conv(inputData, kernel_fft):
            return tf.math.real(fft_rev(fft_fwd(tf.cast(inputData, self.fft_dtype)) * kernel_fft))
			
        def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                    ):
            """Makes 3D gaussian Kernel for convolution."""

            d = tf.distributions.Normal(mean, std)

            vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

            gauss_kernel = tf.einsum('i,j,k->ijk',
                                        vals,
                                        vals,
                                        vals)
            # return the kernel normalised to sum =1
            return gauss_kernel / tf.reduce_sum(gauss_kernel)

        gaussKernel = gaussian_kernel(9, 1.0, 7.0)
        # Expand dimensions of `gauss_kernel` for `tf.nn.conv3d` signature.
        gaussKernel = gaussKernel[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis]

        def body(i, decon,):
            '''# Richardson-Lucy Iteration - logic taken largely from a combination of
            # the scikit-image (real domain) and DeconvolutionLab2 implementations (complex domain)
            # conv1 is the current model blurred with the PSF
            conv1 = conv(decon, kern_fft)

            # High-pass filter to avoid division by very small numbers (see DeconvolutionLab2)
			blur1 = tf.where(conv1 < self.epsilon, tf.zeros_like(datat), datat / conv1, name='blur1')

            # conv2 is the blurred model convolved with the flipped PSF
            conv2 = conv(blur1, kern_fft_conj)

            # Positivity constraint on result for iteration
			decon = tf.maximum(decon * conv2, 0.)
            '''

            # Gold algorithm, ratio method, simpler then RL, doesnt use flipped OTF
            # conv1 is the current model blurred with the PSF
            conv1 = conv(decon, kern_fft)

            # High-pass filter to avoid division by very small numbers (see DeconvolutionLab2)?
            # we wont do it here as we will use the delta parameter in denom and numerrator of division to get blur2
            # as per Stephan Ludwig et al 2019
            # should normalise blur2 and decon each time because numbers get big and we risk overflow when multiplying in next step
            conv1norm = conv1 / (tf.math.reduce_sum(conv1))
            datatNorm = datat / (tf.math.reduce_sum(datat))
            # this value seems to work well fo rthe images that are normalised to sum of 1
            deltaParam = 1e-4
            ratio = (datatNorm + deltaParam) / (conv1norm + deltaParam)
            #blur1 = tf.where(conv1 < self.epsilon, tf.zeros_like(datat), datat / conv1, name='blur1')
            #ratioNorm = ratio / (tf.math.reduce_sum(ratio))
            #deconNorm = decon / (tf.math.reduce_sum(decon))
            # decon is the  normalised blurred model multiplied by the model
            # Positivity constraint on result for iteration
            decon = tf.maximum(decon * ratio, 0.)
			# Smooth the intermediate result image with Gaussian of sigma 1 every 5th iteration
            # to control noise buildup that Gold method is succeptible to.
            # Use tf.nn.conv3d to convolve a Gaussian kernel with an image:
            # Make Gaussian Kernel with desired specs using gaussian_kernel function defined above
            if i % 5 == 0:
                # Convolve decon with gauss kernel.
                tf.nn.conv3d(decon, filter=gaussKernel, strides=[1, 1, 1, 1, 1], padding="SAME")
            # normalise the result so the sum of the data is 1
            decon = decon / (tf.math.reduce_sum(decon))
            
            # TODO - Smoothing every 5 iterations with gaussian or wiener. 
            # TODO rescale back to input data sum intensity -  probably need to adjust deltaParam too. 

            # If given an "observer", pass the current image restoration and iteration counter to it
            if self.observer_fn is not None:
                # Remove any cropping that may have been added as this is usually not desirable in observers
                decon_crop = unpad_around_center(decon, tf.shape(datah))
                # normalise the result so the sum of the data is 1
                decon_crop = decon_crop / (tf.math.reduce_sum(decon_crop))
				# we can use these captured observed tensors to evaluate eg convergence
                # in eg. the observer function used.
                _, i, decon, conv1  = tf_observer(
				    [decon_crop, i, decon, conv1], self.observer_fn)

            return i + 1, decon

        result = tf.while_loop(cond, body, [1, decon], parallel_iterations=1)[1]

        # Crop off any padding that may have been added to reach more efficient dimension sizes
        result = unpad_around_center(result, tf.shape(datah))

        # Wrap output in configured post-processing functions (if any)
        result = tf.identity(self._wrap_output(result, {'data': datah, 'kernel': kernh}), name='result')

        inputs = {
            'niter': niter, 'data': dataph, 'kernel': kernph,
            'pad_mode': padmodh, 'pad_min': padminh, 'start_mode': smodeh
        }
        outputs = {
            'result': result,
            'data_shape': tf.shape(datah), 'kern_shape': tf.shape(kernh),
            'pad_shape': pad_shape, 'pad_mode': padmodh,
            'datat_shape': tf.shape(datat),
            'pad_min': padminh, 'start_mode': smodeh,
        }

        return inputs, outputs
