import unittest
import numpy as np
import tensorflow as tf
from flowdec import fft_utils_np, fft_utils_tf, test_utils
from numpy.testing import assert_array_equal, assert_almost_equal


class TestFFTUtils(unittest.TestCase):

    def _test_padding(self, d, k, actual):
        def tf_fn():
            dt = tf.constant(d, dtype=tf.float32)
            kt = tf.constant(k, dtype=tf.float32)
            return fft_utils_tf.get_fft_pad_dims(dt, kt)

        tf_res = test_utils.exec_tf(tf_fn)
        np_res = fft_utils_np.get_fft_pad_dims(d, k)

        self.assertTrue(type(tf_res) is np.ndarray)
        self.assertTrue(type(np_res) is np.ndarray)
        self.assertTrue(np.array_equal(tf_res, np_res))
        self.assertTrue(np.array_equal(tf_res, actual))

    def test_padding(self):
        """Verify padding operations implemented as TensorFlow ops"""
        self._test_padding(np.ones(1), np.ones(1), np.array([1]))
        self._test_padding(np.ones((10)), np.ones((5)), np.array([14]))
        self._test_padding(np.ones((10, 5)), np.ones((5, 3)), np.array([14, 7]))
        self._test_padding(np.ones((10, 5, 3)), np.ones((5, 3, 1)), np.array([14, 7, 3]))

    def _test_optimize_padding(self, d, k, mode, actual):
        def tf_fn():
            dt = tf.constant(d, dtype=tf.float32)
            kt = tf.constant(k, dtype=tf.float32)
            pad_dims = fft_utils_tf.get_fft_pad_dims(dt, kt)
            return fft_utils_tf.optimize_dims(pad_dims, mode)

        tf_res = test_utils.exec_tf(tf_fn)
        np_res = fft_utils_np.optimize_dims(fft_utils_np.get_fft_pad_dims(d, k), mode)

        self.assertTrue(type(tf_res) is np.ndarray)
        self.assertTrue(type(np_res) is np.ndarray)
        assert_array_equal(tf_res, np_res)
        assert_array_equal(tf_res, actual)

    def test_optimize_padding(self):
        """Verify "round-up" of dimensions to those optimal for FFT"""
        self._test_optimize_padding(np.ones((1)), np.ones((1)), 'log2', np.array([1]))

        self._test_optimize_padding(np.ones((10, 5)), np.ones((6, 3)), 'log2', np.array([16, 8]))
        self._test_optimize_padding(np.ones((10, 5)), np.ones((7, 4)), 'log2', np.array([16, 8]))
        self._test_optimize_padding(np.ones((10, 5)), np.ones((8, 5)), 'log2', np.array([32, 16]))

        self._test_optimize_padding(np.ones((10, 5)), np.ones((6, 3)), 'none', np.array([15, 7]))
        self._test_optimize_padding(np.ones((10, 5)), np.ones((7, 4)), 'none', np.array([16, 8]))
        self._test_optimize_padding(np.ones((10, 5)), np.ones((8, 5)), 'none', np.array([17, 9]))

        self._test_optimize_padding(np.ones((10, 5, 3)), np.ones((8, 5, 1)), 'log2', np.array([32, 16, 4]))

        # Test invalid padding mode
        with self.assertRaises(ValueError):
            self._test_optimize_padding(np.ones((1)), np.ones((1)), 'invalid_mode_name', np.array([1]))

    def _test_shift(self, x, tf_shift_fn, np_shift_fn):
        def tf_fn():
            return tf_shift_fn(tf.constant(x))
        x_shift_actual = test_utils.exec_tf(tf_fn)
        x_shift_expect = np_shift_fn(x)
        assert_array_equal(x_shift_actual, x_shift_expect)

    def _test_all_shifts(self, tf_shift_fn, np_shift_fn):
        # 1D Cases
        self._test_shift(np.arange(99), tf_shift_fn, np_shift_fn)
        self._test_shift(np.arange(100), tf_shift_fn, np_shift_fn)

        # 2D Cases
        x = np.reshape(np.arange(50), (25, 2))
        self._test_shift(x, tf_shift_fn, np_shift_fn)

        # 3D Cases
        self._test_shift(np.reshape(np.arange(125), (5, 5, 5)), tf_shift_fn, np_shift_fn)
        self._test_shift(np.reshape(np.arange(60), (3, 4, 5)), tf_shift_fn, np_shift_fn)

    def test_fftshift(self):
        self._test_all_shifts(fft_utils_tf.fftshift, np.fft.fftshift)

    def test_ifftshift(self):
        self._test_all_shifts(fft_utils_tf.ifftshift, np.fft.ifftshift)

    def _test_convolution(self, d, k, mode, actual=None):
        def tf_fn():
            dt = tf.constant(d, dtype=tf.float32)
            kt = tf.constant(k, dtype=tf.float32)

            # Determine FFT dimensions and functions
            pad_dims = fft_utils_tf.get_fft_pad_dims(dt, kt)
            optim_dims = fft_utils_tf.optimize_dims(pad_dims, mode)
            fft_fwd, fft_rev = fft_utils_tf.get_fft_tf_fns(dt.shape.ndims)

            # Run convolution of data 'd' with kernel 'k'
            dk_fft = fft_fwd(kt, fft_length=optim_dims)
            dconv = fft_utils_tf.convolve(dt, dk_fft, optim_dims, fft_fwd, fft_rev)

            # Extract patch from result matching dimensions of original data array
            return fft_utils_tf.extract(dconv, tf.shape(dt), pad_dims)

        tf_res = test_utils.exec_tf(tf_fn)
        np_res = fft_utils_np.convolve(d, k)

        assert_almost_equal(tf_res, np_res, decimal=3)
        self.assertEquals(tf_res.shape, np_res.shape)
        if actual is not None:
            assert_array_equal(tf_res, actual)

    def test_convolution(self):

        #######################
        # Verified Test Cases #
        #######################
        # * Validate that Numpy == TensorFlow == Manually Defined Expectation

        for mode in fft_utils_tf.OPTIMAL_PAD_MODES:

            # 1D Cases
            actual = [1.]
            self._test_convolution(np.ones((1)), np.ones((1)), mode, actual)

            actual = [1., 2., 2.]
            self._test_convolution(np.ones((3)), np.ones((2)), mode, actual)

            # 2D Case
            # FFT convolution should result in "lower-right" side
            # sums of products of data with kernel values
            # See [here](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.472.2396&rep=rep1&type=pdf)
            # for some decent visual explanations of this
            actual = np.array([
                [1., 2., 2.],
                [2., 4., 4.],
                [2., 4., 4.]
            ])
            self._test_convolution(np.ones((3, 3)), np.ones((2, 2)), mode, actual)

            ###########################
            # Corroborated Test Cases #
            ###########################
            # * Validate that Numpy == TensorFlow results only

            # Test 1-3D cases with larger, rectangular dimensions and unit length axes
            for shape in [
                ((1), (1)),
                ((1, 1), (1, 1)),
                ((100, 100), (10, 10)),
                ((100, 5), (10, 15)),
                ((7, 9), (19, 3)),
                ((3, 1), (1, 3)),
                ((2, 1, 2), (2, 1, 2)),
                ((1, 1, 1), (1, 1, 1))
            ]:
                self._test_convolution(np.ones(shape[0]), np.ones(shape[1]), mode)


        ###############
        # Error Cases #
        ###############
        # * Validate conditions resulting in errors

        mode = fft_utils_tf.OPTIMAL_PAD_MODES[0]

        with self.assertRaises(ValueError):
            # >= 4D should fail
            self._test_convolution(np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1)), mode)

        with self.assertRaises(ValueError):
            # Dimension mismatch for data and kernel should fail
            self._test_convolution(np.ones((1, 1)), np.ones((1, 1, 1)), mode)

        # The empty array case below is an interesting example that actually leads to python
        # crashing "with exit code 136" (SIGFPE --> arithmetic error) with no preemptive
        # checks on empty to arrays so ensure these are handled as explicit exceptions.
        # Adding assertions to the TF graph doesn't fix the problem either but it at least
        # gives a message like "Check failed: size >= 0 (-9223372036854775808 vs. 0)" before
        # the python process again crashes.  As a result, this check will not be run here
        # but it is important that similar checks are run on higher level parts of the API
        # (i.e. pre-TensorFlow) to keep this from coming up.
        # self._test_convolution(np.ones((0)), np.ones((0)), mode, [1.])


if __name__ == '__main__':
    unittest.main()