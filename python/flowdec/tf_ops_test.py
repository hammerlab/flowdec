import unittest
import numpy as np
import tensorflow as tf
from flowdec import tf_ops, test_utils
from numpy.testing import assert_array_equal


class TestTFOps(unittest.TestCase):

    def _test_padding(self, x, **kwargs):
        def tf_fn():
            xt = tf.constant(x)
            yt = tf_ops.pad_around_center(xt, **kwargs)
            zt = tf_ops.unpad_around_center(yt, tf.shape(xt))
            return zt

        x_actual = test_utils.exec_tf(tf_fn)
        assert_array_equal(x, x_actual, 'Padded vs un-padded array not equal')

    def test_padding(self):

        # ######## #
        # 1D Cases #
        # ######## #
        x = np.arange(100)

        # Test noop padding, a unit increase, and a large increase
        self._test_padding(x, target_shape=(100))
        self._test_padding(x, target_shape=(101))
        self._test_padding(x, target_shape=(256))


        # ######## #
        # 2D Cases #
        # ######## #
        x = np.reshape(np.arange(100), (10, 10))

        # Test noop padding and unit increases
        self._test_padding(x, target_shape=(10, 10))
        self._test_padding(x, target_shape=(11, 10))
        self._test_padding(x, target_shape=(10, 11))
        self._test_padding(x, target_shape=(11, 11))

        # Test larger increase in both dimensions
        self._test_padding(x, target_shape=(100, 256))

        # ######## #
        # 3D Cases #
        # ######## #
        x = np.reshape(np.arange(27), (3, 3, 3))

        # Test noop padding and unit increases
        self._test_padding(x, target_shape=(3, 3, 3))
        self._test_padding(x, target_shape=(3, 4, 3))
        self._test_padding(x, target_shape=(4, 4, 4))

        # Test larger increase in all dimensions
        self._test_padding(x, target_shape=(30, 31, 32))


if __name__ == '__main__':
    unittest.main()
