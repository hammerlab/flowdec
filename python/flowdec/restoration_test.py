import unittest

import os
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from flowdec import validation as tfv
from numpy.testing import assert_array_equal
import numpy as np


def dl2_enabled():
    return os.getenv('FLOWDEC_UT_DL2', 'false').lower() == 'true'


class TestRestoration(unittest.TestCase):

    def _validate_restoration(self, acqs, n_iter, thresholds):

        # Assume DL2 use only if threshold provided for it
        use_dl2 = 'dl2' in thresholds and thresholds['dl2']

        # Run all deconvolution implementations on each of the given image acquisitions
        res = [tfv.run_deconvolutions(a, n_iter=n_iter, dl2=use_dl2) for a in acqs]
        scores = [r['scores'] for r in res]

        # TODO: Move this to external documentation somewhere
        # In the event of a test failure, this is useful debugging information especially when analyzed via the
        # python/examples/Deconvolution - Unit Test Debugging.ipynb notebook:
        # import pickle
        # idx = 1
        # with open('/tmp/flowdec-ut-debug-data.pkl', 'wb') as fd:
        #     pickle.dump(res[idx], fd)
        # print(scores)
        # import pandas as pd
        # print(pd.DataFrame(scores))

        tf_key = 'tf'
        for k in res[0]['scores'].keys():
            if k == tf_key:
                continue
            thresh = thresholds.get(k, 1.)
            flags = np.array([s[tf_key] > s[k] for s in scores])
            self.assertTrue(np.mean(flags) >= thresh,
                msg='TF scores not >= {}% of {} scores (pct better = {}) (all scores = {})'
                        .format(thresh * 100, k, 100 * np.mean(flags), '\n'.join([str(m) for m in scores])))

    def test_bars(self):
        """Test reconstruction of blurred "Hollow Bars" volume"""
        n_iter = 10
        thresholds = {
            # Results should always improve on the original
            'original': 1.,

            # Results should also be better than DL2 and scikit-image most of the time
            'sk': .75,
            'dl2': .75 if dl2_enabled() else None
        }

        acq = fd_data.bars_25pct()

        # Initialize list of acquisitions to deconvolve
        acqs = [acq]

        # Add translations to images and kernels to ensure that there aren't
        # any issues supporting non-symmetric inputs
        acqs += [
            tfv.reblur(tfv.shift(acq, data_shift=(0, 10, 10))),
            tfv.reblur(tfv.shift(acq, data_shift=(0, -10, -10))),

            tfv.reblur(tfv.shift(acq, kern_shift=(0, 10, 10))),
            tfv.reblur(tfv.shift(acq, kern_shift=(0, -10, -10))),

            tfv.reblur(tfv.shift(acq, data_shift=(-3, 5, -5), kern_shift=(-3, 5, -5)))
        ]

        # Subset image and kernel to make sure that padding of inputs for fft is added
        # and then cropped out correctly (tests asymmetries and non-power-of-2 dimensions)
        acqs += [
            tfv.reblur(tfv.subset(acq, kern_slice=[slice(None, 24), slice(None, 48), slice(None, 48)])),
        ]

        # Validate that downsampling the original volume also causes no issues
        acqs += [
            tfv.reblur(tfv.downsample(acq, data_factor=.8, kern_factor=.4)),
            tfv.reblur(tfv.downsample(acq, data_factor=.5, kern_factor=.5))
        ]

        self._validate_restoration(acqs, n_iter, thresholds=thresholds)

    def test_basic_shape_2d(self):
        """Validate recovery of simple 2 dimensional shape"""

        def square(n, start, end):
            x = np.zeros((n, n))
            x[start:end, start:end] = 1
            return x

        def cross(n, start, end, mid):
            x = np.zeros((n, n))
            x[start:end, mid] = 1
            x[mid, start:end] = 1
            return x

        shapes = [
            # 3-square, filled in center
            (square(3, 1, 2), square(3, 1, 2)),

            # data = 3-square within 9-square, kernel = dot within 9-square
            (square(9, 3, 6), square(9, 4, 5)),

            # data = 3-square within 7-square, kernel = dot within 7-square
            (square(7, 2, 5), square(7, 3, 4)),

            # data = 3-square within 9-square, kernel = 3-square within 7-square
            (square(9, 3, 6), square(7, 2, 5)),

            # data = 3-square within 8-square, kernel = 3-square within 7-square
            (square(8, 3, 5), square(7, 2, 5)),

            # data = 3-square within 8-square, kernel = 3-square within 7-square
            (square(8, 3, 5), square(7, 2, 5)),

            # data = cross within 9-square, kernel = dot within 9-square
            (cross(9, 3, 6, 5), square(9, 4, 5))
        ]

        for shape in shapes:
            x, k = shape

            # Apply blur to original shape and run restoration on blurred image
            bin_res, bin_tru = self._decon_shape(x, k)

            assert_array_equal(
                    bin_tru, bin_res, '2D restoration not equal to original\nOriginal: {}\nResult: {}'
                        .format(bin_tru, bin_res))

    def test_basic_shape_3d(self):
        """Validate recovery of simple 3 dimensional shape"""

        def square(n, start, end):
            x = np.zeros((n, n, n))
            x[start:end, start:end, start:end] = 1
            return x

        shapes = [
            # data = 3-cube within 5-cube, kernel = 3-cube within 5-cube
            (square(5, 2, 3), square(5, 2, 3)),

            # data = 3-cube within 7-cube, kernel = dot within 5-cube
            (square(7, 2, 5), square(5, 2, 3))
        ]

        for shape in shapes:
            x, k = shape

            # Apply blur to original shape and run restoration on blurred image
            bin_res, bin_tru = self._decon_shape(x, k)

            assert_array_equal(
                    bin_tru, bin_res, '3D restoration not equal to original\nOriginal: {}\nResult: {}'
                        .format(bin_tru, bin_res))

    def test_observer(self):
        acq = fd_data.bars_25pct()
        imgs = []

        def observer(img, *_):
            imgs.append(img)
            self.assertEqual(acq.data.shape, img.shape, msg='Observer image and original shapes not equal')
        algo = fd_restoration.RichardsonLucyDeconvolver(n_dims=3, observer_fn=observer).initialize()
        algo.run(acq, niter=5)
        self.assertEqual(len(imgs), 5)

    def _decon_shape(self, data, kernel):
        # Apply blur to original shape and run restoration on blurred image
        acq = fd_data.Acquisition(data=data, kernel=kernel, actual=data)
        acq = tfv.reblur(acq, scale=.001) # Add low amount of noise
        res = tfv.decon_tf(acq, 10, real_domain_fft=False)

        # Binarize resulting image and validate that pixels match original exactly
        bin_res = (res > res.mean()).astype(np.int64)
        bin_tru = acq.actual.astype(np.int64)
        return bin_res, bin_tru