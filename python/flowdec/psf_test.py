import unittest
import os
from flowdec import psf, data
from skimage import io
from numpy.testing import assert_almost_equal


class TestPSF(unittest.TestCase):

    def test_psf_1(self):
        """Validate a generated kernel against the same thing from PSFGenerator"""

        k_actual = psf.GibsonLanni(
            size_x=16, size_y=16, size_z=8, pz=0., wavelength=.610,
            na=1.4, res_lateral=.1, res_axial=.25
        ).generate()

        # The following image used for comparison was generated via PSFGenerator
        # with these arguments (where any not mentioned were left at default):
        # Optical Model: "Gibson & Lanni 3D Optical Model"
        # NA: 1.4
        # Particle position Z: 0
        # Wavelength: 610 (nm)
        # Pixelsize XY: 100 (nm)
        # Z-step: 250 (nm)
        # Size XYZ: 16, 16, 8
        path = data._get_dataset_path(os.path.join('psfs', 'psfgen1.tif'))
        k_expect = io.imread(path)

        assert_almost_equal(k_actual, k_expect, decimal=4)