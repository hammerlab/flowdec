import unittest
import os
from flowdec import psf, data
from skimage import io
from numpy.testing import assert_almost_equal, assert_array_equal


class TestPSF(unittest.TestCase):

    def test_psf_comparison(self):
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

    def test_psf_config_io(self):
        """Validate that PSF can be saved to and restored from configuration files"""
        import tempfile

        # Generate a PSF object with at least some non-default args
        k1 = psf.GibsonLanni(
            size_x=16, size_y=16, size_z=8, pz=0., wavelength=.425,
            na=.75, res_lateral=.377, res_axial=1.5
        )

        # Create kernel from psf and save as configuration file
        d_before = k1.generate()
        f = tempfile.mktemp(suffix='.json', prefix='psf-config')
        k1.save(f)

        # Restore from configuration file and generate kernel again
        k2 = psf.GibsonLanni.load(f)
        d_after = k2.generate()

        # Check that both configuration and generated kernels are equal
        self.assertEqual(k1.config, k2.config)
        assert_array_equal(d_before, d_after)

        # Delete config file
        os.unlink(f)


