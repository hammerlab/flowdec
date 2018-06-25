"""PSF Generator Module based on Fast Gibson Lanni Approximation

This is the exact same implementation (used with permission from
http://kmdouglass.github.io/posts/implementing-a-fast-gibson-lanni-psf-solver-in-python.html)
"""
import argparse
import json
from collections import OrderedDict


# ##################
# Gibson Lanni PSF #
# ##################

# Defaults set in python implementation
# See: http://kmdouglass.github.io/posts/implementing-a-fast-gibson-lanni-psf-solver-in-python.html
# # Image properties
# # Size of the PSF array, pixels
# size_x = 256
# size_y = 256
# size_z = 128
#
# # Precision control
# num_basis    = 100  # Number of rescaled Bessels that approximate the phase function
# num_samples  = 1000 # Number of pupil samples along radial direction
# oversampling = 2    # Defines the upsampling ratio on the image space grid for computations
#
# # Microscope parameters
# NA          = 1.4
# wavelength  = 0.610 # microns
# M           = 100   # magnification
# ns          = 1.33  # specimen refractive index (RI)
# ng0         = 1.5   # coverslip RI design value
# ng          = 1.5   # coverslip RI experimental value
# ni0         = 1.5   # immersion medium RI design value
# ni          = 1.5   # immersion medium RI experimental value
# ti0         = 150   # microns, working distance (immersion medium thickness) design value
# tg0         = 170   # microns, coverslip thickness design value
# tg          = 170   # microns, coverslip thickness experimental value
# res_lateral = 0.1   # microns
# res_axial   = 0.25  # microns
# pZ          = 2     # microns, particle distance from coverslip
#
# # Scaling factors for the Fourier-Bessel series expansion
# min_wavelength = 0.436 # microns
# scaling_factor = NA * (3 * np.arange(1, num_basis + 1) - 2) * min_wavelength / wavelength

# Defaults set in docs within original Matlab implementation (MicroscPSF.m)
# See: http://www.ee.cuhk.edu.hk/~tblu/monsite/phps/demos.php
# %   (1) image properties
# %           'size'  -  the size of the 3D PSF, e.g. params.size = [256 256 128];
# %   (2) precision control
# %           'numBasis'      - the number of approximation basis, default '100'
# %           'numSamp'       - the number of sampling to determine the basis
# %                             coefficients, default '1000'
# %           'overSampling'  - the oversampling ratio, default 2
# %   (3) microscope parameters
# %        'NA'        - numerical aperture of the microscope, default 1.4
# %        'lambda'    - Emission wavelength in vacuum, default 610nm
# %        'M'         - magnification factor, default 100
# %        'ns'        - specimen refractive index (RI), default 1.33
# %        'ng0'       - coverslip RI, design value, default 1.5
# %        'ng'        - coverslip RI, experimental, default 1.5
# %        'ni0'       - immersion RI, design value, default 1.5
# %        'ni'        - immersion RI, experimental, defualt 1.5
# %        'ti0'       - working distance, design, default 150um
# %        'tg0'       - coverslip thickness, design value, default 170um
# %        'tg'        - coverslip thickness, experimental, default 170um
# %        'resLateral' - lateral pixel size, default 100nm
# %        'resAxial'  - axial pixel size, default 250nm
# %        'pZ'        - position of particle, default 2000nm


GL_PSF_ARGS = [
    ['size_x',      256,  "Number of pixels in result (x dimension)", "Dimensions"],
    ['size_y',      256,  "Number of pixels in result (y dimension)", "Dimensions"],
    ['size_z',      128,  "Number of pixels in result (z dimension)", "Dimensions"],
    ['na',          1.4,  "Numerical aperture of device", "Microscope Parameters"],
    ['wavelength',  .610, "Wavelength of emitted light in vacuum (microns)", "Microscope Parameters"],
    ['m',           100,  "Magnification factor", "Microscope Parameters"],
    ['ns',          1.33, "Specimen refractive index (RI)", "Microscope Parameters"],
    ['ng0',         1.5,  "Coverslip RI, design value", "Microscope Parameters"],
    ['ng',          None, "Coverslip RI, experimental (defaults to ng0 if not given)", "Microscope Parameters"],
    ['ni0',         1.5,  "Immersion RI, design value", "Microscope Parameters"],
    ['ni',          None, "Immersion RI, experimental (defaults to ni0 if not given)", "Microscope Parameters"],
    ['ti0',         150,  "Working distance (microns)", "Microscope Parameters"],
    ['tg0',         170,  "Coverslip thickness, design value (microns)", "Microscope Parameters"],
    ['tg',          None, "Coverslip thickness, experimental (microns) (defaults to tg0 if not given)", "Microscope Parameters"],
    ['res_lateral', 0.1,  "Lateral pizel size / resolution (microns)", "Microscope Parameters"],
    ['res_axial',   0.25, "Axial pizel size / resolution (microns)", "Microscope Parameters"],
    ['pz',          0,    "Particle distance from coverslip (microns)", "Microscope Parameters"],
    ['num_basis',   100,  "Number of rescaled Bessels that approximate the phase function", "Precision Parameters"],
    ['num_samples', 1000, "Number of pupil samples along radial direction", "Precision Parameters"],
    ['oversampling',2,    "Defines the upsampling ratio on the image space grid for computations",
     "Precision Parameters"],
    ['min_wavelength', 0.436,"Reference wavelength used in computation of scaling factor (microns); "
                             "See section titled 'B. Bessel series approximation' in [1] for more details",
                             "Precision Parameters"],
]


class PSF(object):
    pass


class GibsonLanni(PSF):

    def __init__(self, **kwargs):
        """ Python implementation of fast Gibson-Lanni PSF approximation model

        This is based on [1] and was originally developed in Matlab before being
        ported to a Python implementation [2].  This implementation is used verbatim here
        with permission from the author Kyle Douglass.

        References:
        [1] - Li, J., Xue, F., & Blu, T. (2017). Fast and accurate three-dimensional point spread function
            computation for fluorescence microscopy. JOSA A, 34(6), 1029-1034.
        [2] - http://kmdouglass.github.io/posts/implementing-a-fast-gibson-lanni-psf-solver-in-python.html

        Args:
            See psf.GL_PSF_ARGS for a description of arguments applicable for this class.  All of them
            have default values that can be overridden by keyword arguments here matching the argument name.
            Additionally, GibsonLanni.get_arg_parser().print_help() will display arguments, descriptions,
            and defaults in a readable form.
        """
        args = GL_PSF_ARGS

        # Check to see if any arguments were given with invalid names
        known_args = [a[0] for a in args]
        bad_args = set(kwargs.keys()) - set(known_args)
        if len(bad_args) > 0:
            raise ValueError(
                'The following arguments given are not valid: {}\nValid argument names: {}'
                .format(bad_args, known_args)
            )

        # Assign configuration by resolving default arguments and those passed in
        self.config = OrderedDict({a[0]: a[1] for a in args})
        self.config.update(kwargs)

    def to_json(self):
        return json.dumps(self.config)

    def save(self, path):
        """Save PSF configuration as json in the given file"""
        with open(path, 'w') as fd:
            json.dump(self.config, fd, indent=4, sort_keys=True)
        return self

    @staticmethod
    def load(path):
        """Load a PSF object from a json configuration file"""
        with open(path, 'r') as fd:
            return GibsonLanni(**json.load(fd))

    @staticmethod
    def get_arg_parser():
        """ Get PSF argument parser and field descriptions """
        parser = argparse.ArgumentParser(GibsonLanni.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        groups = OrderedDict({})
        for arg in GL_PSF_ARGS:
            group = arg[3]
            if not group in groups:
                groups[group] = parser.add_argument_group(group)
            groups[group].add_argument(
                '--{}'.format(arg[0].replace('_', '-')),
                default=arg[1],
                help=arg[2]
            )

        return parser

    def generate(self):
        import numpy as np
        import scipy.special
        from scipy.interpolate import interp1d

        # ################# #
        # Define Parameters #
        # ################# #

        size_x = self.config['size_x']
        size_y = self.config['size_y']
        size_z = self.config['size_z']
        NA = self.config['na']
        wavelength = self.config['wavelength']
        M = self.config['m']
        ns = self.config['ns']
        ng0 = self.config['ng0']
        ng = self.config['ng'] or ng0
        ni0 = self.config['ni0']
        ni = self.config['ni'] or ni0
        ti0 = self.config['ti0']
        tg0 = self.config['tg0']
        tg = self.config['tg'] or tg0
        res_lateral = self.config['res_lateral']
        res_axial = self.config['res_axial']
        pZ = self.config['pz']
        num_basis = self.config['num_basis']
        num_samples = self.config['num_samples']
        oversampling = self.config['oversampling']
        min_wavelength = self.config['min_wavelength']

        scaling_factor = NA * (3 * np.arange(1, num_basis + 1) - 2) * min_wavelength / wavelength

        # ############################# #
        # Create the coordinate systems #
        # ############################# #

        # Place the origin at the center of the final PSF array
        x0 = (size_x - 1) / 2
        y0 = (size_y - 1) / 2

        # Find the maximum possible radius coordinate of the PSF array by finding the distance
        # from the center of the array to a corner
        max_radius = round(np.sqrt((size_x - x0) * (size_x - x0) + (size_y - y0) * (size_y - y0))) + 1;

        # Radial coordinates, image space
        r = res_lateral * np.arange(0, oversampling * max_radius) / oversampling

        # Radial coordinates, pupil space
        a = min([NA, ns, ni, ni0, ng, ng0]) / NA
        rho = np.linspace(0, a, num_samples)

        # Stage displacements away from best focus
        z = res_axial * np.arange(-size_z / 2, size_z /2) + res_axial / 2

        # ######################################################## #
        # Approximate the pupil phase with a Fourier-Bessel series #
        # ######################################################## #

        # Define the wavefront aberration
        OPDs = pZ * np.sqrt(ns * ns - NA * NA * rho * rho) # OPD in the sample
        OPDi = (z.reshape(-1,1) + ti0) * np.sqrt(ni * ni - NA * NA * rho * rho) - ti0 * np.sqrt(ni0 * ni0 - NA * NA * rho * rho) # OPD in the immersion medium
        OPDg = tg * np.sqrt(ng * ng - NA * NA * rho * rho) - tg0 * np.sqrt(ng0 * ng0 - NA * NA * rho * rho) # OPD in the coverslip
        W    = 2 * np.pi / wavelength * (OPDs + OPDi + OPDg)

        # Sample the phase
        # Shape is (number of z samples by number of rho samples)
        phase = np.cos(W) + 1j * np.sin(W)

        # Define the basis of Bessel functions
        # Shape is (number of basis functions by number of rho samples)
        J = scipy.special.jv(0, scaling_factor.reshape(-1, 1) * rho)

        # Compute the approximation to the sampled pupil phase by finding the least squares
        # solution to the complex coefficients of the Fourier-Bessel expansion.
        # Shape of C is (number of basis functions by number of z samples).
        # Note the matrix transposes to get the dimensions correct.
        C, residuals, _, _ = np.linalg.lstsq(J.T, phase.T, rcond=None)

        # ############### #
        # Compute the PSF #
        # ############### #
        b = 2 * np. pi * r.reshape(-1, 1) * NA / wavelength

        # Convenience functions for J0 and J1 Bessel functions
        J0 = lambda x: scipy.special.jv(0, x)
        J1 = lambda x: scipy.special.jv(1, x)

        # See equation 5 in Li, Xue, and Blu
        denom = scaling_factor * scaling_factor - b * b
        R = (scaling_factor * J1(scaling_factor * a) * J0(b * a) * a - b * J0(scaling_factor * a) * J1(b * a) * a)
        R /= denom

        # The transpose places the axial direction along the first dimension of the array, i.e. rows
        # This is only for convenience.
        PSF_rz = (np.abs(R.dot(C))**2).T

        # Normalize to the maximum value
        PSF_rz /= np.max(PSF_rz)

        # ############################################################# #
        # Resample the PSF onto a rotationally-symmetric Cartesian grid #
        # ############################################################# #

        # Create the fleshed-out xy grid of radial distances from the center
        xy      = np.mgrid[0:size_y, 0:size_x]
        r_pixel = np.sqrt((xy[1] - x0) * (xy[1] - x0) + (xy[0] - y0) * (xy[0] - y0)) * res_lateral

        PSF = np.zeros((size_y, size_x, size_z))

        for z_index in range(PSF.shape[2]):
            # Interpolate the radial PSF function
            PSF_interp = interp1d(r, PSF_rz[z_index, :])

            # Evaluate the PSF at each value of r_pixel
            PSF[:,:, z_index] = PSF_interp(r_pixel.ravel()).reshape(size_y, size_x)

        # **All lines below are changes to original implementation** #

        # Transform to [z, y, x] instead of [y, x, z]
        PSF = np.moveaxis(PSF, 2, 0)

        # Re-normalize to a max of 1
        return PSF / np.max(PSF)


