"""Deconvolution CLI

This script can be used to deconvolve single data files with either a predefined or dynamically generated PSF.

Usage:

usage: deconvolution.py [-h] --data-path DATA_PATH --output-path OUTPUT_PATH
                        [--psf-path PSF_PATH]
                        [--psf-config-path PSF_CONFIG_PATH]
                        [--n-iter N_ITERATIONS] [--log-level LOG_LEVEL]

optional arguments:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        Path to image file containing data to deconvolve (any
                        format compatible with skimage.io.imread)
  --output-path OUTPUT_PATH
                        Path in which deconvolution result will be saved (any
                        format compatible with skimage.io.imsave)
  --psf-path PSF_PATH   Path to image file containing point spread function
                        (any format compatible with skimage.io.imread); One of
                        either this or --psf-config-path must be supplied
  --psf-config-path PSF_CONFIG_PATH
                        Path to file containing PSF configuration (see
                        flowdec.psf.from_file for details); One of either this
                        or --psf-path must be supplied
  --n-iter N_ITERATIONS
                        Number of deconvolution iterations to run (default is
                        25)
  --log-level LOG_LEVEL
                        Logging level name (default is 'INFO')

Examples:

# Invocation with pre-defined PSF
deconvolution \
--data-path=flowdec/datasets/bars-25pct/data.tif \
--psf-path=flowdec/datasets/bars-25pct/kernel.tif \
--output-path=/tmp/result.tif \
--n-iter=25 \
--log-level=DEBUG

# Invocation with dynamic PSF
echo '{"na": 0.75, "wavelength": 0.425, "size_z": 32, "size_x": 64, "size_y": 64}' > /tmp/psf.json
deconvolution \
--data-path=flowdec/datasets/bars-25pct/data.tif \
--psf-config-path=/tmp/psf.json \
--output-path=/tmp/result.tif \
--n-iter=25 \
--log-level=DEBUG
"""

from timeit import default_timer as timer
import argparse
from skimage import io
from flowdec import restoration as fd_restoration
from flowdec import data as fd_data
from flowdec import psf as fd_psf
import logging


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description="Deconvolution CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

# Invocation with pre-defined PSF
deconvolution \\
--data-path=flowdec/datasets/bars-25pct/data.tif \\
--psf-path=flowdec/datasets/bars-25pct/kernel.tif \\
--output-path=/tmp/result.tif \\
--n-iter=25 \\
--log-level=DEBUG

# Invocation with dynamic PSF
echo '{"na": 0.75, "wavelength": 0.425, "size_z": 32, "size_x": 64, "size_y": 64}' > /tmp/psf.json
deconvolution \\
--data-path=flowdec/datasets/bars-25pct/data.tif \\
--psf-config-path=/tmp/psf.json \\
--output-path=/tmp/result.tif \\
--n-iter=25 \\
--log-level=DEBUG""")
    parser.add_argument(
        "--data-path",
        required=True,
        metavar='DATA_PATH',
        help="Path to image file containing data to deconvolve (any format compatible with skimage.io.imread)"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        metavar='OUTPUT_PATH',
        help="Path in which deconvolution result will be saved (any format compatible with skimage.io.imsave)"
    )
    parser.add_argument(
        "--psf-path",
        required=False,
        default=None,
        metavar='PSF_PATH',
        help="Path to image file containing point spread function (any format compatible with skimage.io.imread); "
        "One of either this or --psf-config-path must be supplied"
    )
    parser.add_argument(
        "--psf-config-path",
        required=False,
        default=None,
        metavar='PSF_CONFIG_PATH',
        help="Path to file containing PSF configuration (see flowdec.psf.from_file for details); "
        "One of either this or --psf-path must be supplied"
    )
    parser.add_argument(
        "--n-iter",
        required=False,
        default=25,
        metavar='N_ITERATIONS',
        help="Number of deconvolution iterations to run (default is 25)"
    )
    parser.add_argument(
        "--log-level",
        required=False,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARN', 'ERROR'],
        metavar='LOG_LEVEL',
        help="Logging level name (default is 'INFO')"
    )
    return parser


def resolve_psf(args, logger):
    if args.psf_path and args.psf_config_path:
        raise ValueError('Must supply PSF file path or PSF config path but not both')
    if not args.psf_path and not args.psf_config_path:
        raise ValueError('Must supply either PSF file path or PSF config path')

    # If a PSF data file was given, load it directly
    if args.psf_path:
        return io.imread(args.psf_path)
    # Otherwise, load PSF configuration file and generate a PSF from that
    else:
        psf = fd_psf.GibsonLanni.load(args.psf_config_path)
        logger.info('Loaded psf with configuration: {}'.format(psf.to_json()))
        return psf.generate()


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        format='%(levelname)s:%(asctime)s:%(message)s',
        level=logging.getLevelName(args.log_level.upper())
    )
    logger = logging.getLogger('DeconvolutionCLI')

    acq = fd_data.Acquisition(
        data=io.imread(args.data_path),
        kernel=resolve_psf(args, logger)
    )
    logger.debug('Loaded data with shape {} and psf with shape {}'.format(acq.data.shape, acq.kernel.shape))

    logger.info('Beginning deconvolution of data file "{}"'.format(args.data_path))
    start_time = timer()

    # Initialize deconvolution with a padding minimum of 1, which will force any images with dimensions
    # already equal to powers of 2 (which is common with examples) up to the next power of 2
    algo = fd_restoration.RichardsonLucyDeconvolver(n_dims=acq.data.ndim, pad_min=[1, 1, 1]).initialize()
    res = algo.run(acq, niter=args.n_iter)

    end_time = timer()
    logger.info('Deconvolution complete (in {:.3f} seconds)'.format(end_time - start_time))

    io.imsave(args.output_path, res.data)
    logger.info('Result saved to "{}"'.format(args.output_path))
