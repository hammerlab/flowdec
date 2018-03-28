

import os
import json
import numpy as np
from os import path as osp
import tensorflow as tf
import timeit
from flowdec import restoration as fd_restoration
from flowdec import data as fd_data
from argparse import ArgumentParser
import logging 
import sys 
import re
from skimage.external.tifffile import imread, imsave
from skimage.exposure import rescale_intensity
from shutil import copyfile
from collections import OrderedDict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('AkoyaDeconCLI')


def make_arg_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        required=True,
        metavar='RAW',
        help="Path to original data directory containing acquisitions"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        metavar='INPUT',
        help="Path to directory containing images stacks from CODEX processor"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar='OUTPUT',
        help="Path to directory to contain results"
    )
    parser.add_argument(
        "--psf-dir",
        required=True,
        metavar='PSFDIR',
        help="Path to directory containing psf stacks"
    )
    parser.add_argument(
        "--psf-pattern",
        required=True,
        metavar='PSFPATTERN',
        help="PSF file naming pattern (e.g. 'psf-ch\{channel_id\}.tif' where channel_id is 1-based index)"
    )
    parser.add_argument(
        "--n-iter",
        required=False,
        default=10,
        help="Number of Richardson-Lucy iterations to execute (defaults to 10)"
    )
    parser.add_argument(
        "--dry-run",
        required=False,
        action='store_true',
        help="Flag indicating to only show inputs and proposed outputs"
    )
    return parser



def _get_files(directory, pattern):
    return [
        osp.join(directory, f) for f in os.listdir(directory)
        if re.match(pattern, f)
    ]

def init_output(args):
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

def copy_meta_files(args):
    files = _get_files(args.raw_dir, 'Experiment.json|channelNames.txt')

    logger.debug('Metadata files copied to output:')
    for f in files:
        dest = osp.join(args.output_dir, os.path.basename(f))
        logger.debug('\t{} -> {}'.format(f, dest))
        if not args.dry_run:
            copyfile(f, dest)
    return files

def _load_experiment_config(args):
    f = osp.join(args.raw_dir, 'Experiment.json')
    if not osp.exists(f):
        raise ValueError('Required experiment configuration file "{}" does not exist'.format(f))
    with open(f, 'r') as fd:
        return json.load(fd)

def _load_channel_names(args):
    f = osp.join(args.raw_dir, 'channelNames.txt')
    if not osp.exists(f):
        raise ValueError('Required channel names configuration file "{}" does not exist'.format(f))
    with open(f, 'r') as fd:
        return [l.strip() for l in fd.readlines() if l.strip()]

class AkoyaConfig(object):

    def __init__(self, exp_config, channel_names):
        self.exp_config = exp_config
        self.channel_names = channel_names

    def all_channel_names(self):
        return self.channel_names

    def n_cycles(self):
        return self.exp_config['num_cycles']

    def n_z_planes(self):
        return self.exp_config['num_z_planes']

    def n_channels_per_cycle(self):
        return len(self.exp_config['channel_names'])

    def n_actual_channels(self):
        return len(self.channel_names)

    def n_expected_channels(self):
        return self.n_cycles() * self.n_channels_per_cycle()


def load_config(args):
    config = AkoyaConfig(
        _load_experiment_config(args),
        _load_channel_names(args)
    )

    logger.debug('Experiment configuration summary:')
    logger.debug('\tNum cycles = {}'.format(config.n_cycles()))
    logger.debug('\tNum z planes = {}'.format(config.n_z_planes()))
    logger.debug('\tChannels expected per cycle = {}'.format(config.n_channels_per_cycle()))
    logger.debug('\tChannel names list length = {}'.format(config.n_actual_channels()))

    if config.n_actual_channels() != config.n_expected_channels():
        raise ValueError(
            'Full list of channel names does not have length equal '
            'to num_cycles * n_channels_per_cycle; '
            'n expected channel names = {}, n actual channel names = {}'
            .format(config.n_expected_channels(), config.n_actual_channels())
        )

    return config

def load_psfs(args, config):
    psfs = []
    for i in range(config.n_channels_per_cycle()):
        f = osp.join(args.psf_dir, args.psf_pattern.format(i+1))
        if not osp.exists(f):
            raise ValueError(
                'Expected PSF file "{}" does not exist; '
                'Num channels expected to have PSFs for = {}'
                .format(f, config.n_channels_per_cycle())
            )
        psfs.append((f, imread(f)))

    logger.debug('PSF stacks loaded:')
    for f, psf in psfs:
        logger.debug('\t{} --> shape = {}'.format(f, psf.shape))
    return [psf[1] for psf in psfs]


def img_generator(files):
    for f in files:
        yield f, imread(f)

def _validate_stack_shape(img, psf):
    if img.ndim != 5:
        raise ValueError(
            'Expecting 5 dimensions in image stack, '
            'given shape = {}'.format(img.shape)
        )
    ncyc, nz, nch, nh, nw = img.shape

    if ncyc != config.n_cycles():
        raise ValueError(
            'Expecting {} cycles but found {} in image stack'
            .format(config.n_cycles(), ncyc)
        )

    if nz != config.n_z_planes():
       raise ValueError(
            'Expecting {} z planes but found {} in image stack'
            .format(config.n_z_planes(), nz)
        )

    if nch != config.n_channels_per_cycle():
       raise ValueError(
            'Expecting {} channels but found {} in image stack'
            .format(config.n_channels_per_cycle(), nch)
        )

def run_deconvolution(args, psfs, config):
    files = _get_files(args.input_dir, '.*\.tif$')

    # Tone down TF logging, though only the first setting below actually
    # seems to make any difference
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    tf.logging.set_verbosity(tf.logging.WARN)
    session_config = tf.ConfigProto(log_device_placement=False)

    n_iter = int(args.n_iter)
    algo = fd_restoration.RichardsonLucyDeconvolver(n_dims=3).initialize()
    
    # Stacks load as (cycles, z, channel, height, width)
    imgs = img_generator(files)
    for i, (f, img) in enumerate(imgs):
        logger.debug(
            'Deconvolving stack file "{}" ({} of {}) --> shape = {}, dtype = {}'
            .format(f, i+1, len(files), img.shape, img.dtype)
        )
        
        _validate_stack_shape(img, config)
        ncyc, nz, nch, nh, nw = img.shape

        res_stack = []
        for icyc in range(ncyc):
            res_ch =[]
            for ich in range(nch):
                psf = psfs[ich]
                acq = fd_data.Acquisition(data=img[icyc,:,ich,:,:], kernel=psfs[ich])

                if args.dry_run:
                    continue
                res = algo.run(acq, niter=n_iter, session_config=session_config).data
                res_ch.append(res)

            if args.dry_run:
                continue
            # Stack results to (nz, nch, nh, nw)
            res_ch = np.stack(res_ch, 1)
            if list(res_ch.shape) != [nz, nch, nh, nw]:
                raise ValueError(
                    'Stack across channels has wrong shape --> expected = {}, actual = {}'
                    .format([nz, nch, nh, nw], list(res_ch.shape))
                )
            res_stack.append(res_ch)
 
        if args.dry_run:
            continue

        res_stack = np.stack(res_stack, 0)
        # if list(res_stack.shape) != list(img.shape):
        #     raise ValueError(
        #         'Final stack has wrong shape --> expected = {}, actual = {}'
        #         .format(list(img.shape), list(res_stack.shape))
        #     )

        res_file = osp.join(args.output_dir, osp.basename(f))
        logger.debug(
            'Saving deconvolved stack slice to "{}" --> shape = {}, dtype = {}'
            .format(res_file, res_stack.shape, res_stack.dtype)
        )
        # See tiffwriter docs at http://scikit-image.org/docs/dev/api/skimage.external.tifffile
        # .html#skimage.external.tifffile.TiffWriter for more info on how scikit-image
        # handles imagej formatting -- the docs aren't very explicit but they do mention
        # that with 'imagej=True' it can handle arrays up to 6 dims in TZCYXS order
        imsave(res_file, res_stack, imagej=True)
    return None

if __name__ == "__main__":
    # Parse arguments
    parser = make_arg_parser()
    args = parser.parse_args()

    logger.info('Beginning Stack Deconvolution')
    logger.debug('Arguments:')
    for arg in ['input_dir', 'output_dir', 'psf_dir', 'psf_pattern', 'n_iter', 'dry_run']:
        logger.debug('\t{}="{}"'.format(arg, getattr(args, arg)))
 
    
    logger.info('Initializing output directory')
    init_output(args)

    logger.info('Initializing metadata files')
    copy_meta_files(args)

    logger.info('Loading experiment configuration')
    config = load_config(args)

    logger.info('Loading PSF images')
    psfs = load_psfs(args, config)

    logger.info('Running deconvolution')
    run_deconvolution(args, psfs, config)


# Execute on original data with blank cycles:
# python akoya_deconvolution.py --raw-dir=F:\2018-01-09-Run0 --input-dir=F:\2018-01-09-Run0-Out-1\1-Processor --output-dir=F:\2018-01-09-Run0-Out-1\2-Deconvolution --psf-dir=C:\Users\User\data\deconvolution\data\akoya_experiment_psf --n-iter=10 --psf-pattern="psf-ch{}.tif" --dry-run
# Execute on clean data (only 6 of 8 cycles):
# python akoya_deconvolution.py --raw-dir=F:\2018-01-09-Run0-Clean --input-dir=F:\2018-01-09-Run0-Clean-Out-1\1-Processor --output-dir=F:\2018-01-09-Run0-Clean-Out-1\2-Deconvolution --psf-dir=C:\Users\User\data\deconvolution\data\akoya_experiment_psf --n-iter=10 --psf-pattern="psf-ch{}.tif" --dry-run