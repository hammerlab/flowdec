""" Collection of random utilities used in exploratory notebooks """

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from functools import partial

# ####################### #
# Visualization Utilities #
# ####################### #


def plot_zstack_2d(img, ncols=5, in_per_col=3, in_per_row=3, cmap='Greys_r', idx_offset=0):
    n = img.shape[0]
    nrow = int(np.ceil(n / float(ncols)))
    ncol = min(n, ncols)
    fig, axs = plt.subplots(nrow, ncol)
    axs = axs.ravel()
    fig.set_size_inches((in_per_col * ncols, in_per_row * nrow))
    for i in range(n):
        axs[i].imshow(img[i], cmap=cmap)
        axs[i].set_title('Index {}'.format(idx_offset + i))
    for i in range(len(axs)):
        axs[i].axis('off')


def plot_zstack_3d(data, cmap='Greys_r'):
    return ZStackViewer(data, cmap=cmap).run()


class ZStackViewer(object):

    def __init__(self, volume, cmap='Greys_r'):
        self.volume = volume
        self.cmap = cmap

    def run(self):
        """ Use this to launch interactive window for 3D image visualization """
        fig, ax = plt.subplots()
        ax.volume = self.volume
        ax.index = self.volume.shape[0] // 2
        ax.imshow(self.volume[ax.index], cmap=self.cmap)
        ax.set_title('Z-Index ' + str(ax.index))
        fig.canvas.mpl_connect('key_press_event', process_key)
        return fig, ax

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    """Go to the previous slice."""
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    ax.set_title('Z-Index ' + str(ax.index))

def next_slice(ax):
    """Go to the next slice."""
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    ax.set_title('Z-Index ' + str(ax.index))


def plot_img_preview(img, zstart=None, zstop=None, cmap='viridis', proj_figsize=(12,4), **kwargs):
    """Plot z-projection of volume as well as individual z-slices"""
    plt.imshow(img.max(axis=0), cmap=cmap)
    plt.gcf().set_size_inches(proj_figsize)
    plt.gca().set_title('Max Projection (Over {} Z-Slices)'.format(img.shape[0]))
    plot_zstack_2d(img[slice(zstart, zstop),:,:], idx_offset=zstart if zstart else 0,
                           cmap=cmap, **kwargs)


rotate_xy = partial(rotate, axes=(1, 2))
rotate_yz = partial(rotate, axes=(0, 1))
rotate_xz = partial(rotate, axes=(0, 2))


def plot_rotations(img, projection=lambda img: img.max(axis=0), cmap='viridis', figsize=(12, 12)):
    """Plot the same volume in a 3x3 grid with 0, 45, and 90 degree rotations around each axis"""
    fig, axs = plt.subplots(3, 3)
    fig.set_size_inches(figsize)
    rotate_fns = [rotate_xy, rotate_yz, rotate_xz]
    rotate_angles = [0, 45, 90]
    for i in range(len(rotate_fns)):
        for j in range(len(rotate_angles)):
            im = rotate_fns[i](img, angle=rotate_angles[j])
            im = projection(im)
            axs[i, j].imshow(im, cmap=cmap)


# ######################### #
# Data Generation Utilities #
# ######################### #


def save_dataset(name, acq, path, dtype=np.float32):
    from skimage import io
    import os

    p = os.path.join(path, name)
    if not os.path.exists(p):
        os.mkdir(p)

    print('Exporting data for dataset "{}" to path {}'.format(name, p))

    io.imsave(os.path.join(p, 'data.tif'), acq.data.astype(dtype))
    io.imsave(os.path.join(p, 'kernel.tif'), acq.kernel.astype(dtype))
    if acq.actual is not None:
        io.imsave(os.path.join(p, 'actual.tif'), acq.actual.astype(dtype))