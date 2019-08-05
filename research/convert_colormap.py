#!/usr/bin/env python3

import argparse
import functools

from PIL import Image
import numpy as np
import scipy.interpolate as sinterp

def load_image_rgb(filename):
    with Image.open(filename) as im_f:
        image = np.array(im_f.convert('RGB'))
    return image / 255

def cmap_to_1d(cmap, cmap_orientation):
    ny, nx, _ = cmap.shape
    if cmap_orientation == 'auto':
        if ny > nx:
            cmap_orientation = 'vertical'
        elif ny < nx:
            cmap_orientation = 'horizontal'
        else:
            msg = "Can't determine the orientation of cmap with shape {}x{}"
            msg = msg.format(ny, nx)
            raise ValueError(msg)

    if cmap_orientation == 'vertical':
        return np.median(cmap, axis=1)[::-1]
    if cmap_orientation == 'horizontal':
        return np.median(cmap, axis=0)
    raise ValueError('Unknown cmap orientation: {}'.format(cmap_orientation))

def resize_cmap(cmap, cmap_size):
    target_cmap_u = np.linspace(0, 1, cmap_size)

    if len(cmap) < cmap_size:
        cmap_tck, cmap_u = sinterp.splprep(cmap.T, s=0)
        # TODO: handle repeated colors in the cmap
        # (splrep throws ValueError: Invalid inputs.)
        cmap = sinterp.splev(target_cmap_u, cmap_tck)
        cmap = np.array(cmap).T

    if len(cmap) > cmap_size:
        cmap_u = np.linspace(0, 1, len(cmap))
        cmap_interp = sinterp.interp1d(cmap_u, cmap.T)
        cmap = cmap_interp(target_cmap_u).T

    assert cmap.shape == (cmap_size, 3)
    return cmap

def distance_to_cmap(cmap, arr):
    ''' Compute the color distance between a RGB 1D array and a color map '''
    nu, _ = cmap.shape
    na, _ = arr.shape
    dist_rgb = cmap - arr.reshape(na, 1, 3) # nu, na, 3
    dist = np.sqrt(np.sum(dist_rgb**2, axis=-1))
    min_dist = np.min(dist, axis=-1)
    argmin_dist = np.argmin(dist, axis=-1) / (nu - 1)
    return min_dist, argmin_dist

def cmap_to_grayscale(cmap, image, max_dist=None):
    ''' Convert an image rendered with a cmap to grayscale

    Parameters
    ==========
    cmap : (Nc, 3) ndarray
        The colormap.
    image : (Nx, Ny, 3) ndarray
        The image rendered with cmap.
    max_dist : float or None (default: None)
        The maximum distance between a color in the image and the colormap.
        Colors which are further away from the colormap than this distance are
        not inverted, and set to NaN.
        The distance is computed as:
            sqrt((R_cmap - R_img)**2 + (G_cmap - G_img)**2 + (B_cmap - B_img)**2)
            where R, G, and B are the RGB values between 0 and 1.

    Returns
    =======
    image_grayscale (Nx, Ny)
        The image converted into grayscale, with values between 0 and 1.
        Contains NaN values where image_cmap_dist > max_dist.
    image_cmap_dist (Nx, Ny)
        The distance between the image colors and the cmap.
    '''
    image_grayscale_and_dist = np.array(list(map(
        functools.partial(distance_to_cmap, cmap),
        image)))
    image_cmap_dist = image_grayscale_and_dist[:, 0, :]
    image_grayscale = image_grayscale_and_dist[:, 1, :]
    image_grayscale[image_cmap_dist > max_dist] = np.nan
    return image_grayscale, image_cmap_dist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert colormap')
    parser.add_argument(
        'cmap',
        type=str,
        help='image containing the original colormap',
        )
    parser.add_argument(
        'image',
        type=str,
        help='image to convert',
        )
    parser.add_argument(
        '--target-cmap',
        type=str,
        default='gray',
        help=('matplotlib colormap name, or image containing the target '
              'colormap (default: gray)')
        )
    parser.add_argument(
        '--cmap-orientation',
        type=str,
        default='auto',
        help=('orientation of the original cmap image '
              '(auto, vertical, or horizontal)'),
        )
    parser.add_argument(
        '--max-dist',
        type=float,
        default=0.05,
        help=('maximum distance (0-1) in the RGB space between a color in the '
              'image and the matched color in the colormap (default: 0.05)'),
        )
    parser.add_argument(
        '--cmap-size',
        type=int,
        default=255,
        help='size of the cmap used for the inversion (default: 255)',
        )
    args = parser.parse_args()

    image = load_image_rgb(args.image)
    cmap = load_image_rgb(args.cmap)
    cmap = cmap_to_1d(cmap, args.cmap_orientation)
    cmap = resize_cmap(cmap, args.cmap_size)
    image_grayscale, image_cmap_dist = cmap_to_grayscale(cmap, image, max_dist=args.max_dist)

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.ioff()

    plt.clf()
    plt.step(cmap[:, 0], 'r')
    plt.step(cmap[:, 1], 'g')
    plt.step(cmap[:, 2], 'b')
    plt.savefig(f'{args.cmap}_cmap.pdf')

    plt.clf()
    gray = mpl.cm.gray
    gray.set_bad('r')
    plt.imshow(image_grayscale, origin='upper', cmap=gray)
    plt.savefig(f'{args.image}_gray.pdf')

    plt.clf()
    plt.imshow(image_cmap_dist, origin='upper', cmap='gray')
    plt.colorbar()
    plt.savefig(f'{args.image}_dist.pdf')
