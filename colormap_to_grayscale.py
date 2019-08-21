#!/usr/bin/env python3

import argparse
import functools
import os

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
        # When the colormap contains repeated colors,
        # splrep throws ValueError: Invalid inputs.
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
    if max_dist >= 0:
        image_grayscale[image_cmap_dist > max_dist] = np.nan
    return image_grayscale, image_cmap_dist

def save_image_png(image, filename):
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(filename)

def save_image_npy(image, filename):
    np.save(filename, image, allow_pickle=False)

def save_image_fits(image, filename):
    from astropy.io import fits
    hdulist = fits.HDUList(fits.PrimaryHDU(image))
    hdulist.writeto(filename, overwrite=True)

def save_image(img, filename, overwrite=False):
    if os.path.exists(filename) and not overwrite:
        msg = "output file '{}' exists, use -O to overwrite it"
        raise OSError(msg.format(filename))
    _, ext = os.path.splitext(filename)
    if ext == '.png':
        save_image_png(img, filename)
    elif ext == '.npy':
        save_image_npy(img, filename)
    elif ext == '.fits':
        save_image_fits(img, filename)
    else:
        raise ValueError('Unsupported output extension: ' + ext)

def plot_debug_figure(image, cmap, image_grayscale, image_cmap_dist, save_to):
    import matplotlib as mpl
    mpl.use('pdf')
    import matplotlib.pyplot as plt
    plt.figure(clear=True, figsize=[12, 6])
    gs = mpl.gridspec.GridSpec(
        2, 4,
        left=.01, right=.99,
        top=.94, bottom=.08,
        hspace=0,
        wspace=0.1,
        height_ratios=[1, .05],
        )
    ax_input  = plt.subplot(gs[0, 0])
    cax_input = plt.subplot(gs[1, 0])
    ax_orig   = plt.subplot(gs[0, 1])
    cax_orig  = plt.subplot(gs[1, 1])
    ax_gray   = plt.subplot(gs[0, 2])
    cax_gray  = plt.subplot(gs[1, 2])
    ax_dist   = plt.subplot(gs[0, 3])
    cax_dist  = plt.subplot(gs[1, 3])

    # original cmap
    rgb_names = ('red', 'green', 'blue')
    x = np.linspace(0, 1, len(cmap))
    cdict = {k: list(zip(x, v, v))
             for k, v in zip(rgb_names, cmap.T)}
    cm_orig = mpl.colors.LinearSegmentedColormap('cm_orig', cdict)

    # gray cmap with red NaNs
    cm_gray = mpl.cm.gray
    cm_gray.set_bad('r')

    imshow_kw = dict(
        origin='upper',
        )
    cbar_kw = dict(
        orientation='horizontal',
        aspect=1/40,
        )

    # plot input image and cmap
    ax_input.imshow(image, **imshow_kw)
    height = len(cmap) // 10
    cmap_2d = np.repeat(cmap, height, axis=0).reshape(-1, height, 3).swapaxes(0, 1)
    cax_input.imshow(cmap_2d, **imshow_kw)

    # plot rendered grayscale and image-cmap distance
    im_orig = ax_orig.imshow(image_grayscale, cmap=cm_orig, vmin=0, vmax=1, **imshow_kw)
    im_gray = ax_gray.imshow(image_grayscale, cmap=cm_gray, vmin=0, vmax=1, **imshow_kw)
    im_dist = ax_dist.imshow(image_cmap_dist, cmap=cm_gray, vmin=0, **imshow_kw)

    # add colorbars
    cb_orig = plt.colorbar(im_orig, cax=cax_orig, **cbar_kw)
    cb_gray = plt.colorbar(im_gray, cax=cax_gray, **cbar_kw)
    cb_dist = plt.colorbar(im_dist, cax=cax_dist, **cbar_kw)

    # set ticks
    for ax in (ax_input, cax_input, ax_orig, ax_gray, ax_dist):
        ax.set_xticks([])
        ax.set_yticks([])
    for cb in (cb_orig, cb_gray):
        cb.set_ticks([0, 1])

    # set titles
    ax_input.set_title('Inputs')
    ax_orig.set_title('Rendered with colormap')
    ax_gray.set_title('Rendered in grayscale')
    ax_dist.set_title('Distance to colormap')

    plt.savefig(save_to)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=(
            'Convert an image rendered with a colormap (cmap) to grayscale.'
            ' This allows the image to be later rendered with any colormap.')
        )
    parser.add_argument(
        'cmap',
        type=str,
        help='Image of the colormap.',
        )
    parser.add_argument(
        'image',
        type=str,
        help='Image to convert, rendered with cmap.',
        )
    parser.add_argument(
        '-o', '--output-image',
        type=str,
        help='Output grayscale image (.png, .npy, or .fits).'
        )
    parser.add_argument(
        '-O', '--overwrite',
        action='store_true',
        help='Overwrite output_image.'
        )
    parser.add_argument(
        '--cmap-orientation',
        type=str,
        default='auto',
        help=('Orientation of cmap (auto, vertical, or horizontal).'),
        )
    parser.add_argument(
        '--cmap-size',
        type=int,
        default=255,
        help=('Size of the internal cmap used for the inversion'
              ' (default: 255).'),
        )
    parser.add_argument(
        '-m', '--max-dist',
        type=float,
        default=0.05,
        help=('Colors in the image which are not in the cmap (!) might be'
              ' inverted into nonsensical values.'
              ' This option sets the maximum distance between a color in the'
              ' image and colors in the cmap.'
              ' Colors further away are transformed into NaN values.'
              ' Distances are computed in the RGB space,'
              ' with values between 0 and 1.'
              ' Negative values ignore this threshold.'
              ' (Default: 0.05)'
              ),
        )
    parser.add_argument(
        '-d', '--debug-figure',
        action='store_true',
        help='Save a debug figure.')
    args = parser.parse_args()

    if not args.output_image:
        path, _ = os.path.splitext(args.image)
        args.output_image = path + '-grayscale.png'

    if os.path.exists(args.output_image) and not args.overwrite:
        msg = "output file '{}' exists, use -O to overwrite it"
        raise OSError(msg.format(args.output_image))

    image = load_image_rgb(args.image)
    cmap = load_image_rgb(args.cmap)
    cmap = cmap_to_1d(cmap, args.cmap_orientation)
    cmap = resize_cmap(cmap, args.cmap_size)
    image_grayscale, image_cmap_dist = cmap_to_grayscale(cmap, image, max_dist=args.max_dist)

    save_image(image_grayscale, args.output_image, overwrite=args.overwrite)

    if args.debug_figure:
        path, _ = os.path.splitext(args.image)
        filename = path + '-debug.pdf'
        plot_debug_figure(
            image, cmap, image_grayscale, image_cmap_dist, filename)
