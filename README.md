# colormap_to_grayscale

Convert an image rendered with a colormap to grayscale.

![Sample input and output: image rendered with the jet
colormap, converted back into grayscale.](example/overview.png)

## Installation

Simply call [`colormap_to_grayscale.py`](colormap_to_grayscale.py) from the command line.
You may add it to your `$PATH`, or directly run `./colormap_to_grayscale.py`.

This script depends on fairly standard libraries:
[PIL](https://python-pillow.org),
[numpy](https://www.numpy.org),
and [scipy](https://www.scipy.org/scipylib/index.html).

## Example

Sample inputs and output are given in the [`example/`](example/) directory.

The output is generated with the following command:

~~~bash
./colormap_to_grayscale.py example/input_cmap.png example/input_image.png -o example/output.png
~~~

## Usage

Use `colormap_to_grayscale.py` from the command line:

~~~
usage: colormap_to_grayscale.py [-h] [-o OUTPUT_IMAGE] [-O]
                                [--cmap-orientation CMAP_ORIENTATION]
                                [--cmap-size CMAP_SIZE] [-m MAX_DIST] [-d]
                                cmap image

Convert an image rendered with a colormap (cmap) to grayscale. This allows the
image to be later rendered with any colormap.

positional arguments:
  cmap                  Image of the colormap.
  image                 Image to convert, rendered with cmap.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_IMAGE, --output-image OUTPUT_IMAGE
                        Output grayscale image (.png, .npy, or .fits).
  -O, --overwrite       Overwrite output_image.
  --cmap-orientation CMAP_ORIENTATION
                        Orientation of cmap (auto, vertical, or horizontal).
  --cmap-size CMAP_SIZE
                        Size of the internal cmap used for the inversion
                        (default: 255).
  -m MAX_DIST, --max-dist MAX_DIST
                        Colors in the image which are not in the cmap (!)
                        might be inverted into nonsensical values. This option
                        sets the maximum distance between a color in the image
                        and colors in the cmap. Colors further away are
                        transformed into NaN values. Distances are computed in
                        the RGB space, with values between 0 and 1. Negative
                        values ignore this threshold. (Default: 0.05)
  -d, --debug-figure    Save a debug figure.
~~~

## Licence

This tool is distributed under an open source MIT license. See `LICENSE.txt`
