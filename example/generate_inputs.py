#!/usr/bin/env python3

'''
Generate sample inputs: 'input_cmap.png' and 'input_image.png'.
'''

from PIL import Image
import matplotlib.cm
import numpy as np
import scipy.misc

def save_image_png(image, filename):
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(filename)

# sample image
img = scipy.misc.face(gray=True)

# render img with jet colormap
img_rgb = matplotlib.cm.jet(img / 255)
cmap_rgb = matplotlib.cm.jet(np.linspace(0, 1, 255))
cmap_rgb = np.expand_dims(cmap_rgb, 0).repeat(20, axis=0)

# write images to files
save_image_png(img_rgb, 'input_image.png')
save_image_png(cmap_rgb, 'input_cmap.png')
