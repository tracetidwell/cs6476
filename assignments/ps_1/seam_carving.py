import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import convolve
from scipy import ndimage
from skimage.color import rgb2gray

from energy_image import energy_image
from reduceWidth import reduceWidth
from reduceHeight import reduceHeight


def load_image(filename):
    return mpimg.imread(filename)


def reduce_width_by(im, energyImage, energy_function, n_pixels):

    for i in range(n_pixels):
        im, energyImage = reduceWidth(im, energyImage, energy_function)

    return im, energyImage


def reduce_height_by(im, energyImage, energy_function, n_pixels):

    for i in range(n_pixels):
        im, energyImage = reduceHeight(im, energyImage, energy_function)

    return im, energyImage


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--filenames', nargs='*', type=str, action='store',
                        help='Filename of image to load')
    parser.add_argument('--width', type=int, default=100, help='Number of pixels by which to reduce width')
    parser.add_argument('--height', type=int, default=100, help='Number of pixels by which to reduce height')
    parser.add_argument('--energy_function', type=str, default='gradient', help='Energy function to use in calculating Energy Image')
    parser.add_argument('--ef_in_out', action='store_true')
    args = parser.parse_args()


    for filename in args.filenames:

        im = load_image('inputSeamCarving{}.jpg'.format(filename))
        energyImage = energy_image(im, args.energy_function)

        reducedIm, reducedEnergyImage = reduce_width_by(im, energyImage, args.energy_function, args.width)
        reducedIm, reducedEnergyImage = reduce_height_by(reducedIm, reducedEnergyImage, args.energy_function, args.height)

        if args.ef_in_out:
            mpimg.imsave('outputReduce_{}_{}.png'.format(args.energy_function, filename), reducedIm)
        else:
            mpimg.imsave('outputReduce{}.png'.format(filename), reducedIm)

        plt.imshow(reducedIm)
        plt.show()
