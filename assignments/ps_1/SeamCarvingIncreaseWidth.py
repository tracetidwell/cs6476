import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import convolve
from scipy import ndimage
from skimage.color import rgb2gray

from energy_image import energy_image
from cumulative_minimum_energy_map import cumulative_minimum_energy_map
from find_optimal_vertical_seams import find_optimal_vertical_seams
from reduceWidth import reduceWidth

def load_image(filename):
    return mpimg.imread(filename)


def increase_width_by(im, vertical_seams):

    old_im = im.copy()

    for j in range(vertical_seams.shape[1]):

        new_im = np.zeros((old_im.shape[0], old_im.shape[1]+1, old_im.shape[2])).astype('uint8')

        for i in range(vertical_seams.shape[0]):
            idx = vertical_seams[i, j]
            new_pixel = np.mean(old_im[i, idx-1:idx+2], 0).astype('uint8')
            new_im[i, :idx+1] = old_im[i, :idx+1]
            new_im[i, idx+1] = new_pixel
            new_im[i, idx+2:] = old_im[i, idx+1:]

        old_im = new_im.copy()

    return new_im


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--filenames', nargs='*', type=str, action='store', default=['Prague', 'Mall'],
                        help='Filename of image to load')
    parser.add_argument('--n_pixels', type=int, default=100, help='Number of pixels by which to reduce width')
    parser.add_argument('--energy_function', type=str, default='gradient', help='Energy function to use in calculating Energy Image')
    parser.add_argument('--ef_in_out', action='store_true')
    args = parser.parse_args()

    for filename in args.filenames:

        im = load_image('inputSeamCarving{}.jpg'.format(filename))
        energyImage = energy_image(im, args.energy_function)
        cumulativeEnergyMap = cumulative_minimum_energy_map(energyImage, 'VERTICAL')
        vertical_seams = find_optimal_vertical_seams(cumulativeEnergyMap, args.n_pixels)
        increasedIm = increase_width_by(im, vertical_seams)

        if args.ef_in_out:
            mpimg.imsave('outputIncreaseWidth_{}_{}.png'.format(args.energy_function, filename), increasedIm)
        else:
            mpimg.imsave('outputIncreaseWidth{}.png'.format(filename), increasedIm)

        plt.imshow(increasedIm)
        plt.show()
