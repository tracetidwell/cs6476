import argparse
import math
import operator
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve

from utils import *
from quantizeRGB import quantizeRGB
from quantizeHSV import quantizeHSV
from computeQuantizationError import computeQuantizationError
from getHueHists import getHueHists


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Quantize images')
    parser.add_argument('--filename', type=str, default='fish.jpg', help='Filename of image to load')
    parser.add_argument('--k', nargs='*', type=int, action='store', default=[3, 5, 7, 9],
                        help='Filename of image to load')
    args = parser.parse_args()

    errors = {'rgb': [], 'hsv': []}
    #hists = {'rgb': [], 'hsv': []}
    rgb_im = load_image(args.filename)

    for k in args.k:

        segmented_rgb_im, mean_colors = quantizeRGB(rgb_im, k)
        cv2.imwrite('k={}_segmented_rgb_{}'.format(k, args.filename), segmented_rgb_im)

        segmented_hsv_im, mean_hues = quantizeHSV(rgb_im, k)
        cv2.imwrite('k={}_segmented_hsv_{}'.format(k, args.filename), segmented_hsv_im)

        quantized_rgb_error = computeQuantizationError(rgb_im, segmented_rgb_im)
        quantized_hsv_error = computeQuantizationError(rgb_im, segmented_hsv_im)
        errors['rgb'].append(quantized_rgb_error)
        errors['hsv'].append(quantized_hsv_error)

        hist_eq, hist_cl = getHueHists(rgb_im, k, False)

    plt.figure(figsize=(8,8))
    plt.xlabel('k (Number of Clusters)')
    plt.ylabel('Quantzation Error (SSD)')
    plt.plot(args.k, errors['rgb'], label='RGB Quantization Error')
    plt.plot(args.k, errors['hsv'], label='HSV Quantization Error')
    plt.legend()
    plt.savefig('SSD vs. k.jpg')
    #plt.show()

        # im = load_image('inputSeamCarving{}.jpg'.format(filename))
        # energyImage = energy_image(im, args.energy_function)
        #
        # reducedIm, reducedEnergyImage = reduce_width_by(im, energyImage, args.energy_function, args.width)
        # reducedIm, reducedEnergyImage = reduce_height_by(reducedIm, reducedEnergyImage, args.energy_function, args.height)
        #
        # if args.ef_in_out:
        #     mpimg.imsave('outputReduce_{}_{}.png'.format(args.energy_function, filename), reducedIm)
        # else:
        #     mpimg.imsave('outputReduce{}.png'.format(filename), reducedIm)
        #
        # plt.imshow(reducedIm)
        # plt.show()
