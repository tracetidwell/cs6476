import numpy as np
from skimage.color import rgb2gray
from scipy.signal import convolve, correlate
from skimage import filters

def energy_image(im, function='gradient'):

    if len(im.shape) == 3:
        im = rgb2gray(im.copy())

    if function == 'gradient':

        dx = np.abs(convolve(im, [[1, -1]], 'same'))
        dy = np.abs(convolve(im, [[1], [-1]], 'same'))

        return dx + dy

    elif function == 'gaussian':

        return filters.gaussian(im)

    elif function == 'median':

        return filters.median(im)

    elif function == 'laplace':

        return filters.laplace(im)

    elif function == 'sobel':

        dx = np.abs(correlate(im, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 'same'))
        dy = np.abs(correlate(im, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 'same'))

        return dx + dy

    elif function =='extended':

        dx = np.abs(convolve(im, [[2, 1, 0, -1, -2]], 'same'))
        dy = np.abs(convolve(im, [[2], [1], [0], [-1], [-2]], 'same'))

        return dx + dy

    elif function == 'disruption':

        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
        mean = correlate(im, kernel, 'same')

        return np.abs(im - mean)
