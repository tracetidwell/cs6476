import numpy as np

from utils import *


def huMoments(mhi, scaled=True):

    im_moment_00 = image_moment(mhi, 0, 0, scaled)
    im_moment_10 = image_moment(mhi, 1, 0, scaled)
    im_moment_01 = image_moment(mhi, 0, 1, scaled)

    x_bar = im_moment_10 / im_moment_00
    y_bar = im_moment_01 / im_moment_00

    moments = np.zeros(7)

    for i in range(7):
        moment = eval(f'h{i+1}')
        moments[i] = moment(mhi, x_bar, y_bar, scaled)

    return moments
