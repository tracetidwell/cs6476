import glob
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from huMoments import huMoments


def generateAllHuMoments(mhis, scaled=True, save=False, save_dir='allHuMoments.npy',
						 hu_moment_func=huMoments):

    h, w, n_mhis = mhis.shape

    moments = np.zeros((n_mhis, 7))
    for i in range(n_mhis):
        moments[i] = hu_moment_func(mhis[:, :, i], scaled)

    if save:

        np.save(save_dir, moments)

    return moments
