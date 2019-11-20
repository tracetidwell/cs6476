import glob
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from huMoments import huMoments


def generateAllHuMoments(mhis_dir=None, scaled=True, save=False, save_dir=''):

    if mhis_dir is None:
        mhis = np.load('allMHIs.npy')

    h, w, n_mhis = mhis.shape

    moments = np.zeros((n_mhis, 7))
    for i in range(n_mhis):
        moments[i] = huMoments(mhis[:, :, i], scaled)

    if save:

        np.save(os.path.join(save_dir, 'allHuMoments.npy'), moments)

    return moments
