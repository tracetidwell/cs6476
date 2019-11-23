import glob
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from huMoments import huMoments


def generateImprovedAllHuMoments(mhis, scaled=True, save=False, save_dir='allHuMoments.npy'):

    h, w, n_mhis = mhis.shape
    moments = np.zeros((n_mhis, 7))

    for i in range(n_mhis):

    	cropped_mhi = mhis[:, :, i].copy()
    	i_idxs, j_idxs = np.where(cropped_mhi != 0)
		i_min, i_max = i_idxs[np.argmin(i_idxs)], i_idxs[np.argmax(i_idxs)]
		j_min, j_max = j_idxs[np.argmin(j_idxs)], j_idxs[np.argmax(j_idxs)]
		cropped_mhi = cropped_mhi[i_min:i_max+1, j_min:j_max+1]
        moments[i] = huMoments(cropped_mhi, scaled)

    if save:

        np.save(save_dir, moments)

    return moments
