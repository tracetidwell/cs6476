import glob
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from computeMHI import computeMHI


def generateAllMHIs(base_dir=None, threshold=39000, save=False,
                    save_dir=''):

    if base_dir is None:
        base_dir = 'PS5_Data'
        actions = sorted(os.listdir(base_dir))

    mhis = []

    for action in actions:

        actions_dir = os.path.join(base_dir, action)
        sequences = sorted(os.listdir(actions_dir))

        for sequence in sequences:

            sequence_dir = os.path.join(actions_dir, sequence)
            mhi = computeMHI(sequence_dir, threshold)
            h, w = mhi.shape
            mhis.append(mhi.reshape(h, w, 1))

    mhis = np.concatenate(mhis, axis=2)

    if save:

        np.save(os.path.join(save_dir, 'allMHIs.npy'), mhis)

    return mhis
