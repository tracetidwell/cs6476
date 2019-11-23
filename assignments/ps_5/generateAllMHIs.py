import glob
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from computeMHI import computeMHI


def generateAllMHIs(base_dir='PS5_Data', threshold=40000, save=False,
                    save_dir='allMHIs.py', compute_func=computeMHI):

    actions = sorted(os.listdir(base_dir))
    mhis = []

    for action in actions:

        actions_dir = os.path.join(base_dir, action)
        sequences = sorted(os.listdir(actions_dir))

        for sequence in sequences:

            sequence_dir = os.path.join(actions_dir, sequence)
            mhi = compute_func(sequence_dir, threshold)
            h, w = mhi.shape
            mhis.append(mhi.reshape(h, w, 1))

    mhis = np.concatenate(mhis, axis=2)

    if save:

        np.save(save_dir, mhis)

    return mhis
