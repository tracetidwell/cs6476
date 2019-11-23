import os
import glob
import numpy as np
from skimage.io import imread


def computeMHI(directoryName, threshold=5000):

    depth_files = glob.glob(os.path.join(directoryName, '*.pgm'))
    depth_files = np.sort(depth_files)

    depth_ims = []
    for depth_file in depth_files:
        depth_im = imread(depth_file)
        depth_ims.append(depth_im)

    foreground_ims = []
    for i in range(1, len(depth_ims)):
        diff = np.abs(depth_ims[i] - depth_ims[i-1])
        diff[diff < threshold] = 0
        diff[diff >= threshold] = 1
        foreground_ims.append(diff)

    mhi = np.zeros(foreground_ims[0].shape)
    for i, foreground_im in enumerate(foreground_ims):
        mhi[foreground_im == 1] = i + 1
        mhi[foreground_im == 0] -= 1
    mhi[mhi < 0] = 0
    mhi /= np.max(mhi)

    return mhi
