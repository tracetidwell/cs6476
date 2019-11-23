import glob
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread


def computeImprovedMHI(directoryName, threshold1=40000, threshold2=30000):

    depth_files = glob.glob(os.path.join(directoryName, '*.pgm'))
    depth_files = np.sort(depth_files)

    depth_ims = []
    for depth_file in depth_files:
        depth_im = imread(depth_file)
        depth_ims.append(depth_im)

    # foreground_ims = []
    # for i in range(1, len(depth_ims)):
    #     diff = depth_ims[i] - depth_ims[i-1]
    #     diff[diff >= threshold] = 0
    #     diff[diff < threshold] = 1
    #     foreground_ims.append(diff)

    # foreground_ims = []
    # for depth_im in depth_ims:
    #     foreground_im = depth_im.copy()
    #     foreground_im[foreground_im < threshold] = 1
    #     foreground_im[foreground_im != 1] = 0
    #     foreground_ims.append(foreground_im)

    foreground_ims = []
    for i in range(1, len(depth_ims)):
        d0 = depth_ims[i-1].copy()
        d1 = depth_ims[i].copy()
        d0[d0 > threshold1] = 0
        d1[d1 > threshold1] = 0
        diff = np.abs(d1 - d0)
        diff[diff < threshold2] = 0
        diff[diff >= threshold2] = 1
        foreground_ims.append(diff)

    mhi = np.zeros(foreground_ims[0].shape)
    for i, foreground_im in enumerate(foreground_ims):
        mhi[foreground_im == 1] = i + 1
        mhi[foreground_im == 0] -= 1
    mhi[mhi < 0] = 0
    mhi /= np.max(mhi)

    return mhi
