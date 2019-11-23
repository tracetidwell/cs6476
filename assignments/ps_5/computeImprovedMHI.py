import glob
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread


def computeImprovedMHI(directoryName, threshold=12000):

    # depth_files = glob.glob(os.path.join(directoryName, '*.pgm'))
    # depth_files = np.sort(depth_files)

    # depth_ims = []
    # for depth_file in depth_files:
    #     depth_im = imread(depth_file)
    #     depth_ims.append(depth_im)

    # foreground_ims = []
    # for depth_im in depth_ims:
    #     foreground_im = depth_im.copy()
    #     foreground_im[foreground_im > threshold] = 0
    #     foreground_ims.append(foreground_im)

    # difference_ims = []
    # for i in range(1, len(foreground_ims)):
    #     foreground_im1 = foreground_ims[i-1].copy()
    #     foreground_im2 = foreground_ims[i].copy()
    #     foreground_im1[foreground_im1>0] = 1
    #     foreground_im2[foreground_im2>0] = 1
    #     difference_im = foreground_im2 - foreground_im1
    #     difference_im[difference_im != 0] = 1
    #     # difference_im[470:] = 0
    #     # difference_im[250:, 550:] = 0
    #     difference_ims.append(difference_im)

    depth_files = glob.glob(os.path.join(directoryName, '*.pgm'))
    depth_files = np.sort(depth_files)

    depth_ims = []
    for depth_file in depth_files:
        depth_im = imread(depth_file)
        depth_ims.append(depth_im)

    foreground_ims = []
    for depth_im in depth_ims:
        foreground_im = depth_im.copy()
        foreground_im[foreground_im < threshold] = 1
        foreground_im[foreground_im != 1] = 0
        foreground_ims.append(foreground_im)

    mhi = np.zeros(foreground_ims[0].shape)
    for i, foreground_im in enumerate(foreground_ims):
        mhi[foreground_im == 1] = i + 1
        mhi[foreground_im == 0] -= 1
    mhi[mhi < 0] = 0
    mhi[470:] = 0
    mhi[250:, 525:] = 0
    mhi /= np.max(mhi)

    return mhi
