import numpy as np
import argparse
import scipy.io
import glob
#from scipy import misc
from skimage import io
import matplotlib.pyplot as plt
from displaySIFTPatches import displaySIFTPatches
from selectRegion import roipoly
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from skimage.color import rgb2gray
import matplotlib.cm as cm
import pylab as pl
import pdb


def raw_descriptor_matches(mat_file, save_ims):

    mat = scipy.io.loadmat('twoFrameData.mat')
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.imshow(mat['im1'])
    ax.set_xticks([])
    ax.set_yticks([])
    roi = roipoly(roicolor='r')
    idx = roi.getIdx(mat['im1'], mat['positions1'])

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.imshow(mat['im1'])
    ax.set_xticks([])
    ax.set_yticks([])
    roi.displayROI()
    if save_ims:
        plt.savefig('raw_descriptor_match_im1.png', bbox_inches='tight')
    plt.show()

    descriptors1 = mat['descriptors1'][idx]
    descriptors2 = mat['descriptors2']
    match_idxs = np.argmin(np.sum((descriptors1 - descriptors2[:, None])**2, axis=2), axis=0)
    #np.save('descriptors.npy', mat['descriptors1'][Ind])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.imshow(mat['im2'])
    corners = displaySIFTPatches(mat['positions2'][match_idxs], mat['scales2'][match_idxs], mat['orients2'][match_idxs])
    for j in range(len(corners)):
        ax.plot([corners[j][0][1], corners[j][1][1]], [corners[j][0][0], corners[j][1][0]], color='g', linestyle='-', linewidth=1)
        ax.plot([corners[j][1][1], corners[j][2][1]], [corners[j][1][0], corners[j][2][0]], color='g', linestyle='-', linewidth=1)
        ax.plot([corners[j][2][1], corners[j][3][1]], [corners[j][2][0], corners[j][3][0]], color='g', linestyle='-', linewidth=1)
        ax.plot([corners[j][3][1], corners[j][0][1]], [corners[j][3][0], corners[j][0][0]], color='g', linestyle='-', linewidth=1)
    ax.set_xlim(0, mat['im2'].shape[1])
    ax.set_ylim(0, mat['im2'].shape[0])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().invert_yaxis()
    if save_ims:
        plt.savefig('raw_descriptor_match_im2.png', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_file', default='twoFrameData.mat', help='.mat file containing frames data to compare')
    parser.add_argument('--save_ims', action='store_true', help='Flag whether to save images')
    args = parser.parse_args()

    raw_descriptor_matches(args.mat_file, args.save_ims)
