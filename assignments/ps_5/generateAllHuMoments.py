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


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--load_mhis', action='store_false', help='Flag for whether to not load MHIs')
	parser.add_argument('--mhis_path', type=str, default='allMHIs.npy', help='Path from which to load MHIs if flag is True')
	parser.add_argument('--base_dir', type=str, default='PS5_Data', help='Base directory for action sequences')
	parser.add_argument('--threshold', type=int, default=5000, help='Threshold to use in background subtraction')
	parser.add_argument('--save_mhis', action='store_true', help='Flag for whether to save MHIs')
	parser.add_argument('--save_mhis_path', type=str, default='allMHIs.py', help='Path to save MHIs to')
	parser.add_argument('--mhis_func', type=str, default='computeMHI', help='String with name of function to be used for computing MHI')
	parser.add_argument('--scaled', action='store_false', help='Flag for whether to not scale Hu Moments')
	parser.add_argument('--save_hus', action='store_true', help='Flag for whether to save Hu Moments')
	parser.add_argument('--save_hus_path', type=str, default='huVectors.npy', help='Path to save Hu Moments to')
	parser.add_argument('--hus_func', type=str, default='computeMHI', help='String with name of function to be used for computing Hu Moments')
	args = parser.parse_args()

	if args.load_mhis:
		mhis = np.load(args.mhis_path)
	else:
		mhis = generateAllMHIs(args.base_dir, args.threshold, args.save_mhis, args.save_mhis_path, eval(args.mhis_func))

	hu_moments = generateAllHuMoments(mhis, args.scaled, args.save_hus, args.save_hus_path, eval(args.hus_func))
