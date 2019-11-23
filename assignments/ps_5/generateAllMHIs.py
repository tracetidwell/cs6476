import os
import glob
import argparse
import numpy as np

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='PS5_Data', help='Base directory for action sequences')
    parser.add_argument('--threshold', type=int, default=5000, help='Threshold to use in background subtraction')
    parser.add_argument('--save', action='store_true', help='Flag for whether to save MHIs')
    parser.add_argument('--save_path', type=str, default='allMHIs.py', help='Path to save MHIs to')
    parser.add_argument('--compute_func', type=str, default='computeMHI', help='String with name of function to be used for computing MHI')
    args = parser.parse_args()

    mhis = generateAllMHIs(args.base_dir, args.threshold, args.save, args.save_path, eval(args.compute_func))
