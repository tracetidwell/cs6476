import numpy as np
import datetime
import scipy.io
import argparse
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
from sklearn.cluster import MiniBatchKMeans
import joblib
from collections import Counter
import operator

from utils import bag_of_words_histogram, calc_similarity_score, get_rows_cols, plot_images_grid


def visualize_vocabulary(k=1500, im_sample_rate=0.25, desc_sample_rate=0.25,
                         top_n=2, n_patches=25, save_model=True, save_path='',
                         load_model=False, load_path='', frames_dir='PS4Frames/frames/',
                         sift_dir='PS4SIFT/sift/', seed=None):

    fnames = glob.glob(sift_dir + '*.mat')
    fnames = [i[-27:] for i in fnames]
    n_imgs = len(fnames)
    n_sampled_imgs = int(n_imgs * im_sample_rate)
    img_idxs = np.random.choice(n_imgs, n_sampled_imgs, replace=False)
    descriptors = np.zeros((1, 128))
    orients = np.zeros((1, 1))
    positions = np.zeros((1, 2))
    scales = np.zeros((1, 1))
    images = []

    if seed:
        np.random.seed(seed)

    print('Sampling descriptors...')

    for idx in img_idxs:
        fname = sift_dir + fnames[idx]
        try:
            mat = scipy.io.loadmat(fname, verify_compressed_data_integrity=False)
        except ValueError:
            continue
        n_desc = len(mat['descriptors'])
        n_sampled_desc = int(n_desc * desc_sample_rate)
        desc_idxs = np.random.choice(n_desc, n_sampled_desc, replace=False)
        descriptors = np.append(descriptors, mat['descriptors'][desc_idxs], axis=0)
        orients = np.append(orients, mat['orients'][desc_idxs], axis=0)
        positions = np.append(positions, mat['positions'][desc_idxs], axis=0)
        scales = np.append(scales, mat['scales'][desc_idxs], axis=0)
        images += [fnames[idx][:-4]] * n_sampled_desc

    descriptors = descriptors[1:]
    orients = orients[1:]
    positions = positions[1:]
    scales = scales[1:]
    images = np.array(images)

    if load_model:

        print('Loading model...')

        model = joblib.load(load_path)
        labels = model.predict(descriptors)

    else:

        print('Training model...')

        model = MiniBatchKMeans(n_clusters=k, init_size=3*k, random_state=0).fit(descriptors)
        labels = model.labels_

        if save_model:
            joblib.dump(model, save_path)

    label_count = Counter(labels)
    #filter = np.mean(list(label_count.values())) + 3 * np.std(list(label_count.values()))
    #filtered_labels = {key: value for key, value in label_count.items() if value < filter}
    #top_items = sorted(filtered_labels.items(), key=operator.itemgetter(1), reverse=True)[:top_n]
    top_items = [sorted(label_count.items(), key=operator.itemgetter(1), reverse=True)[n-1] for n in top_n]
    top_labels = [label for label, _ in top_items]
    top_orients = [orients[labels==label] for label in top_labels]
    top_positions = [positions[labels==label] for label in top_labels]
    top_scales = [scales[labels==label] for label in top_labels]
    top_images = [images[labels==label] for label in top_labels]
    patch_idxs = [np.random.choice(len(top_images[i]), n_patches) for i in range(len(top_n))]

    patches = [[] for i in range(len(top_n))]
    names = [[] for i in range(len(top_n))]

    print('Plotting patches...')

    for i in range(n_patches):

        for n in range(len(top_n)):

            imname = frames_dir + top_images[n][patch_idxs[n][i]]
            im = io.imread(imname)
            img_patch = getPatchFromSIFTParameters(top_positions[n][patch_idxs[n][i]], top_scales[n][patch_idxs[n][i]],
                                                   top_orients[n][patch_idxs[n][i]], rgb2gray(im))
            patches[n].append(img_patch)
            names[n].append(top_images[n][patch_idxs[n][i]][-9:])

    rows, cols = get_rows_cols(n_patches)

    plot_images_grid(patches, names, rows, cols, cmap='gray', title='visualize_vocab')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=1500, help='Number of clusters to use')
    parser.add_argument('--im_sample_rate', type=float, default=0.25, help='Rate at which to sample frames')
    parser.add_argument('--desc_sample_rate', type=float, default=0.25, help='Rate at which to sample descriptors from a frame')
    parser.add_argument('--top_n', nargs='*', type=int, action='store', default=[1, 2],
                        help='List of images to be warped')
    #parser.add_argument('--top_n', type=int, default=2, help='Number of top words to use for analysis')
    parser.add_argument('--n_patches', type=int, default=25, help='Number of patches to extract for a given word for analysis')
    parser.add_argument('--save_model', action='store_true', help='Flag for whether to save model')
    parser.add_argument('--save_path', type=str, default='model.sav', help='Path to save model if flag is True')
    parser.add_argument('--load_model', action='store_true', help='Flag whether to load model')
    parser.add_argument('--load_path', type=str, default='model.sav', help='Path to load model if flag is True')
    parser.add_argument('--frames_dir', type=str, default='PS4Frames/frames/', help='Path to image frames')
    parser.add_argument('--sift_dir', type=str, default='PS4SIFT/sift/', help='Path to SFIT data')
    parser.add_argument('--seed', type=int, default=24, help='Random seed for replicating results')
    args = parser.parse_args()

    print(args)

    visualize_vocabulary(args.k, args.im_sample_rate, args.desc_sample_rate,
                         args.top_n, args.n_patches, args.save_model, args.save_path,
                         args.load_model, args.load_path, args.frames_dir, args.sift_dir, args.seed)
