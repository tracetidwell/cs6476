import argparse
import joblib
import glob
import numpy as np
import scipy.io
from skimage import io
import matplotlib.pyplot as plt
from selectRegion import roipoly
from sklearn.cluster import MiniBatchKMeans

from utils import bag_of_words_histogram, calc_similarity_score, get_rows_cols, plot_images_grid


def region_queries(qry_frames=None, qry_regions=None, n_regions=4, top_m=5, model_name='kmeans_k=1500.sav',
                   frames_dir='PS4Frames/frames/', sift_dir='PS4SIFT/sift/', seed=None):

    model = joblib.load(model_name)
    mat_paths = glob.glob(sift_dir + '*.mat')
    top_scores = np.zeros((n_regions, top_m))
    top_image_names = [['none']*top_m for _ in range(n_regions)]
    rois = []
    #my_roi = roipoly(roicolor='r')

    if seed:
        np.random.seed(seed)

    if qry_frames is None:

        qry_regions = []
        frame_idxs = np.random.choice(len(mat_paths), n_regions, replace=False)
        qry_frames = [mat_paths[idx] for idx in frame_idxs]

    for i, qry_frame in enumerate(qry_frames):

        print(f'Finding matches for regions {i+1}...')

        try:
            mat = scipy.io.loadmat(qry_frame, verify_compressed_data_integrity=False)
        except ValueError:
            continue

        im = io.imread(frames_dir + qry_frame[-27:-4])
        plt.imshow(im)
        my_roi = roipoly(roicolor='r')
        qry_regions.append(my_roi.getIdx(im, mat['positions']))
        rois.append(my_roi.getROI())

        try:
            hist1, bins1 = bag_of_words_histogram(model, mat['descriptors'][qry_regions[i]])
        except ValueError:
            continue

        for mat_path in mat_paths:

            if mat_path == qry_frame:
                continue

            try:
                mat = scipy.io.loadmat(mat_path, verify_compressed_data_integrity=False)
            except ValueError:
                continue
            try:
                hist2, bins2 = bag_of_words_histogram(model, mat['descriptors'])
            except ValueError:
                continue

            sim_score = calc_similarity_score(hist1, hist2)

            if any(sim_score > top_scores[i]):

                top_scores[i][0] = sim_score
                top_image_names[i][0] = mat_path[-27:-4]
                top_image_names[i] = list(np.array(top_image_names[i])[np.argsort(top_scores[i])])
                #top_image_names[i] = top_image_names[i][np.argsort(top_scores[i])]
                top_scores[i] = np.sort(top_scores[i])

    # top_image_names = [[io.imread(frames_dir + im_file) for im_file in row] for row in top_image_names]
    # qry_images = [[io.imread(frames_dir + qry_frame[-27:-4])] for qry_frame in qry_frames]
    # all_images = [qry_images[i] + top_image_names[i] for i in range(n_regions)]

    top_images = [[io.imread(frames_dir + im_file) for im_file in row] for row in top_image_names]
    qry_images = [[io.imread(frames_dir + qry_frame[-27:-4])] for qry_frame in qry_frames]
    all_images = [qry_images[i] + top_images[i] for i in range(n_regions)]
    top_image_names = [[image[-9:] for image in row] for row in top_image_names]
    all_names = [[qry_frames[i][-13:-4]] + top_image_names[i] for i in range(n_regions)]

    rows, cols = get_rows_cols(top_m + 1)

    print('Plotting patches...')

    plot_images_grid(all_images, all_names, rows, cols, title='region_query', **{'rois': rois})


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--qry_frames', nargs='?', type=int, action='store', help='List of file paths of .mat files to be used')
    parser.add_argument('--qry_regions', nargs='?', type=int, action='store', help='List of indexes that indicate region for corresponding .mat file')
    parser.add_argument('--n_regions', type=int, default=4, help='Number of regions to find matches for')
    parser.add_argument('--top_m', type=int, default=5, help='Number of matches to find for each region')
    parser.add_argument('--model_name', type=str, default='kmeans_k=2000_isr=04_dsr=04.sav', help='Path to clustering model')
    parser.add_argument('--frames_dir', type=str, default='PS4Frames/frames/', help='Path to image frames')
    parser.add_argument('--sift_dir', type=str, default='PS4SIFT/sift/', help='Path to SFIT data')
    parser.add_argument('--seed', type=int, default=1111, help='Random seed for replicating results')
    args = parser.parse_args()

    region_queries(args.qry_frames, args.qry_regions, args.n_regions, args.top_m, args.model_name,
                   args.frames_dir, args.sift_dir, args.seed)
