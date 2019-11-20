import joblib
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def create_idf(model, sift_dir, save=False, save_path='idf'):

    sift_dir='PS4SIFT/sift/'
    mat_paths = glob.glob(sift_dir + '*.mat')

    doc_freq_counts = {i: 0 for i in range(model.n_clusters)}
    doc_count = 0

    for mat_path in mat_paths:

        mat = scipy.io.loadmat(mat_path, verify_compressed_data_integrity=False)

        try:
            for i in set(model.predict(mat['descriptors'])):
                doc_freq_counts[i] += 1
        except ValueError:
            continue

        doc_count += 1

    if save:
        joblib.dump(idf, f'save_path_k={model.n_clusters}.pkl')

    return {key: np.log(doc_count/(1 + value)) for key, value in doc_freq_counts.items()}


def create_stop_list(model, pct, save=False, save_path='stop_list'):

    label_count = Counter(model.labels_)
    multiple = st.norm.ppf((1 - pct) / 2 + pct)
    cutoff = np.mean(list(label_count.values())) + multiple * np.std(list(label_count.values()))

    if save:
        joblib.dump(idf, f'save_path_{pct}.pkl')

    return [i for i in label_count.keys() if label_count[i] > cutoff]


def bag_of_words_histogram(model, descriptors):

    return np.histogram(model.predict(descriptors), range(model.n_clusters+1))


def calc_similarity_score(hist1, hist2):

    return np.dot(hist1, hist2) / (np.sqrt(np.sum(hist1**2)) * np.sqrt(np.sum(hist2**2)))


def get_rows_cols(n):

    cols = [i for i in range(1, int(np.sqrt(n)) + 1) if n % i == 0][-1]
    rows = n // cols

    return rows, cols


def plot_images_grid(images, names, rows, cols, cmap=None, title='frame_query', **kwargs):

    for i, image_row in enumerate(images):

        fig, axes = plt.subplots(nrows=rows, ncols=cols)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        fig.subplots_adjust(hspace=0.25)
        #fig.suptitle(f'{n_patches} Patches from #{i+1} Most Common Word')

        #all_frames = [qry_frames[i][-27:-4]] + top_images[i]
        #print([qry_frames[i]][-27:-4] + top_images[i])
        for j, ax, im, name in zip(range(rows*cols), axes.flatten(), image_row, names[i]):
            if j == 0 and kwargs.get('rois'):
                #ax2 = plt.gca()
                #ax2.add_line(kwargs['rois'][i])
                #plt.draw()
                ax.add_line(kwargs['rois'][i])

            #im = io.imread(frames_dir + im_file)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name)
            ax.imshow(im, cmap=cmap)

        plt.savefig(f'{title}_grid_{i+1}_{len(image_row)}_matches.png', bbox_inches='tight')
        plt.show()


    # for i in range(top_n):
    #
    #     fig, axes = plt.subplots(nrows=rows, ncols=cols)
    #     fig.set_figheight(15)
    #     fig.set_figwidth(15)
    #     fig.subplots_adjust(hspace=0.5)
    #     fig.suptitle(f'{n_patches} Patches from #{i+1} Most Common Word')
    #
    #     for ax, patch in zip(axes.flatten(), patches[i]):
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         ax.imshow(patch, cmap='gray')
    #
    #     plt.show()
    #     plt.savefig(f'num{i}_word_{n_patches}_patches.png')
