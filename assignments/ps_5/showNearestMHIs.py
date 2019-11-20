import heapq
import numpy as np
import matplotlib.pyplot as plt
from utils import normalized_euclidean_distance, get_rows_cols


def showNearestMHIs(testMoments, testIdx, trainMoments, trainLabels, trainIdxs,
                    mhis, k=4, displayIm=True, saveIm=False):

    dists = []
    variances = np.std(trainMoments, axis=0)**2

    for train_example, train_idx in zip(trainMoments, trainIdxs):

        heapq.heappush(dists, (normalized_euclidean_distance(train_example.reshape(1, -1), testMoments, variances), train_idx))

        # if len(dists) < 4:
        #     heapq.heappush(dists, (normalized_euclidean_distance(train_example.reshape(1, -1), testMoments, variances), train_idx))
        # else:
        #     heapq.heappushpop(dists, (normalized_euclidean_distance(train_example.reshape(1, -1), testMoments, variances), train_idx))

    top_k = [heapq.heappop(dists) for _ in range(k)]
    top_k.insert(0, (0., testIdx))
    names = ['Original Sequence'] + [f'Number {i} match' for i in range(1, k+1)]

    n = k + 1
    if n % 2 != 0:
        n += 1

    rows, cols = get_rows_cols(n)

    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    fig.set_figheight(15)
    fig.set_figwidth(10)
    if k % 2 == 0:
        fig.delaxes(axes[-1, -1])
    #fig.subplots_adjust(hspace=0.25)
    #fig.suptitle(f'{n_patches} Patches from #{i+1} Most Common Word')

    #all_frames = [qry_frames[i][-27:-4]] + top_images[i]
    #print([qry_frames[i]][-27:-4] + top_images[i])
    for ax, match, name in zip(axes.flatten(), top_k, names):

        #im = io.imread(frames_dir + im_file)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name + f': NED = {round(match[0], 3)}')
        ax.imshow(mhis[:, :, int(match[1])])

    if saveIm:
        plt.savefig(f'Top_{k}_matches_grid.png', bbox_inches='tight')
        #plt.savefig(f'{title}_grid_{i+1}_{len(image_row)}_matches.png', bbox_inches='tight')

    if displayIm:
        plt.show()
