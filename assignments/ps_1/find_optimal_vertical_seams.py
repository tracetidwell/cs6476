import numpy as np

def find_optimal_vertical_seams(cumulativeEnergyMap, k_seams):

    rows, cols = cumulativeEnergyMap.shape
    v_seams = np.zeros((rows, k_seams)).astype(int)
    top_k = np.argsort(cumulativeEnergyMap[-1])[:k_seams]
    v_seams[-1, :] = top_k

    for j in range(v_seams.shape[1]):

        col = v_seams[-1, j]

        for i in range(rows-2, -1, -1):

            v_seams[i, j] = col - 1 + np.argmin(cumulativeEnergyMap[i, int(max(0, col-1)):int(min(col+2, cols))])
            col = v_seams[i, j]

    return v_seams
