import numpy as np

def find_optimal_vertical_seam(cumulativeEnergyMap):

    rows, cols = cumulativeEnergyMap.shape
    verticalSeam = np.zeros(cumulativeEnergyMap.shape[0])
    col = np.argmin(cumulativeEnergyMap[-1])
    verticalSeam[-1] = col

    for i in range(cumulativeEnergyMap.shape[0]-2, -1, -1):
        verticalSeam[i] = col - 1 + np.argmin(cumulativeEnergyMap[i, int(max(0, col-1)):int(min(col+2, cols))])
        col = verticalSeam[i]

    return verticalSeam.astype(np.int)
