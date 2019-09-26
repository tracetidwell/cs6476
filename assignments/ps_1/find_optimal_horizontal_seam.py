import numpy as np

def find_optimal_horizontal_seam(cumulativeEnergyMap):

    rows, cols = cumulativeEnergyMap.shape
    horizontalSeam = np.zeros(cumulativeEnergyMap.shape[1])
    row = np.argmin(cumulativeEnergyMap[:, -1])
    horizontalSeam[-1] = row

    for j in range(cumulativeEnergyMap.shape[1]-2, -1, -1):
        horizontalSeam[j] = row - 1 + np.argmin(cumulativeEnergyMap[int(max(0, row-1)):int(min(row+2, rows)), j])
        row = horizontalSeam[j]

    return horizontalSeam.astype(np.int)
