import numpy as np

def find_greedy_horizontal_seam(energyImage):

    seam = np.zeros(energyImage.shape[1])
    rows, cols = energyImage.shape
    seam[0] = row = np.argmin(energyImage[:, 0])

    for j in range(1, len(seam)):
        seam[j] = row - 1 + np.argmin(energyImage[int(max(0, row-1)):int(min(row+2, rows)), j])
        row = seam[j]

    return seam
