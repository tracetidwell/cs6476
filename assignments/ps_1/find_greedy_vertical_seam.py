import numpy as np

def find_greedy_vertical_seam(energyImage):

    seam = np.zeros(energyImage.shape[0])
    rows, cols = energyImage.shape
    seam[0] = col = np.argmin(energyImage[0, :])

    for i in range(1, len(seam)):
        seam[i] = col - 1 + np.argmin(energyImage[i, int(max(0, col-1)):int(min(col+2, cols))])
        col = seam[i]

    return seam
