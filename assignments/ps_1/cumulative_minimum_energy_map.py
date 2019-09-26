import numpy as np
from scipy.ndimage.filters import minimum_filter1d

def cumulative_minimum_energy_map(energyImage, seamDirection):

    cumulativeEnergyMap = energyImage.copy()
    rows, cols = energyImage.shape

    if seamDirection == 'VERTICAL':
        for i in range(1, rows):
            cumulativeEnergyMap[i] += minimum_filter1d(cumulativeEnergyMap[i-1], 3, -1)

    elif seamDirection == 'HORIZONTAL':
        for j in range(1, cols):
            cumulativeEnergyMap[:, j] += minimum_filter1d(cumulativeEnergyMap[:, j-1], 3, 0)

    return cumulativeEnergyMap
