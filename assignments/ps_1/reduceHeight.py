import numpy as np
from energy_image import energy_image
from cumulative_minimum_energy_map import cumulative_minimum_energy_map
from find_optimal_horizontal_seam import find_optimal_horizontal_seam

def reduceHeight(im, energyImage, function='gradient'):

    horizontalCumulativeEnergyMap = cumulative_minimum_energy_map(energyImage, 'HORIZONTAL')
    horizontalSeam = find_optimal_horizontal_seam(horizontalCumulativeEnergyMap)
    mask = np.ones((im.shape[1], im.shape[0]), dtype=np.bool)
    mask[np.arange(len(horizontalSeam)), horizontalSeam] = False

    reducedColorImage = np.transpose(im, (1, 0, 2))[mask].reshape(im.shape[1], -1, 3)
    #reducedEnergyImage = np.transpose(energyImage)[mask].reshape(energyImage.shape[1], -1)
    reducedEnergyImage = energy_image(reducedColorImage, function)

    return np.transpose(reducedColorImage, (1, 0, 2)), np.transpose(reducedEnergyImage)
