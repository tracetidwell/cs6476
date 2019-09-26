import numpy as np
from energy_image import energy_image
from cumulative_minimum_energy_map import cumulative_minimum_energy_map
from find_optimal_vertical_seam import find_optimal_vertical_seam

def reduceWidth(im, energyImage, function='gradient'):

    verticalCumulativeEnergyMap = cumulative_minimum_energy_map(energyImage, 'VERTICAL')
    verticalSeam = find_optimal_vertical_seam(verticalCumulativeEnergyMap)
    mask = np.ones((im.shape[0], im.shape[1]), dtype=np.bool)
    mask[np.arange(len(verticalSeam)), verticalSeam] = False

    reducedColorImage = im[mask].reshape(im.shape[0], -1, 3)
    #reducedEnergyImage = energyImage[mask].reshape(energyImage.shape[0], -1)
    reducedEnergyImage = energy_image(reducedColorImage, function)

    return reducedColorImage, reducedEnergyImage
