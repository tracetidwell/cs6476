import numpy as np
import matplotlib.pyplot as plt

def displaySeam(im, seam, seamDirection):

    fig, ax = plt.subplots()
    ax.imshow(im, extent=[0, im.shape[1], 0, im.shape[0]])

    if seamDirection == 'VERTICAL':
        ax.plot(seam[::-1], range(len(seam)), '-', linewidth=1, color='firebrick')
    elif seamDirection == 'HORIZONTAL':
        ax.plot(range(len(seam)), im.shape[0]-seam, '-', linewidth=1, color='firebrick')
