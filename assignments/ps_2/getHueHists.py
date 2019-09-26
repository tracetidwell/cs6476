import numpy as np
import matplotlib.pyplot as plt
from quantizeRGB import quantizeRGB

def getHueHists(im, k):

    w, h, c = im.shape
    hsv_im = rgb2hsv(im)
    segmented_hue, meanHues = quantizeRGB(hsv_im[:, :, 0].reshape(-1, 1), k)
    segmented_hsv_im = hsv_im.copy()
    segmented_hsv_im[:, :, 0] = segmented_hue.reshape(w, h)

    histEqual = np.histogram(segmented_hue, k)
    histClustered = np.histogram(segmented_hue, sorted(meanHues.reshape(-1)))

    fig = plt.figure(figsize=(10,4))

    ax1 = fig.add_subplot(121, title='Equal Bins')#, xticks=[], yticks=[])
    plt.subplot(ax1)
    ax1.hist(segmented_hue, k)

    ax2 = fig.add_subplot(122, title='Cluster Bins')#, xticks=[], yticks=[])
    plt.subplot(ax2)
    bins = np.concatenate([sorted(meanHues.reshape(-1)), [255]]).astype(int)
    ax2.hist(segmented_hue, bins)

    return histEqual, histClustered
