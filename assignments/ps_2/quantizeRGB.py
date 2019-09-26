import numpy as np


def quantizeRGB(origImg, k):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - r x c x ch
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    """

    three_dim = False
    epsilon = 1

    if len(origImg.shape) == 3:
        three_dim = True
        x, y, z = origImg.shape
        origImg = origImg.reshape(-1, 3)

    outputImg = np.zeros((origImg.shape))
    meanColors = origImg[np.random.choice(range(origImg.shape[0]), k, replace=False)]

    while epsilon > 0:
        clusters = np.argmin(np.sum((origImg - meanColors[:, None])**2, axis=2), axis=0)
        newMeanColors = np.array([np.mean(origImg[clusters==i], axis=0) for i in range(k)])
        #new_means, clusters = k_means_step(image_values, k, means)
        epsilon = np.sqrt(np.square(meanColors - newMeanColors).sum(axis=1)).sum()
        meanColors = newMeanColors

    for i in range(k):
        outputImg[clusters==i] = meanColors[i]

    if three_dim:
        outputImg = outputImg.reshape(x, y, z)

    return outputImg.astype('uint8'), meanColors
